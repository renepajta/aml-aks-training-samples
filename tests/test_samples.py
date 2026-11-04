import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml
from azure.ai.ml import load_job


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigurationTests(unittest.TestCase):
    def test_python_syntax(self):
        paths = [ROOT / "submit.py", ROOT / "operations.py"]
        paths += list((ROOT / "samples").rglob("*.py"))
        paths += list((ROOT / "upstream").rglob("*.py"))
        for path in paths:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_job_configuration_and_local_sdk_loading(self):
        paths = list((ROOT / "samples" / "jobs").glob("*.yml"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(path=path.name):
                document = yaml.safe_load(path.read_text())
                job = load_job(source=path)
                self.assertEqual(job.type, document["type"])
                nodes = document.get("jobs", {"command": document})
                for node in nodes.values():
                    self.assertEqual(
                        node["compute"], "azureml:YOUR_KUBERNETES_COMPUTE"
                    )
                    self.assertEqual(node["resources"]["instance_count"], 1)
                    self.assertLessEqual(node["limits"]["timeout"], 300)
                    self.assertTrue((path.parent / node["code"]).is_dir())
                    for value in node.get("inputs", {}).values():
                        if isinstance(value, dict) and value["path"].startswith(".."):
                            self.assertTrue((path.parent / value["path"]).is_file())
        pipeline = yaml.safe_load(
            (ROOT / "samples/jobs/pipeline.yml").read_text()
        )
        self.assertEqual(
            pipeline["jobs"]["consume"]["inputs"]["prepared"],
            "${{parent.jobs.prepare.outputs.prepared}}",
        )


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mlflow = Mock()
        with patch.dict(sys.modules, {"mlflow": cls.mlflow}):
            cls.workload = load_module("tutorial_job", "samples/src/tutorial_job.py")

    def test_train_resume_matches_uninterrupted_training(self):
        rows, _ = self.workload.load_rows(ROOT / "samples/data/training.json")
        weights, bias, _ = self.workload.train(rows, target_epoch=5)
        checkpoint = {"weights": weights, "bias": bias, "epoch": 5}
        resumed = self.workload.train(rows, checkpoint, target_epoch=8)
        uninterrupted = self.workload.train(rows, target_epoch=8)
        self.assertEqual(resumed[:2], uninterrupted[:2])
        self.assertEqual([x["epoch"] for x in resumed[2]], [6, 7, 8])
        self.assertLess(uninterrupted[2][-1]["train_mse"], uninterrupted[2][0]["train_mse"])
        with self.assertRaises(ValueError):
            self.workload.train(rows, checkpoint, target_epoch=5)

    def test_artifacts_checkpoint_and_pipeline_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                data=ROOT / "samples/data/training.json",
                model_output=root / "model",
                checkpoint_output=root / "checkpoint",
                checkpoint=root / "checkpoint",
                target_epoch=5,
                output=root / "prepared",
            )
            self.workload.run_train(args)
            checkpoint = json.loads((root / "checkpoint/checkpoint.json").read_text())
            self.assertEqual(checkpoint["epoch"], 5)
            args.target_epoch = 8
            self.workload.run_train(args, resume=True)
            model = json.loads((root / "model/model.json").read_text())
            self.assertEqual(model["trainedEpochs"], 8)
            self.workload.prepare(args)
            args.input = root / "prepared"
            args.output = root / "result"
            self.workload.consume(args)
            result = json.loads((root / "result/result.json").read_text())
            self.assertEqual(result["producer"], "consume")
            self.assertEqual(result["rows"], 8)
            self.assertTrue(result["handoffVerified"])
            checkpoint["inputSha256"] = "mismatched-synthetic-input"
            (root / "checkpoint/checkpoint.json").write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.workload.run_train(args, resume=True)

    def test_gpu_smoke_with_mock_cuda_only(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_name.return_value = "synthetic-device"
        torch.ones.return_value.__matmul__ = Mock(return_value=Mock())
        torch.ones.return_value.__matmul__.return_value.sum.return_value.item.return_value = 32768
        with patch.dict(sys.modules, {"torch": torch}):
            module = load_module("gpu_smoke", "samples/src/gpu_smoke.py")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["gpu_smoke.py", "--output-dir", directory]):
                module.main()
            result = json.loads((Path(directory) / "gpu-smoke.json").read_text())
            self.assertEqual(result["checksum"], 32768)
            self.assertEqual(result["visibleGpuCount"], 1)
        torch.cuda.is_available.return_value = False
        with patch.object(sys, "argv", ["gpu_smoke.py", "--output-dir", "unused"]):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                module.main()


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.operations = load_module("operations", "operations.py")
        self.submit = load_module("submit", "submit.py")
        self.args = [
            "--name", "synthetic-job",
            "--subscription-id", "placeholder-subscription",
            "--resource-group", "placeholder-group",
            "--workspace", "placeholder-workspace",
        ]

    def test_placeholder_submission_fails_before_client_creation(self):
        with patch.object(sys, "argv", [
            "submit.py", str(ROOT / "samples/jobs/train.yml"), *self.args
        ]), patch.object(self.submit, "MLClient") as client:
            with self.assertRaises(SystemExit) as result:
                self.submit.main()
            self.assertEqual(result.exception.code, 2)
            client.assert_not_called()

    def test_cancel_polls_service_state(self):
        client = Mock()
        client.jobs.get.side_effect = [
            SimpleNamespace(name="synthetic-job", id="placeholder-id", status="Running"),
            SimpleNamespace(name="synthetic-job", id="placeholder-id", status="Canceled"),
        ]
        with patch.object(sys, "argv", ["operations.py", "cancel", *self.args]), \
                patch.object(self.operations, "MLClient", return_value=client), \
                patch.object(self.operations, "ManagedIdentityCredential"):
            self.operations.main()
        client.jobs.begin_cancel.assert_called_once_with("synthetic-job")
        client.jobs.begin_cancel.return_value.result.assert_called_once_with(timeout=60)

    def test_cancel_does_not_report_completion_as_cancellation(self):
        client = Mock()
        client.jobs.get.return_value = SimpleNamespace(
            name="synthetic-job", id="placeholder-id", status="Completed"
        )
        with patch.object(sys, "argv", ["operations.py", "cancel", *self.args]), \
                patch.object(self.operations, "MLClient", return_value=client), \
                patch.object(self.operations, "ManagedIdentityCredential"):
            with self.assertRaises(SystemExit) as result:
                self.operations.main()
        self.assertEqual(result.exception.code, 1)
        client.jobs.begin_cancel.assert_not_called()

    def test_cancel_times_out_explicitly(self):
        client = Mock()
        client.jobs.get.return_value = SimpleNamespace(status="Running")
        with patch.object(sys, "argv", ["operations.py", "cancel", *self.args]), \
                patch.object(self.operations, "MLClient", return_value=client), \
                patch.object(self.operations, "ManagedIdentityCredential"), \
                patch.object(self.operations.time, "monotonic", side_effect=[0, 301]):
            with self.assertRaises(TimeoutError):
                self.operations.main()


if __name__ == "__main__":
    unittest.main()
