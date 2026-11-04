# Azure ML attached Kubernetes training samples

Small Azure Machine Learning SDK v2 examples for an **existing, training-enabled
`KubernetesCompute` attachment** to AKS. These are generic samples, not deployment
automation, architecture documentation, benchmarks, or qualification results.
They do not provision resources, attach compute, or change permissions.

## Choose an example

| Sample | Purpose | Default bound |
|---|---|---|
| [train](samples/jobs/train.yml) | Synthetic linear regression, metrics, and JSON checkpoint | 5 epochs, 1 instance, 300-second timeout |
| [pipeline](samples/jobs/pipeline.yml) | Pass a small artifact from prepare to consume | 2 steps, 1 instance per step, 300 seconds per step |
| [resume](samples/jobs/resume.yml) | Continue the same synthetic input from a saved checkpoint | Target epoch 8, 1 instance, 300-second timeout |
| [cancel](samples/jobs/cancel.yml) | Hold a worker so cancellation can be requested | 240-second hold, 300-second timeout |
| [GPU smoke](samples/jobs/gpu-smoke.yml) | One 32-by-32 CUDA matrix operation | 1 instance, 300-second timeout |

`samples/` contains independently authored adaptations, **not Microsoft-published
samples**. The CPU examples use the included [synthetic input](samples/data/training.json);
no real training data is included. Timeouts bound job execution, not queue waiting,
artifact transfer, or end-to-end billing.

`upstream/` contains selected, byte-for-byte original source files from
[`Azure/azureml-examples` at `27f4487a6060b80982ef0793e6ebcea90f509d37`](https://github.com/Azure/azureml-examples/tree/27f4487a6060b80982ef0793e6ebcea90f509d37).
See [provenance and scope](upstream/SOURCES.md) and the [upstream MIT license](upstream/LICENSE).
The selected originals are references, not a complete runnable upstream checkout.

## Prerequisites

Use Python 3.10+ supported by the selected Azure SDK version on an authorized
Azure-hosted runner with an approved managed identity. The runner must reach the
workspace and required data/artifact endpoints. The job runtime identity also
needs the existing permissions and connectivity required by your attachment.
These scripts use `ManagedIdentityCredential`, not an interactive login or keys.
For a user-assigned managed identity, pass its client ID with `--managed-identity-client-id`.

Before submitting anything, configure your own existing resources:

1. Replace `YOUR_KUBERNETES_COMPUTE` and `YOUR_CPU_INSTANCE_TYPE` in CPU job YAML.
   The instance type must already exist and select the intended pool and limits.
2. Replace `YOUR_CPU_ENVIRONMENT:1` with an existing versioned CPU environment
   containing Python, `mlflow`, and `azureml-mlflow` for AML metric logging.
   These dependencies belong in the job environment, not just the submitting runner.
3. For GPU smoke, replace `YOUR_GPU_INSTANCE_TYPE` and
   `YOUR_CUDA_PYTORCH_ENVIRONMENT:1`. Use an existing CUDA-enabled PyTorch environment,
   compatible host drivers/device plugin, and an instance type with a GPU resource
   limit. One instance does not itself mean one GPU.
4. For resume, replace `YOUR_TRAINING_JOB_NAME` with the exact submitted training
   job name. Wait for a successfully written checkpoint and keep the input unchanged.

Do not substitute legacy `AksCompute`: it is not the attached Kubernetes training
target shown here. No C# compute provisioning operation is required.

## Run from the repository root

Install the submission dependencies in an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `AZURE_SUBSCRIPTION_ID`, `AML_RESOURCE_GROUP`, and `AML_WORKSPACE` in the
runner's process environment to your own resource identifiers. No credentials or
workspace configuration files should be committed.

The following commands **submit jobs or operate on existing jobs**. They incur
normal service/resource usage and must only be run with authorization.

```powershell
$workspaceArgs = @("--subscription-id", $env:AZURE_SUBSCRIPTION_ID, "--resource-group", $env:AML_RESOURCE_GROUP, "--workspace", $env:AML_WORKSPACE)
python submit.py samples\jobs\train.yml @workspaceArgs
python submit.py samples\jobs\pipeline.yml @workspaceArgs
# After updating the training job output reference:
python submit.py samples\jobs\resume.yml @workspaceArgs
# Optional GPU visibility smoke:
python submit.py samples\jobs\gpu-smoke.yml @workspaceArgs
# Optional bounded cancellation worker:
python submit.py samples\jobs\cancel.yml @workspaceArgs
```

Submission prints the actual job name, ID, and status. Substitute that name for
`YOUR_JOB_NAME` below. For a pipeline, inspect its child jobs for step logs and
outputs; the train job's `model_output` is not a pipeline-level output.

```powershell
python operations.py status --name YOUR_JOB_NAME @workspaceArgs
python operations.py logs --name YOUR_JOB_NAME @workspaceArgs
python operations.py download --name YOUR_JOB_NAME --output-name model_output --download-path .\outputs @workspaceArgs
python operations.py cancel --name YOUR_JOB_NAME @workspaceArgs
```

`logs` streams and can wait for completion. `cancel` requests cancellation, then
waits for a terminal AML state with a bounded polling interval; failure to observe
cancellation is reported as an error/nonzero exit. `Canceled` is **not proof that
worker pods or allocated resources have been released**.

## Security and support boundaries

- Keep shared-key access disabled where required; these samples do not recommend
  enabling it. Service/resource-access exceptions do not prove private-endpoint-only
  Storage access, even when public network access is disabled. Validate actual
  endpoints, private DNS, routing, and identity authorization for your own policy.
- Do not use a local-only runtime flag as an AKS fix. Resolve supported attachment,
  runtime, storage, and network configuration instead.
- Attached Kubernetes training-node autoscaling is documented as unsupported.
  Plan capacity explicitly; do not infer support from AKS autoscaler availability.
- Queueing, fairness, precise start-after-release timing, same-node GPU concurrency,
  RTX/vGPU qualification, MIG/fractional GPUs, RDMA, distributed training, inference,
  and production performance are not established by these examples.
- Where pod-placement/release evidence is necessary, use an already authorized,
  least-privilege Kubernetes diagnostic identity with API connectivity. Do not
  grant cluster-admin or extra AKS permissions merely to run a sample.
- No evaluation logs, actual job/node names, private network details, resource
  identifiers, measured costs, or environment-specific proof are included.

References: [attach Kubernetes](https://learn.microsoft.com/azure/machine-learning/how-to-attach-kubernetes-anywhere?view=azureml-api-2),
[Kubernetes compute limitations](https://learn.microsoft.com/azure/machine-learning/how-to-attach-kubernetes-anywhere?view=azureml-api-2#limitations-for-kubernetes-compute-target),
[instance types](https://learn.microsoft.com/azure/machine-learning/how-to-kubernetes-instance-type?view=azureml-api-2),
[Storage network security](https://learn.microsoft.com/azure/storage/common/storage-network-security).

## Local checks only

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

These tests check Python syntax, configuration wiring, synthetic training/resume,
pipeline handoff, and command handling with local mocks. They never authenticate
to Azure, submit jobs, exercise CUDA, or demonstrate deployment compatibility.
