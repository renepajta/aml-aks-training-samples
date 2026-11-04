import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import mlflow


def load_rows(path):
    raw = Path(path).read_bytes()
    rows = json.loads(raw)["rows"]
    if not rows or any(len(row["features"]) != 8 for row in rows):
        raise ValueError("Unexpected synthetic training input shape.")
    return rows, hashlib.sha256(raw).hexdigest()


def train(rows, initial=None, first_epoch=0, target_epoch=5):
    if initial:
        weights = [float(x) for x in initial["weights"]]
        bias = float(initial["bias"])
        first_epoch = int(initial["epoch"])
    else:
        weights = [0.0] * 8
        bias = 0.0
    if target_epoch <= first_epoch:
        raise ValueError("Target epoch must be later than the checkpoint.")
    metrics = []
    for epoch in range(first_epoch + 1, target_epoch + 1):
        wg = [0.0] * 8
        bg = 0.0
        for row in rows:
            features = [float(value) for value in row["features"]]
            error = bias + sum(w * x for w, x in zip(weights, features)) - float(row["target"])
            bg += 2 * error
            for i, value in enumerate(features):
                wg[i] += 2 * error * value
        bias -= 0.001 * bg / len(rows)
        weights = [w - 0.001 * g / len(rows) for w, g in zip(weights, wg)]
        mse = sum(
            (bias + sum(w * float(x) for w, x in zip(weights, row["features"])) - float(row["target"])) ** 2
            for row in rows
        ) / len(rows)
        mlflow.log_metric("train_mse", mse, step=epoch)
        metrics.append({"epoch": epoch, "train_mse": mse})
    return weights, bias, metrics


def run_train(args, resume=False):
    rows, input_hash = load_rows(args.data)
    checkpoint = None
    if resume:
        checkpoint = json.loads((Path(args.checkpoint) / "checkpoint.json").read_text())
        if checkpoint["inputSha256"] != input_hash:
            raise ValueError("Checkpoint does not match the mounted synthetic input.")
    weights, bias, metrics = train(
        rows, checkpoint, target_epoch=args.target_epoch
    )
    os.makedirs(args.model_output, exist_ok=True)
    os.makedirs(args.checkpoint_output, exist_ok=True)
    model = {
        "format": "linear-regression-v1",
        "weights": weights,
        "bias": bias,
        "trainedEpochs": args.target_epoch,
        "inputSha256": input_hash,
    }
    saved_checkpoint = {
        "epoch": args.target_epoch,
        "weights": weights,
        "bias": bias,
        "inputSha256": input_hash,
    }
    Path(args.model_output, "model.json").write_text(json.dumps(model, sort_keys=True))
    checkpoint_path = Path(args.checkpoint_output, "checkpoint.json")
    checkpoint_path.write_text(json.dumps(saved_checkpoint, sort_keys=True))
    Path(args.model_output, "metrics.json").write_text(
        json.dumps({"rows": len(rows), "metrics": metrics}, sort_keys=True)
    )
    mlflow.log_metric("rows", len(rows))
    mlflow.log_artifact(str(checkpoint_path))
    print(json.dumps({"inputRows": len(rows), "inputSha256": input_hash, "metrics": metrics}), flush=True)


def prepare(args):
    rows, input_hash = load_rows(args.data)
    os.makedirs(args.output, exist_ok=True)
    result = {"producer": "prepare", "rows": len(rows), "inputSha256": input_hash}
    Path(args.output, "prepared.json").write_text(json.dumps(result, sort_keys=True))
    mlflow.log_metric("prepared_rows", len(rows))
    print(json.dumps(result), flush=True)


def consume(args):
    prepared = json.loads((Path(args.input) / "prepared.json").read_text())
    if prepared.get("producer") != "prepare":
        raise ValueError("Expected output from the prepare pipeline step.")
    os.makedirs(args.output, exist_ok=True)
    result = {**prepared, "producer": "consume", "handoffVerified": True}
    Path(args.output, "result.json").write_text(json.dumps(result, sort_keys=True))
    mlflow.log_metric("consumed_rows", prepared["rows"])
    print(json.dumps(result), flush=True)


def hold(args):
    import time

    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        print("tutorial cancellation worker active", flush=True)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "resume", "prepare", "consume", "hold"))
    parser.add_argument("--data")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-output")
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--target-epoch", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args()
    if args.action in ("train", "resume"):
        run_train(args, resume=args.action == "resume")
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "consume":
        consume(args)
    else:
        hold(args)


if __name__ == "__main__":
    main()
