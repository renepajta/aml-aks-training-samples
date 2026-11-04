import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job environment.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device is visible to this job.")

    matrix = torch.ones((32, 32), device="cuda:0")
    checksum = int((matrix @ matrix).sum().item())
    result = {
        "cudaAvailable": torch.cuda.is_available(),
        "visibleGpuCount": torch.cuda.device_count(),
        "gpuName": torch.cuda.get_device_name(0),
        "matrixSize": 32,
        "checksum": checksum,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gpu-smoke.json").write_text(json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
