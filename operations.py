import argparse
import json
import sys
import time

from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential


TERMINAL = {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}


def main():
    parser = argparse.ArgumentParser(description="Inspect, download, or cancel an AML job.")
    parser.add_argument("action", choices=("status", "logs", "download", "cancel"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--managed-identity-client-id")
    parser.add_argument("--output-name")
    parser.add_argument("--download-path", default="./outputs")
    args = parser.parse_args()

    client = MLClient(
        ManagedIdentityCredential(client_id=args.managed_identity_client_id),
        args.subscription_id,
        args.resource_group,
        args.workspace,
    )

    if args.action == "logs":
        client.jobs.stream(args.name)
        return
    if args.action == "download":
        client.jobs.download(
            name=args.name,
            output_name=args.output_name,
            download_path=args.download_path,
        )
        return
    if args.action == "cancel":
        job = client.jobs.get(args.name)
        status = getattr(job.status, "value", job.status)
        if status not in TERMINAL:
            client.jobs.begin_cancel(args.name).result(timeout=60)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                job = client.jobs.get(args.name)
                status = getattr(job.status, "value", job.status)
                if status in TERMINAL:
                    break
                time.sleep(5)
        if status not in TERMINAL:
            raise TimeoutError(f"Job {args.name} did not reach a terminal state.")
    else:
        job = client.jobs.get(args.name)
        status = getattr(job.status, "value", job.status)

    print(json.dumps({"name": job.name, "id": job.id, "status": status}))
    if args.action == "cancel" and status not in {"Canceled", "Cancelled"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
