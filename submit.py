import argparse
import json
import re
from pathlib import Path

from azure.ai.ml import MLClient, load_job
from azure.identity import ManagedIdentityCredential


def main():
    parser = argparse.ArgumentParser(description="Submit an Azure ML SDK v2 job YAML.")
    parser.add_argument("job_yaml")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--name", help="Optional explicit job name.")
    parser.add_argument("--managed-identity-client-id")
    args = parser.parse_args()

    if re.search(r"\bYOUR_[A-Z_]+\b", Path(args.job_yaml).read_text()):
        parser.error("Replace all YOUR_* placeholders in the job YAML before submitting.")
    job = load_job(source=args.job_yaml)
    client = MLClient(
        ManagedIdentityCredential(client_id=args.managed_identity_client_id),
        args.subscription_id,
        args.resource_group,
        args.workspace,
    )
    if args.name:
        job.name = args.name

    submitted = client.jobs.create_or_update(job)
    print(
        json.dumps(
            {
                "name": submitted.name,
                "id": submitted.id,
                "status": getattr(submitted.status, "value", submitted.status),
            }
        )
    )


if __name__ == "__main__":
    main()
