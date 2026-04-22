"""Upload a synthesized Q&A .parquet dataset to a HuggingFace dataset repo.

Usage:
    python experiments/utils/upload_dataset_to_hf.py \
        --dataset-path ./outputs/amd_2021/dataset.parquet \
        --hf-repo-id username/amd-2021-dataset \
        [--private] \
        [--split train]
"""

import argparse
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", required=True, help="Local .parquet file")
parser.add_argument("--hf-repo-id", required=True, help="e.g. 'username/amd-2021-dataset'")
parser.add_argument("--private", action="store_true", default=False)
parser.add_argument("--split", default="train", help="Dataset split name")
args = parser.parse_args()

dataset_path = Path(args.dataset_path)
assert dataset_path.exists(), f"Not found: {dataset_path}"
assert dataset_path.suffix == ".parquet", f"Expected .parquet file, got: {dataset_path.suffix}"

api = HfApi()
api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", private=args.private, exist_ok=True)

# Load and push dataset
dataset = Dataset.from_parquet(str(dataset_path))
dataset.push_to_hub(
    args.hf_repo_id,
    split=args.split,
    private=args.private,
    commit_message=f"Upload dataset: {dataset_path.name}",
)
print(f"Uploaded dataset ({len(dataset)} rows, split='{args.split}')")

# Upload config.yaml from the same directory
config_path = dataset_path.parent / "config.yaml"
if config_path.exists():
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.yaml",
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        commit_message="Add training config",
    )
    print(f"Uploaded config.yaml")
else:
    print(f"Warning: No config.yaml found at {config_path}")

# Upload dataset card
dataset_card = f"""---
tags: [continual-cartridges, self-study-synthesis]
task_categories: [question-answering]
---
# Dataset: {args.hf_repo_id}

## Usage
```python
from datasets import load_dataset
ds = load_dataset("{args.hf_repo_id}")
```
"""
api.upload_file(
    path_or_fileobj=dataset_card.encode(),
    path_in_repo="README.md",
    repo_id=args.hf_repo_id,
    repo_type="dataset",
    commit_message="Add dataset card",
)

print(f"Uploaded to: https://huggingface.co/datasets/{args.hf_repo_id}")
