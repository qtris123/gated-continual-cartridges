"""Upload a trained cartridge .pt file to a HuggingFace model repo.

Usage:
    python experiments/continual_learning/upload_cartridge_to_hf.py \
        --cartridge-path ./outputs/train_amd_2020/cache_last.pt \
        --hf-repo-id username/amd-2020-cartridge \
        [--private] \
        [--model-name meta-llama/Llama-3.2-3B-Instruct]
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--cartridge-path", required=True, help="Local .pt file")
parser.add_argument("--hf-repo-id", required=True, help="e.g. 'username/amd-2020-cartridge'")
parser.add_argument("--private", action="store_true", default=False)
parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
args = parser.parse_args()

cartridge_path = Path(args.cartridge_path)
assert cartridge_path.exists(), f"Not found: {cartridge_path}"

api = HfApi()
api.create_repo(repo_id=args.hf_repo_id, repo_type="model", private=args.private, exist_ok=True)

# Upload .pt file
api.upload_file(
    path_or_fileobj=str(cartridge_path),
    path_in_repo=cartridge_path.name,
    repo_id=args.hf_repo_id,
    repo_type="model",
    commit_message=f"Upload cartridge: {cartridge_path.name}",
)

# Upload model card
model_card = f"""---
base_model: {args.model_name}
tags: [cartridge, continual-cartridges]
---
# Cartridge: {args.hf_repo_id}
Base model: `{args.model_name}`

## Usage
```bash
python experiments/continual_learning/synthesize_data_with_cartridge.py \\
    --company <COMPANY> --year <YEAR> \\
    --cartridge-hf-id {args.hf_repo_id}
```
"""
api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=args.hf_repo_id,
    repo_type="model",
    commit_message="Add model card",
)

print(f"Uploaded to: https://huggingface.co/{args.hf_repo_id}")
