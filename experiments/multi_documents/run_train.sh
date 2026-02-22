#!/bin/bash
set -e

echo "=== Starting AMD Train ==="
python experiments/multi_documents/train_amd.py

# echo "=== Starting PepsiCo Train ==="
# python experiments/multi_documents/train_pepsi.py



echo "=== All Train complete ==="
