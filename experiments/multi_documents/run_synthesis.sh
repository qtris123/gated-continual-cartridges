#!/bin/bash
set -e

echo "=== Starting AMD synthesis ==="
python experiments/multi_documents/synthesize_amd.py

echo "=== Starting PepsiCo synthesis ==="
python experiments/multi_documents/synthesize_pepsi.py



echo "=== All synthesis complete ==="
