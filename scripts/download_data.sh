#!/usr/bin/env bash
# Downloads PTB-XL v1.0.3 from PhysioNet into data/raw/
# Freely downloadable, no access request needed: https://physionet.org/content/ptb-xl/1.0.3/
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "Downloading PTB-XL 1.0.3 into $RAW_DIR (this is ~1.7GB zipped)..."
wget -r -N -c -np \
  https://physionet.org/files/ptb-xl/1.0.3/ \
  -P "$RAW_DIR" \
  --cut-dirs=3 -nH

echo "Done. Verify with: ls $RAW_DIR"
echo "You should see ptbxl_database.csv, scp_statements.csv, and records100/ records500/ folders."
