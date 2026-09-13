#!/usr/bin/env bash
# Download the MaleCNS v1.0 connectome tables (~1.1 GB) from Janelia FlyEM (CC-BY 4.0).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in \
  body-annotations-male-cns-v1.0-minconf-0.5.feather \
  body-neurotransmitters-male-cns-v1.0.feather \
  connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  if [ -s "data/$f" ]; then echo "have $f"; continue; fi
  echo "downloading $f"
  curl -fL --progress-bar -o "data/$f" "$BASE/$f"
done
