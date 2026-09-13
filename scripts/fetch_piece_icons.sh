#!/usr/bin/env bash
# Piece icons belong to the Intransitive site (meaf.us/rps2); fetched locally, not redistributed here.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets
for p in rock-BtOQJn3v paper-Bt1E0vrq scissors-Bu9hok1o; do
  curl -fsSL -A Mozilla/5.0 "https://meaf.us/rps2/assets/$p.svg" -o "assets/${p%%-*}.svg"
  rsvg-convert -h 256 "assets/${p%%-*}.svg" -o "assets/${p%%-*}.png"
done
echo "icons in assets/"
