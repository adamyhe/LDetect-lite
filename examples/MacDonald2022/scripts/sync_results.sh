#!/bin/bash
# Copy MacDonald2022 per-chromosome pipeline outputs, skipping the large
# HDF5 covariance partition archives (results/{block_set}/chr*/chr*/*.h5).
#
# Keeps: vector-*.txt.gz (correlation-sum vectors), breakpoints-*.json,
# *_partitions.txt, *-ld-blocks*.bed, and anything else under results/
# that isn't a covariance archive.
#
# Usage:
#   ./scripts/sync_results.sh SOURCE DEST
#
# SOURCE/DEST may be local paths or rsync remote specs (e.g.
# user@host:/path/to/examples/MacDonald2022/results/). Trailing slash on
# SOURCE matters as usual for rsync (copies contents, not the dir itself).

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 SOURCE DEST" >&2
    exit 1
fi

SOURCE="$1"
DEST="$2"

rsync -avh --progress \
    --exclude='*.h5' \
    "$SOURCE" "$DEST"
