#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: scripts/reproduce.sh MODEL [OUT_ROOT]"
    exit 2
fi

RAZOR_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
razor sweep \
    --model "$1" \
    --out "${2:-outputs/grid}" \
    --data "${DATA:-$RAZOR_HOME/data/RazorCal.json}" \
    --methods "${METHODS:-razor,reap,ean,frequency}" \
    --ratios "${RATIOS:-0.25,0.5,0.75}" \
    --max-len "${MAX_LEN:-32768}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --num-batches "${NUM_BATCHES:--1}"
