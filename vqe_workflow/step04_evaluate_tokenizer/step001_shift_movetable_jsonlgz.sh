#!/usr/bin/env bash
set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# INPUT_PATH="${1:-/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe342s036000l1.jsonl.gz}"
# OUTPUT_DIR="${2:-$SCRIPT_DIR}"



PYTHON_BIN="${PYTHON_BIN:-python3}"

FILE_PATTERN="LB07"

INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_${FILE_PATTERN}.modified.reformed.jsonl.gz"
# --- 初始变量 ---
SHIFT_VALUE=4

# 预定义基础路径（不包含 r/l 标识的部分）
BASE_OUTPUT="/mnt/zzbnew/poregpt/dnadata/movetable/signal_${FILE_PATTERN}.modified.reformed"
# --- 判断逻辑 ---
if [ "$SHIFT_VALUE" -gt 0 ]; then
    # 正数：使用 r*
    OUTPUT_FILE="${BASE_OUTPUT}.shiftr${SHIFT_VALUE}.jsonl.gz"
elif [ "$SHIFT_VALUE" -lt 0 ]; then
    # 负数：使用 l* (abs取绝对值，避免出现 shiftl-3 的情况)
    ABS_SHIFT=${SHIFT_VALUE#-}
    OUTPUT_FILE="${BASE_OUTPUT}.shiftl${ABS_SHIFT}.jsonl.gz"
else
    # 零的情况：不带位移标识
    OUTPUT_FILE="${BASE_OUTPUT}.jsonl.gz"
fi


if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input file not found: $INPUT_PATH" >&2
  exit 1
fi


python3 scripts/step003_shift_modified_movetable_jsonlgz.py \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_FILE" \
  --shift "$SHIFT_VALUE"
