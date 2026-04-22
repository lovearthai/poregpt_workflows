#!/bin/bash

# 默认参数值
INPUT_CSV="token_freq_sort_desc_rsq340s147000_5x4x2_freq_human_dna595g.csv"
OUTPUT_CSV="token_freq_sort_desc_rsq340s147000_5x4x2_freq_human_dna595g_with_fsq_codes.csv"
CODEBOOK_FSQD=5
CODEBOOK_FSQN=4
CODEBOOK_NQTZ=2
DEBUG_FLAG=""


echo "Running script with:"
echo "  Input CSV: $INPUT_CSV"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Codebook FSQD: $CODEBOOK_FSQD"
echo "  Codebook FSQN: $CODEBOOK_FSQN"
echo "  Codebook NQTZ: $CODEBOOK_NQTZ"
if [[ -n "$DEBUG_FLAG" ]]; then
    echo "  Debug Mode: ON"
fi

# 执行 Python 脚本
python step17_tokenid_to_fsqcode.py \
    --input-csv "$INPUT_CSV" \
    --output-csv "$OUTPUT_CSV" \
    --codebook-fsqd $CODEBOOK_FSQD \
    --codebook-fsqn $CODEBOOK_FSQN \
    --codebook-nqtz $CODEBOOK_NQTZ \
    $DEBUG_FLAG

echo "Script execution completed."
