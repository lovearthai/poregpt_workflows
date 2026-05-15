#!/bin/bash 

pen=50
kernel=linear
max_shift=100


#jsonl_path_in=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/03.token/result_0331/LB07_1/signal_none.jsonl
#jsonl_path_out=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/03.token/result_0331/LB07_1/signal_none.adjusted.jsonl

jsonl_path_in=/mnt/zzbnew/poregpt/dnadata/movetable/LB06/signal_LB06.modified.reformed.shiftr4.mongoq30.vqe340s147000l1.aligned.jsonl.gz
jsonl_path_out=/mnt/zzbnew/poregpt/dnadata/movetable/LB06/signal_LB06.modified.reformed.shiftr4.mongoq30.vqe340s147000l1.aligned.adjusted.jsonl.gz


python3 scripts/step00_adjust_boundaries_jsonl.py \
    --input ${jsonl_path_in} \
    --output ${jsonl_path_out} \
    --kernel ${kernel} \
    --pen ${pen} \
    --mode left \
    --max-shift ${max_shift} \
    --keep-edge-boundaries-fixed \



python3 scripts/step00_span_offset_eval.py \
  --data-jsonl ${jsonl_path_out} \
  --out-dir step00_span_offset_eval/span_${pen}_${kernel}_${max_shift}_stats_4 \
  --span-field base_sample_spans_rel \
  --offsets -4 \
  --limit 10000 \
  --prefix ${pen}_${kernel}_${max_shift}_stats_4



python3 scripts/step00_span_offset_eval.py \
  --data-jsonl ${jsonl_path_out} \
  --out-dir step00_span_offset_eval/span_${pen}_${kernel}_${max_shift}_stats_37 \
  --span-field base_sample_spans_rel \
  --offsets -3.7 \
  --limit 10000 \
  --prefix ${pen}_${kernel}_${max_shift}_stats_37


python3 scripts/step00_span_offset_eval.py \
  --data-jsonl ${jsonl_path_out} \
  --out-dir step00_span_offset_eval/span_${pen}_${kernel}_${max_shift}_stats_auto \
  --span-field base_sample_spans_rel_adj \
  --offsets -4 \
  --limit 10000 \
  --prefix ${pen}_${kernel}_${max_shift}_stats_auto