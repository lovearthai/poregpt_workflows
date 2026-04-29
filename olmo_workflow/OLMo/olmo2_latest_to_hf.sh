OLMO_OUTPUT_PATH=output_300m_ctx1280-gbsz4096-lr1e4-vqe342s036000l1
OLMO_OUTPUT_PATH=output_vqe340s147000_cb390k_300m_ctx1280_dna595g
OLMO_OUTPUT_PATH=output_190m_ctx1536-gbsz4096-lr5e5-vqe340s147000l1-mlp2-dim0768-layer32
OLMO_OUTPUT_PATH=output_336m_ctx1536-gbsz4096-lr5e5-vqe340s147000l1-mlp2-dim1024-layer32
OLMO_OUTPUT_PATH=output_402m_ctx1536-gbsz4096-lr1e4-vqe340s147000l1-mlp4-dim1024-layer24
OLMO_OUTPUT_PATH=output_526m_ctx1536-gbsz4096-lr5e5-vqe340s147000l1-mlp2-dim1280-layer32
python3 scripts/convert_olmo2_to_hf.py --input_dir "../$OLMO_OUTPUT_PATH/steps/latest-unsharded" --output_dir "../$OLMO_OUTPUT_PATH/hf_latest" --tokenizer_json_path "olmo_data/tokenizers/pore_1k/tokenizer.json"
