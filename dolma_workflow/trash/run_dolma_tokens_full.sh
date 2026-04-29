dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep22500/validation/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/tokens_vqe_model_pass25_c64k_gstep22500/validation \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
