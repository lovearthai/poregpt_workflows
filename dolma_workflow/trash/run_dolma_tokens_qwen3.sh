dolma tokens \
    --documents "/mnt/nas_syy/dataset/huada_rna_80G/liuh/data/dna/chunk_cluster_8192_32_5/cluster_text_split/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/dataset/huada_rna_80G/liuh/data/dna/chunk_cluster_8192_32_5/cluster_text_split_tokens \
    --dtype "uint16" \
    --tokenizer.bos_token_id 8192 \
    --tokenizer.eos_token_id 8193 \
    --tokenizer.pad_token_id 8194 \
    --processes 32
	
