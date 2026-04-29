dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/shared/dataset/dna/dna814g/jsonlgz_vqe_pass25_c256k_cnn3_step22500/split1280_overlap640/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/shared/dataset/dna/dna814g/tokens_vqe_pass25_c256k_cnn3_step22500 \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
    

