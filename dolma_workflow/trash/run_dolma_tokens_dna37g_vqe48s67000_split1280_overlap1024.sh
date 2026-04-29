dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_dna_037g/jsonlgz_vqe48s67000_split1280_overlap1024/validation/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_dna_037g/tokens_vqe48s67000_split1280_overlap1024/validation \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
    
dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_dna_037g/jsonlgz_vqe48s67000_split1280_overlap1024/test/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_dna_037g/tokens_vqe48s67000_split1280_overlap1024/test \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_dna_037g/jsonlgz_vqe48s67000_split1280_overlap1024/train/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_dna_037g/tokens_vqe48s67000_split1280_overlap1024/train \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
