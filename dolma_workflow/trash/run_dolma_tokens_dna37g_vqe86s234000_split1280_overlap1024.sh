dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_min0_max2_read96655_p90/memap_lemon5/jsonlgz_vqe86s234000_split1280_overlap1024/validation/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_min0_max2_read96655_p90/memap_lemon5/tokens_vqe86s234000_split1280_overlap1024/validation \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
    
dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_min0_max2_read96655_p90/memap_lemon5/jsonlgz_vqe86s234000_split1280_overlap1024/test/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_min0_max2_read96655_p90/memap_lemon5/tokens_vqe86s234000_split1280_overlap1024/test \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_min0_max2_read96655_p90/memap_lemon5/jsonlgz_vqe86s234000_split1280_overlap1024/train/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_min0_max2_read96655_p90/memap_lemon5/tokens_vqe86s234000_split1280_overlap1024/train \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
