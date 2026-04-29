dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_dna_152g/memap_lemon5/jsonlgz_vqe84s286000_split1280_overlap256/validation/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_dna_152g/memap_lemon5/tokens_vqe84s286000_split1280_overlap256/validation \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
    
dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_dna_152g/memap_lemon5/jsonlgz_vqe84s286000_split1280_overlap256/test/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_dna_152g/memap_lemon5/tokens_vqe84s286000_split1280_overlap256/test \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
dolma tokens \
    --documents "/mnt/nas_syy/default/poregpt/dataset/human_dna_152g/memap_lemon5/jsonlgz_vqe84s286000_split1280_overlap256/train/*.gz" \
    --tokenizer.name_or_path "tokenizer_bwav/tokenizer.json" \
    --destination /mnt/nas_syy/default/poregpt/dataset/human_dna_152g/memap_lemon5/tokens_vqe84s286000_split1280_overlap256/train \
    --dtype "uint32" \
    --tokenizer.bos_token_id 1 \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id 3 \
    --processes 32
	
