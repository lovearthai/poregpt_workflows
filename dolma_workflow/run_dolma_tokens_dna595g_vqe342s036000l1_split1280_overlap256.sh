sleep 7200

dolma tokens \
    --documents "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe342s036000l1_split1280_overlap256/validation/*.gz" \
    --tokenizer.name_or_path "/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_16k/tokenizer.json" \
    --destination /mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/tokens_vqe342s036000l1_split1280_overlap256/validation \
    --dtype "uint16" \
    --tokenizer.pad_token_id 1 \
    --tokenizer.bos_token_id 2 \
    --tokenizer.eos_token_id 3 \
    --processes 32
    
dolma tokens \
    --documents "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe342s036000l1_split1280_overlap256/test/*.gz" \
    --tokenizer.name_or_path "/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_16k/tokenizer.json" \
    --destination /mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/tokens_vqe342s036000l1_split1280_overlap256/test \
    --dtype "uint16" \
    --tokenizer.pad_token_id 1 \
    --tokenizer.bos_token_id 2 \
    --tokenizer.eos_token_id 3 \
    --processes 32
 
dolma tokens \
    --documents "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe342s036000l1_split1280_overlap256/train/*.gz" \
    --tokenizer.name_or_path "/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_16k/tokenizer.json" \
    --destination /mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/tokens_vqe342s036000l1_split1280_overlap256/train \
    --dtype "uint16" \
    --tokenizer.pad_token_id 1 \
    --tokenizer.bos_token_id 2 \
    --tokenizer.eos_token_id 3 \
    --processes 32
 
