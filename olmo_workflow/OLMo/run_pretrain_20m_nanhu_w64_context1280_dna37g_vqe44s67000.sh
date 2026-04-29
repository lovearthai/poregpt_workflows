#!/bin/bash
# nohup ./run.sh > train.log 2>&1 &
# 参考: https://github.com/allenai/OLMo/blob/bf536fdfb5ab9b77c8defac2d7ca37db05eea733/scripts/augusta/peteish1-anneal.sh

# 如何断点续训: 将如下行替换下面命令中的--load_path="" 
# --load_path="/workspace/zzb_tutorial/olmo20m_pt_output/steps" 
#export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca259235j( j forfor)
export WANDB_API_KEY=PasteYourWandbApikeyHere
export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:29502 \
	scripts/train.py configs/config_20m_vqe44s67000_ctx1280_dna37g.yaml \
	--run_name="olmo-pt-bioseq-20m-dna37g-split1280_overlap1024-vqe44s67000-ctx1280-gbsz4096-lr5e4-1ep" \
        --wandb.entity="jiaoshuaihit-hit" \
        --wandb.project="bioseq-rna-80g" \
        --load_path="" \
        --save_folder="../output_20m_ctx1280-gbsz4096-lr5e4-vqe44s67000/steps/" 
