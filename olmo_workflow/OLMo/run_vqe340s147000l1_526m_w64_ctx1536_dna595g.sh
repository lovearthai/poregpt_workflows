#!/bin/bash
# nohup ./run.sh > train.log 2>&1 &
# 参考: https://github.com/allenai/OLMo/blob/bf536fdfb5ab9b77c8defac2d7ca37db05eea733/scripts/augusta/peteish1-anneal.sh

# 如何断点续训: 将如下行替换下面命令中的--load_path="" 
# --load_path="/workspace/zzb_tutorial/olmo20m_pt_output/steps" 
#export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca259235j( j forfor)
export WANDB_API_KEY=PasteYourWandbApikeyHere
export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:29502 \
	scripts/train.py configs/config_526m_vqe340s147000l1_ctx1536_dna595g_mlp2_dim1280_layer32.yaml \
	--run_name="olmo-pt-bioseq-526m-dna595g-split1536_overlap256-vqe340s147000l1-ctx1536-gbsz4096-lr5e5-1ep-cnn12-mlp2-dim1280-layer32" \
        --wandb.entity="jiaoshuaihit-hit" \
        --wandb.project="bioseq-dna-04g" \
        --load_path="" \
        --save_folder="../output_526m_ctx1536-gbsz4096-lr5e5-vqe340s147000l1-mlp2-dim1280-layer32/steps/" 
