#!/bin/bash


lam=$1
model=$2
kshots=$3
python task_style_vector.py \
	    --dataset paradetox \
	    --prompt_version default \
	    --exemplar_method random \
	    --num_k_shots $kshots \
	    --model_type $model \
	    --model_size 7b \
	    --batch_size 1 \
	    --gpus 0 \
	    --in_8bit True \
	    --lam $lam \
	    --seed 0
