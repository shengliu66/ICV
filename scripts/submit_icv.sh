#!/bin/bash

#SBATCH --partition=a100_short
##SBATCH --mail-type=END,FAIL      
##SBATCH --mail-user=sl5924@nyu.edu     
#SBATCH --ntasks=1            
#SBATCH --mem-per-cpu=64G                   
#SBATCH --time=24:00:00               
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --exclude=a100-4010

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