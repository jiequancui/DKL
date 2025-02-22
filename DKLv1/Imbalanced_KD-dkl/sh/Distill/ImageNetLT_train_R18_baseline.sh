#!/bin/bash
#SBATCH --job-name=imagenetlt_r18_baseline
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=imagenetlt.log
#SBATCH --gres=gpu:4
#SBATCH -c 32 
#SBATCH -p dvlab 
#SBATCH -x proj77
#SBATCH --constraint 3090

PORT=$[$RANDOM + 10000]
source activate py3.8pt1.8.1 

python imagenetlt_distill_dkl.py \
  --dataset imagenet \
  --arch resnet18 \
  --data /mnt/proj75/jqcui/Data/ImageNet \
  --wd 5e-4 \
  --mark imagenetlt_r18_90e \
  --lr 0.05 \
  --aug regular_val \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 90 \
  -b 128
