#!/bin/bash
#SBATCH --job-name=imagenetlt_r50_x101_distill
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
  --arch resnet50 \
  --data /mnt/proj193/jqcui/data/ImageNet/imagenet-1k/data \
  --wd 5e-4 \
  --mark imagenetlt_r50_x101_distill_90e \
  --lr 0.05 \
  --aug regular_val \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 90 \
  -b 128 \
  --distill \
  --distill_loss 'KL_KD' \
  --distill_w 3.0  \
  --teacher_arch resnext101_32x4d \
  --model_fixed \
  --model_fixed_path data/imagenet/imagenetlt_x101_90e/moco_ckpt.best.pth.tar 
