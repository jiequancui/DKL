#!/bin/bash
#SBATCH --job-name=imagenetlt_r50_r50_distill_gkl
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=imagenetlt.log
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH -p dvlab 
#SBATCH -x proj194,proj198
#SBATCH --constraint 3090

PORT=$[$RANDOM + 10000]
source activate py3.8pt1.8.1 

python imagenetlt_distill_dkl.py \
  --dataset imagenet \
  --arch resnet50 \
  --data /mnt/proj193/jqcui/data/ImageNet/imagenet-1k/data \
  --wd 5e-4 \
  --mark imagenetlt_r50_r50_distill_dkl_90e \
  --lr 0.05 \
  --aug regular_val \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 90 \
  -b 128 \
  --distill \
  --teacher_arch resnet50 \
  --model_fixed \
  --model_fixed_path data/imagenet/imagenetlt_r50_90e/moco_ckpt.best.pth.tar \
  --distill_loss 'GKL_KD' \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 0.3 \
  --T2 1.0 \
  --GI
