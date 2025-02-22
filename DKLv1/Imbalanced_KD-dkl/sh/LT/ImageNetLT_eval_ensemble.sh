#!/bin/bash
#SBATCH --job-name=gpaco_ensemble
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=gpaco_ensemble.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj77
#SBATCH --constraint=3090

source activate py3.8_pt1.8.1 

PORT=$[$RANDOM + 10000]
python paco_lt_ensemble.py \
  --dataset imagenet \
  --arch resnext50_32x4d \
  --data /mnt/proj75/jqcui/Data/ImageNet \
  --alpha 0.05 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --lr 0.06 \
  --moco-t 0.2 \
  --aug randcls_randclsstack \
  --randaug_m 10 \
  --randaug_n 3 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 \
  --evaluate \
  --ensemble \
  --ensemble_paths "paco_models/gpaco_x101_imagenetlt.pth.tar,paco_models/gpaco_x50_imagenetlt.pth.tar" \
  --ensemble_archs "resnext101_32x4d, resnext50_32x4d" \
  --mark "ensemble_x50_x101"
