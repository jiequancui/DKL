#!/bin/bash
#SBATCH --job-name=meanteacher_baseline
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=meanteacher_baseline
#SBATCH --gres=gpu:1
#SBATCH -c 4 
#SBATCH -p dvlab 
#SBATCH -w proj200

source activate usb 

python train.py --c config/usb_cv/meanteacher/meanteacher_cifar100_200_0_mask.yaml 
