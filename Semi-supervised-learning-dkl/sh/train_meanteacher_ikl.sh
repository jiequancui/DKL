#!/bin/bash
#SBATCH --job-name=meanteacher_ikl_a2b3
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=meanteacher_ikl_a2b3
#SBATCH --gres=gpu:1
#SBATCH -c 4 
#SBATCH -p dvlab 
#SBATCH -w proj203

source activate usb 

python train.py --c dkl_configs/meanteacher_cifar100_200_0_mask_ikl_a2b3.yaml 
