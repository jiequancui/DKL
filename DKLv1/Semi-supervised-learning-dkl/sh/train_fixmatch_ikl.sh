#!/bin/bash
#SBATCH --job-name=fixmatch_ikl_a1b01
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=fixmatch_ikl_a1b01
#SBATCH --gres=gpu:1
#SBATCH -c 4 
#SBATCH -p dvlab 
#SBATCH -w proj196

source activate usb 

python train.py --c dkl_configs/fixmatch_cifar100_200_0_ikl_a1b01.yaml
