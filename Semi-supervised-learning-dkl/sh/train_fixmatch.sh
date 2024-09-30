#!/bin/bash
#SBATCH --job-name=fixmatch_baseline
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=fixmatch_baseline
#SBATCH --gres=gpu:1
#SBATCH -c 4 
#SBATCH -p dvlab 
#SBATCH -w proj196

source activate usb 

python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml
