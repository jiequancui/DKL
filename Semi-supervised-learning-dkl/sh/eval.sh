#!/bin/bash
#SBATCH --job-name=semi
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=dda_baseline_imagenetlt_r50.log
#SBATCH --gres=gpu:1
#SBATCH -c 4 
#SBATCH -p dvlab 
#SBATCH -w proj198

source activate usb 

python eval.py --dataset cifar100 --num_classes 100 --load_path saved_models/usb_cv/fixmatch_cifar100_200_0_org_hard/latest_model.pth --net vit_small_patch2_32
