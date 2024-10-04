#!/bin/bash
#SBATCH --job-name=ikl_kd_r50_mv_
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imgnet.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH -p 2080
#SBATCH -q 2080
#SBATCH -x proj73

source activate py3.8_pt1.8.1
python3 tools/train.py --cfg imagenet_kd_dkl_configs/r50_mv_ikl_kd.yaml 
