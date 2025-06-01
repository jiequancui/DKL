#!/bin/bash
#SBATCH --job-name=gkl_kd_r50_mv
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=imgnet.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH -p dvlab
#SBATCH -w proj195

source activate py3.8_pt1.8.1
python3 tools/train.py --cfg imagenet_config_gkl/r50_mv_gkl_kd.yaml
