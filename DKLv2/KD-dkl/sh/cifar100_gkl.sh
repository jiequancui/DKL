#!/bin/bash
#SBATCH --job-name=cifar100_gkl
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cifar100.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH --constrain 3090
#SBATCH -x proj198,proj194


source activate py3.8_pt1.8.1
pip install -e .
python3 tools/train.py --cfg cifar100_kd_dkl_configs/res32x4_res8x4.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/res32x4_shuv2.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/res56_res20.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/wrn40_2_shuv1.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/wrn40_2_wrn_16_2.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/res32x4_shuv1.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/res50_mv2.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/vgg13_vgg8.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/wrn40_2_wrn_40_1.yaml
python3 tools/train.py --cfg cifar100_kd_dkl_configs/vgg13_mv2.yaml
python3 tools/train.py --cfg cifar100_config_gkl/res110_res32.yaml
