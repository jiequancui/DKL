#!/bin/bash
#SBATCH --job-name=cifar10_dkl
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=cifar10_dkl.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj77,proj194,proj198
#SBATCH --constrain 3090

source activate py3.8_pt1.8.1
python train_dkl_cifar10.py \
       --arch WideResNet34_10 \
       --data CIFAR10 \
       --train_budget 'high' \
       --mark cifar10_dkl_a4b20t4 \
       --epsilon 8 \
       --lr 0.2 \
       --beta 20.0 \
       --alpha 4.0 \
       --T 4.0 \
       --epochs 200 \
       --seed 0

python swa.py workdir/cifar10_dkl_a4b20t4 0.9 1 200


python train_dkl_cifar10.py \
       --arch WideResNet34_10 \
       --data CIFAR10 \
       --train_budget 'high' \
       --mark cifar10_dkl_a4b20g02 \
       --epsilon 8 \
       --lr 0.2 \
       --beta 20.0 \
       --alpha 4.0 \
       --gamma 0.2
       --T 1.0 \
       --epochs 200 \
       --seed 0

python swa.py workdir/cifar10_dkl_a4b20g02 0.9 1 200
