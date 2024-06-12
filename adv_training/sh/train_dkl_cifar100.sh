#!/bin/bash
#SBATCH --job-name=cifar100_dkl_a4b20t2
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=dkl.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj77,proj193
#SBATCH --constrain 3090



source activate py3.8_pt1.8.1

# T = 2.0 for better robustness
python train_dkl_cifar100.py \
       --arch WideResNet34_10 \
       --data CIFAR100 \
       --train_budget 'high' \
       --mark cifar100_dkl_a4b20t4 \
       --epsilon 8 \
       --lr 0.2 \
       --beta 20.0 \
       --alpha 4.0 \
       --T 4.0 

python swa.py workdir/cifar100_dkl_a4b20t4 0.9 1

python train_dkl_cifar100.py \
       --arch WideResNet34_10 \
       --data CIFAR100 \
       --train_budget 'high' \
       --mark cifar100_dkl_a5b20t4 \
       --epsilon 8 \
       --lr 0.2 \
       --beta 20.0 \
       --alpha 5.0 \
       --T 4.0 

python swa.py workdir/cifar100_dkl_a5b20t4 0.9 1

