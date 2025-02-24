#!/bin/bash
#SBATCH --job-name=adv_eval
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=adv.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p dvlab
#SBATCH -x proj77
#SBATCH --constraint=3090


source activate py3.8_pt1.8.1

#First download our pretrained checkpoints
#Second evaluate the model as what follows
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/cifar100_a4b20t4.pt  --data CIFAR100 --preprocess '01' --log_path "logs/cifar100_dkl_a4b20t4.log"
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/cifar10_a4b20t4.pt   --data CIFAR10  --preprocess '01' --log_path "logs/cifar10_dkl_a4b20t4.log"
