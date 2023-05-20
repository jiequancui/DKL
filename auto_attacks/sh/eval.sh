#!/bin/bash
#SBATCH --job-name=adv_eval
#SBATCH --mail-user=jiequancui@link.cuhk.edu.hk
#SBATCH --output=adv.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH -p dvlab
#SBATCH -x proj77
#SBATCH --constraint=3090


source activate py3.8_pt1.8.1
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/cifar100_a3_b20_t4.pt  --data CIFAR100 --preprocess '01' --log_path "logs/cifar100_dkl_a3_b20_T4.log"
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/cifar100_a4_b20_t2.pt  --data CIFAR100 --preprocess '01' --log_path "logs/cifar100_dkl_a4_b20_T2.log"
python eval.py --arch WideResNet34 --checkpoint ../pretrained_models/cifar10_a3_b20_t4.pt   --data CIFAR10  --preprocess '01' --log_path "logs/cifar10_dkl_a3_b20_T4.log"
python eval.py --arch wrn-28-10-swish --checkpoint ../pretrained_models/cifar100_a5_b12_t4_50m.pt --data CIFAR100 --preprocess '01' --log_path "logs/cifar100_dkl_a5_b12_T4_50m.log"
python eval.py --arch wrn-28-10-swish --checkpoint ../pretrained_models/cifar10_a3_b10_t4_20m.pt --data CIFAR10 --preprocess '01' --log_path "logs/cifar10_dkl_a3_b10_T4_20m.log"
