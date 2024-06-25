
source activate py3.8pt1.8.1
python train_trades_cifar.py \
       --arch WideResNet34_10 \
       --data CIFAR100 \
       --train_budget 'high' \
       --mark cifar100_trades \
       --epsilon 8 \
       --lr 0.2 \
       --beta 9.0


python swa.py workdir/cifar100_trades 0.9 1 200
