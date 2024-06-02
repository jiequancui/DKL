from __future__ import print_function
import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
from autoaug import CIFAR10Policy, Cutout
from dataset import cifar


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='WideResNet34')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'CIFAR10V2', 'CIFAR100V2'])
parser.add_argument('--data-path', type=str, default='../data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./workdir',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
# AWP
parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')


## DKL
parser.add_argument('--mark', type=str)
parser.add_argument('--train_budget', type=str, default='low')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--beta', default=15.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--T', default=1.0, type=float,
                    help='label smoothing')
parser.add_argument('--aug', default='basic', type=str,
                    help='aug strategy')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='loss weight for aug data')
parser.add_argument('--lr-warmup', default=10, type=int,
                    help='warmup learning rate')
parser.add_argument('--m', default=1.0, type=float,
                    help='momentum weight')
parser.add_argument('--ls', default=0.0, type=float,
                    help='label smoothing')


args = parser.parse_args()
epsilon = args.epsilon / 255
step_size = args.step_size / 255
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty

if args.aug != 'basic':
    data_test = args.data[:-2]
else:
    data_test = args.data

if data_test == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

# settings
model_dir = args.model_dir + "/" + args.mark
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_aug_cutout = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),

])

transform_aug_autoaug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(), 
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.aug == 'basic':
   trainset = getattr(datasets, args.data)(
      root=args.data_path, train=True, download=True, transform=transform_train)
elif args.aug == 'autoaug':
   trainset = getattr(cifar, args.data)(
      root=args.data_path, train=True, download=True, transform=[transform_train, transform_aug_autoaug])
elif args.aug == 'cutout':
   trainset = getattr(cifar, args.data)(
      root=args.data_path, train=True, download=True, transform=[transform_train, transform_aug_cutout])

testset = getattr(datasets, data_test)(
    root=args.data_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def dkl_loss(logits_nat, logits_adv, weight, alpha, beta):
    num_classes = logits_nat.size(1)
    delta_n = logits_nat.view(-1, num_classes, 1) - logits_nat.view(-1, 1, num_classes)
    delta_a = logits_adv.view(-1, num_classes, 1) - logits_adv.view(-1, 1, num_classes)
    
    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * weight).sum() / logits_nat.size(0)
    loss_sce = -(F.softmax(logits_nat, dim=1).detach() * F.log_softmax(logits_adv, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce 


def cross_entropy(logits_nat, target, smooth=0.1):
    num_classes = logits_nat.size(1)
    onehot = F.one_hot(target, num_classes).float()
    onehot_sm = onehot * (1-smooth) + (1-onehot) * smooth / (num_classes-1)
    loss_sce = - (onehot_sm * F.log_softmax(logits_nat, dim=-1)).sum(1).mean()
    return loss_sce


def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  weight=None,
                  alpha=1.0,
                  beta=1.0,
                  ):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = dkl_loss(model(x_natural), model(x_adv), weight, 1.0, 1.0)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def train(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs, weight=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    WEIGHT = torch.zeros(NUM_CLASSES, NUM_CLASSES,).cuda()
    weight = weight if weight is not None else torch.ones(NUM_CLASSES, NUM_CLASSES).cuda() / NUM_CLASSES
    epoch_scale = args.epochs / 200.0

    for batch_idx, (data, target) in enumerate(train_loader):
        if isinstance(data, list):
            x_natural, x_aug, target = data[0].to(device), data[1].to(device), target.to(device)
            x_natural, target = torch.cat((x_natural, x_aug), dim=0), torch.cat((target, target), dim=0)
        else:
            x_natural, target = data.to(device), target.to(device)

        varepsilon = epsilon * (epoch / args.epochs)
        if args.train_budget=='low':
            step_size = varepsilon
            iters_attack = 2
        elif args.train_budget=='high':
            if epoch<=int(50 * epoch_scale):
                step_size = varepsilon
                iters_attack = 2
            if epoch<=int(100 * epoch_scale):
                step_size = 2*varepsilon/3
                iters_attack = 3
            if epoch<=int(150 * epoch_scale):
                step_size = varepsilon/2
                iters_attack = 4
            if epoch<=int(200 * epoch_scale):
                step_size = varepsilon/2
                iters_attack = 5


        # calculate sample weights
        with torch.no_grad():
            onehot = F.one_hot(target, NUM_CLASSES).float()
            s_n = onehot @ weight
            sample_weight = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)

        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              step_size=step_size,
                              epsilon=varepsilon,
                              perturb_steps=iters_attack,
                              distance=args.norm,
                              weight=sample_weight,
                              alpha=args.alpha,
                              beta=args.beta)

        model.train()

        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         alpha=args.alpha,
                                         beta=args.beta,
                                         weight=sample_weight)
            awp_adversary.perturb(awp)

        # optimize
        optimizer.zero_grad()

        # output
        logits_adv, logits_nat = model(x_adv), model(x_natural)

        # calculate natural loss and backprop
        bt = target.size(0) if args.aug == 'basic' else target.size(0) // 2
        with torch.no_grad():
            # update 
            WEIGHT = WEIGHT + (onehot[:bt].t() @ F.softmax(logits_nat[:bt].clone().detach() / args.T, dim=-1))

        if args.aug != 'basic':
            logits_na, logits_nb, logits_aa, logits_ab = logits_nat[:bt], logits_nat[bt:], logits_adv[:bt], logits_adv[bt:]
            loss_robust = dkl_loss(logits_na, logits_aa, sample_weight[:bt], args.alpha, args.beta) + args.gamma * \
                          dkl_loss(logits_nb, logits_ab, sample_weight[bt:], args.alpha, args.beta)
        else:
            loss_robust = dkl_loss(logits_nat, logits_adv, sample_weight, args.alpha, args.beta)
        loss_natural = F.cross_entropy(logits_nat, target)
        loss = loss_natural + loss_robust

        prec1, prec5 = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        # wa
        for start_ep, tau, new_state_dict in zip(start_wa, tau_list, exp_avgs):
            if epoch == start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = value
            elif epoch > start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]
            else:
                pass


        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()

    WEIGHT = WEIGHT / WEIGHT.sum(dim=1, keepdim=True) * args.m + weight * (1 - args.m) 
    return losses.avg, top1.avg, WEIGHT 


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            if isinstance(data, list):
                x_natural, x_aug, targets = data[0].to(device), data[1].to(device), targets.to(device)
            else:
                x_natural, targets = data.to(device), targets.to(device)

            outputs = model(x_natural)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), x_natural.size(0))
            top1.update(prec1.item(), x_natural.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epcoh < args.lr_warmup:
        lr = (args.lr / args.lr_warmup) * (epoch+1)
    else:
        lr = args.lr

    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_cosine(optimizer, epoch):
    lr = args.lr
    if epoch < args.lr_warmup:
       lr = args.lr / args.lr_warmup * epoch 
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.lr_warmup) / (args.epochs - args.lr_warmup + 1)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    # init model, ResNet18() can be also used here for training
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    # wa
    start_wa = [(150*args.epochs)//200]
    tau_list = [0.9996]
    exp_avgs = []
    model_tau = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    exp_avgs.append(model_tau.state_dict())


    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate',
                      'Adv Train Loss', 'Nat Train Loss', 'Nat Val Loss',
                      'Adv Train Acc.', 'Nat Train Acc.', 'Nat Val Acc.'])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    weight = None
    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate_cosine(optimizer, epoch)

        # adversarial training
        adv_loss, adv_acc, weight = train(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs, weight=weight)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_acc = test(model, train_loader, criterion)
        val_loss, val_acc = test(model, test_loader, criterion)
        print('================================================================')

        logger.append([lr, adv_loss, train_loss, val_loss, adv_acc, train_acc, val_acc])

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))

        if epoch >=args.epochs-1:
            for idx, start_ep, tau, new_state_dict in zip(range(len(tau_list)), start_wa, tau_list, exp_avgs):
                if start_ep <= epoch:
                    torch.save(new_state_dict,os.path.join(model_dir, 'ours-model-epoch-SWA{}{}{}.pt'.format(tau,start_ep,epoch)))


if __name__ == '__main__':
    main()
