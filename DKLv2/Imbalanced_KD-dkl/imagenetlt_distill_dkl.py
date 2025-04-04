import argparse
import builtins
import math
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models import resnet_imagenet
from randaugment import rand_augment_transform, GaussianBlur
import moco.loader
import moco.builder
from dataset.imagenet import ImageNetLT
from dataset.inat import INaturalist
from dataset.imagenet_moco import ImageNetLT_moco
from dataset.inat_moco import INaturalist_moco
from losses import PaCoLoss
from utils import shot_acc

from dda_builder_v4 import DDAModel


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names += ['resnext101_32x4d']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:9996', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=False, type=bool,
                    help='use cosine lr schedule')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='warmup epochs')
parser.add_argument('--aug', default=None, type=str,
                    help='aug strategy')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--num_classes', default=1000, type=int, help='num classes in dataset')

# fp16
parser.add_argument('--fp16', action='store_true', help=' fp16 training')

# distill
parser.add_argument('--model_fixed', action='store_true', help=' fp16 training')
parser.add_argument('--model_fixed_path', default=None, type=str, help='model_fixed_path')
parser.add_argument('--distill', action='store_true', help='distill')
parser.add_argument('--temperature', default=1.0, type=float, help='for distill')
parser.add_argument('--distill_w', default=1.0, type=float, help='for distill')
parser.add_argument('--distill_loss', default='kl', type=str, help='loss for distill')
parser.add_argument('--teacher_arch', default=None, type=str, help='teacher arch')

# dkl
parser.add_argument('--alpha', default=1.0, type=float, help='for distill')
parser.add_argument('--beta', default=1.0, type=float, help='for distill')
parser.add_argument('--gamma', default=1.0, type=float, help='for distill')
parser.add_argument('--GI', action='store_true', help='distill')
parser.add_argument('--T2', default=1.0, type=float, help='for gkl')

best_acc = 0

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def dkl_loss(logits_student, logits_teacher, temperature, alpha, beta, gamma, CLASS_PRIOR=None, GI=False, T2=1.0):
    _, NUM_CLASSES = logits_student.shape
    delta_n = (logits_teacher.view(-1, NUM_CLASSES, 1) - logits_teacher.view(-1, 1, NUM_CLASSES))
    delta_a = (logits_student.view(-1, NUM_CLASSES, 1) - logits_student.view(-1, 1, NUM_CLASSES))
    # GI with class prior
    if GI:
        assert CLASS_PRIOR is not None, 'CLASS_PRIOR information should be collected'
        with torch.no_grad():
            CLASS_PRIOR = torch.pow(CLASS_PRIOR, gamma)
            p_n = CLASS_PRIOR.view(-1, NUM_CLASSES, 1) @ CLASS_PRIOR.view(-1, 1, NUM_CLASSES)
    else:
        s_n = F.softmax(logits_teacher / T2, dim=1)
        s_n = torch.pow(s_n, gamma)
        p_n = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum() / p_n.sum()
    loss_sce = -(F.softmax(logits_teacher / temperature, dim=-1).detach() * F.log_softmax(logits_student / temperature, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce

def main():
    args = parser.parse_args()
    args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
    os.makedirs(args.root_model, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # num_classes = 8142 if args.dataset == 'inat' else 1000
    print("=> creating model '{}'".format(args.arch))
    model = DDAModel(models.__dict__[args.arch] if args.arch != 'resnext101_32x4d' else getattr(resnet_imagenet, args.arch), args.num_classes)

    # fixed model
    if args.model_fixed:
        model_fixed = DDAModel(models.__dict__[args.teacher_arch] if args.teacher_arch != 'resnext101_32x4d' else getattr(resnet_imagenet, args.teacher_arch), args.num_classes)
        for param in model_fixed.parameters():
            param.requires_grad = False

    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

            filename=f'{args.root_model}/moco_ckpt.pth.tar'
            if os.path.exists(filename):
               args.resume = filename

            if args.reload:
               state_dict_ssp = torch.load(args.reload)['state_dict']
               model.load_state_dict(state_dict_ssp)

            # model_fixed
            if args.model_fixed:
                model_fixed.cuda(args.gpu)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # model_fixed
    if args.model_fixed:
        assert args.model_fixed_path is not None, "given the path of model_fixed"
        checkpoint = torch.load(args.model_fixed_path)['state_dict']
        new_state_dict = {}
        for key in checkpoint.keys():
            new_state_dict[key[7:]]=checkpoint[key]
        model_fixed.load_state_dict(new_state_dict)
        model_fixed.eval()


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    txt_train = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_train.txt' if args.dataset == 'inat' \
        else f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_train.txt'

    txt_test = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_val.txt' if args.dataset == 'inat' \
        else f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_test.txt'

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if args.dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ]

    augmentation_regular = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
    augmentation_randnclsstack = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
    ]

    augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
    ]

    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    val_dataset = INaturalist(
        root=args.data,
        txt=txt_test,
        transform=val_transform
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_test,
        transform=val_transform)

    if args.aug == 'sim_sim':
       transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randcls_sim':
         transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randclsstack_sim':
         transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif args.aug == 'randcls_randclsstack':
         transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack)]
    elif args.aug == 'regular_val':
         transform_train = [transforms.Compose(augmentation_regular), val_transform]

    train_dataset = INaturalist_moco(
        root=args.data,
        txt=txt_train,
        transform=transform_train
    ) if args.dataset == 'inat' else ImageNetLT_moco(
        root=args.data,
        txt=txt_train,
        transform=transform_train)
    print(f'===> Training data length {len(train_dataset)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, train_loader, model, criterion_ce, args)
        return

    # mixed precision 
    scaler = GradScaler()
    CLASS_PRIORS = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        CLASS_PRIORS = train(train_loader, model, criterion_ce, optimizer, epoch, scaler, args, model_fixed=None if not args.model_fixed else model_fixed, CLASS_PRIORS=CLASS_PRIORS)
        acc = validate(val_loader, train_loader, model, criterion_ce, args)
        if acc >best_acc:
           best_acc = acc
           is_best = True
        else:
           is_best = False

        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'acc': acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, filename=f'{args.root_model}/moco_ckpt.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, scaler, args, model_fixed=None, CLASS_PRIORS=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    if args.distill and args.distill_loss == "GKL_KD":
            PRIOR = torch.zeros(args.num_classes, args.num_classes).cuda()
            prior = CLASS_PRIORS if CLASS_PRIORS is not None else \
                    torch.ones(args.num_classes, args.num_classes).cuda() / args.num_classes

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if not args.fp16:
           logits_q = model(images[0])
           # model_fixed
           if args.model_fixed:
               with torch.no_grad():
                   logits_k = model_fixed(images[0])
                   if args.distill_loss == "GKL_KD":
                       onehot = F.one_hot(target, num_classes=args.num_classes).float()
                       onehot_a, logits_k_a = concat_all_gather(onehot), concat_all_gather(logits_k)
                       PRIOR = PRIOR + (onehot_a.t() @ F.softmax(logits_k_a / args.T2, dim=-1))

           logits = logits_q
           loss = criterion(logits, target)
           if args.distill:
               assert args.model_fixed == True, 'distillation should have a teacher model'
               if args.distill_loss == 'GKL_KD':
                   CLASS_PRIOR = onehot @ prior
                   loss += dkl_loss(logits, logits_k, args.temperature, args.alpha, args.beta, args.gamma, CLASS_PRIOR=CLASS_PRIOR, GI=args.GI, T2=args.T2)
               elif args.distill_loss == 'KL_KD':
                   loss += args.distill_w * kd_loss(logits, logits_k, args.temperature)
        else:
            with autocast():
                logits_q = model(images[0])
                # model_fixed
                if args.model_fixed:
                    with torch.no_grad():
                        logits_k = model_fixed(images[0])
                        if args.distill_loss == "GKL_KD":
                            onehot = F.one_hot(target, num_classes=args.num_classes).float()
                            onehot_a, logits_k_a = concat_all_gather(onehot), concat_all_gather(logits_k)
                            PRIOR = PRIOR + (onehot_a.t() @ F.softmax(logits_k_a / args.T2, dim=-1))

                logits = logits_q
                loss = criterion(logits, target)
                if args.distill:
                    assert args.model_fixed == True, 'distillation should have a teacher model'
                    if args.distill_loss == 'GKL_KD':
                        CLASS_PRIOR = onehot @ prior
                        loss += dkl_loss(logits, logits_k, args.temperature, args.alpha, args.beta, args.gamma, CLASS_PRIOR=CLASS_PRIOR, GI=args.GI, T2=args.T2)
                    elif args.distill_loss == 'KL_KD':
                        loss += args.distill_w * kd_loss(logits, logits_k, args.temperature)

        output = logits
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(acc1[0], output.size(0))
        top5.update(acc5[0], output.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if not args.fp16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimier)
            scaler.update()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)

    # class priors
    if args.distill_loss == "GKL_KD":
        return PRIOR / PRIOR.sum(1, keepdim=True)
    else:
        return None


def validate(val_loader, train_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        open(args.root_model+"/"+args.mark+".log","a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
              .format(top1=top1, top5=top5))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(preds, total_labels, train_loader, acc_per_cls=True)
        open(args.root_model+"/"+args.mark+".log","a+").write('Many_acc: %.5f, Medium_acc: %.5f Low_acc: %.5f\n'%(many_acc_top1, median_acc_top1, low_acc_top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(args.root_model+"/"+args.mark+".log","a+").write('\t'.join(entries)+"\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
       lr = lr / args.warmup_epochs * (epoch + 1 )
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1 ) / (args.epochs - args.warmup_epochs + 1 )))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    main()
