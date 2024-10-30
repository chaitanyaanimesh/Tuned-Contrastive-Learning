import argparse
import math
import numpy as np
import os
import torch
import copy
import gc
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

from torch.backends import cudnn
from torch import nn
from torch import optim
from torchvision import datasets, transforms

import cub2011
import imagenet_mini
from loss.tcl import TCL
from loss.supcon import SUPCON


from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform
from torch.utils.data import RandomSampler 

from models.resnet_big_pretrained import get_resnet_contrastive_big_pretrained # For ImageNet
from models.resnet_big_pretrained_cifar import get_resnet_contrastive_big_pretrained_cifar #For CIFAR and FMNIST

val_accuracy1 = []
curEpoch=-1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101"
        ],
        help="Select encoder architecture",
    )
    parser.add_argument(
        "--dataset",
        default="cifar100",
        choices=["cifar10", "cifar100","FMNIST","ImageNet-100"],
        help="select dataset",
    )
    parser.add_argument(
        "--training_mode",
        default="contrastive",
        choices=["contrastive", "cross-entropy"],
        help="Type of training — contrastive or cross-entropy",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="For contrastive training thisis multiplied by two.",
    )

    parser.add_argument("--temperature", default=0.1, type=float, help="Constant for loss no thorough ")
    parser.add_argument("--augment", default='autoaug',choices=['autoaug','simaug','randaug'], type=str)

    parser.add_argument("--n_epochs_contrastive", default=100, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=50, type=int)

    parser.add_argument("--lr_contrastive", default=1e-1, type=float)
    parser.add_argument("--lr_cross_entropy", default=5e-1, type=float)
    
    parser.add_argument("--k1", default=5000.0, type=float, help="Set value for k1 for TCL")
    parser.add_argument("--k2", default=1.0, type=float, help="Set value for k2 for TCL")
    
    parser.add_argument("--run_type", default='tcl', type=str)
    parser.add_argument("--checkpoint_path", default='./checkpoint/cifar100/', type=str, help="path for checkpointing")
    
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers for Dataloader")

    parser.add_argument("--cosine", default=True, type=bool, help="Check this tos use cosine annealing instead of ")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")
    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=[150, 300, 500],
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    
    
    
    parser.add_argument("--dataset_path", default='./data/cifar100', type=str, help="path where dataset would be downloaded")
    parser.add_argument("--logs_path", default='./logs/cifar100', type=str, help="path where logs will be made")
    parser.add_argument("--validation_frequency", default=100, type=int)
    parser.add_argument("--gpu", default=1, type=int, help="Number of GPUs to use")
    

    args = parser.parse_args(args=[])

    return args


def adjust_learning_rate(optimizer, epoch, mode, args, method):
    """

    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    
    if method=='normal':
        if mode == "contrastive":
            lr = args.lr_contrastive
            n_epochs = args.n_epochs_contrastive
        elif mode == "cross_entropy":
            lr = args.lr_cross_entropy
            n_epochs = args.n_epochs_cross_entropy
        else:
            raise ValueError("Mode %s unknown" % mode)

        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
        else:
            n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if n_steps_passed > 0:
                lr = lr * (args.lr_decay_rate ** n_steps_passed)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        if mode == "contrastive":
            lr = args.lr_contrastive_mod
            n_epochs = args.n_epochs_contrastive
        elif mode == "cross_entropy":
            lr = args.lr_cross_entropy
            n_epochs = args.n_epochs_cross_entropy
        else:
            raise ValueError("Mode %s unknown" % mode)

        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
        else:
            n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if n_steps_passed > 0:
                lr = lr * (args.lr_decay_rate ** n_steps_passed)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            

def train_contrastive(model, train_loader, train_loader_cross_entropy, test_loader, criterion1,
                      optimizer1, args):
    
    model.train()
    best_loss = float("inf")
    for epoch in range(args.n_epochs_contrastive):
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_contrastive))
        global curEpoch
        curEpoch=epoch
        
        train_loss1 = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs = torch.cat(inputs)
            targets = targets.repeat(2)
            inputs, targets = inputs.to(args.device,non_blocking=True), targets.to(args.device,non_blocking=True)
            optimizer1.zero_grad()

            projections1 = model.forward_constrative(inputs)
            loss1 = criterion1(projections1, targets)
            loss1.backward()
            optimizer1.step()

            train_loss1 += loss1.item()
    

        avg_loss1 = train_loss1 / (batch_idx + 1)
        print('Loss1=', avg_loss1, ' after epoch=', epoch + 1)

        adjust_learning_rate(optimizer1, epoch, mode="contrastive", args=args, method='normal')

        if ((epoch+1) % args.validation_frequency == 0):
            val_accuracy1.append(
                train_cross_entropy(model, train_loader_cross_entropy, test_loader, criterion1, optimizer1,
                                    args,'supconNew'))


def train_cross_entropy(model_s, train_loader, test_loader, criterion_s, optimizer_s, args, method):

    val_accuracy = 0.0
    model = copy.deepcopy(model_s)
    model.freeze_projection()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_cross_entropy,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    args.best_acc = 0.0
    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device,non_blocking=True), targets.to(args.device,non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total_batch = targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            total += total_batch
            correct += correct_batch

        val_accuracy = validation(epoch, model, test_loader, criterion, args, method)

        adjust_learning_rate(optimizer, epoch, mode='cross_entropy', args=args, method='normal')

    del model
    del criterion
    del optimizer
    gc.collect()

    return val_accuracy


def validation(epoch, model, test_loader, criterion, args, method):
    """

    :param epoch: int
    :param model: torch.nn.Module, Model
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module, Loss
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """
    global curEpoch
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.0 * correct / total

    print('Final Validation Accuracy:', acc, ' after ', epoch + 1, ' epochs')

    if (curEpoch+1) in [100,200,400,1000]:
        print("Saving the model..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "save_epoch": epoch,
            "lr_contrastive": args.lr_contrastive,
            "n_epochs_contrastive": args.n_epochs_contrastive,
            "augment" : args.augment,
            
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
            
        path=args.checkpoint_path+ \
                            args.run_type+"_"+args.model+"_"+args.dataset+"_"+str(args.lr_contrastive)+"_"+str(args.batch_size)+".pth"
        torch.save(state, path)
        args.best_acc = acc
    
    return acc


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    transform_train=None
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        if args.augment=='autoaug':
            transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ]
            transform_train.append(AutoAugment())
            
        elif args.augment=='simaug':
             transform_train = [
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ]
            
        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 10
        in_channel = 3

    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        if args.augment=='autoaug':
            transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ]
            transform_train.append(AutoAugment())
            
        elif args.augment=='simaug':
             transform_train = [
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ]
                
        elif args.augment=='randaug':
             transform_train = [
                    transforms.Resize(32),
                    RandAugment(2,14),

                ]

        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 100
        in_channel = 3

    elif args.dataset == "cub2011":
        mean = (0.485,0.456,0.406)
        std = (0.229,0.224,0.225)
        
        if args.augment=='autoaug':
            transform_train = [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            ]
            transform_train.append(AutoAugment())
            
        elif args.augment=='simaug':
             transform_train = [
                    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ]

        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = cub2011.Cub2011(root=args.dataset_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = cub2011.Cub2011(root=args.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 200
        in_channel = 3
    
    elif args.dataset == "FMNIST":
        mean = (0.5,)
        std = (0.5,)
        
        if args.augment=='autoaug':
            transform_train = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            ]
            
        elif args.augment=='simaug':
             transform_train = [
                    transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ]    
            
           
        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = datasets.FashionMNIST(root=args.dataset_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = datasets.FashionMNIST(root=args.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 10
        in_channel = 1
        
    elif args.dataset == "ImageNet-100":
        mean = (0.5,0.5,0.5)
        std = (0.25,0.25,0.25)
        
        if args.augment=='autoaug':
            transform_train = [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            ]
            transform_train.append(AutoAugment())
            
        elif args.augment=='simaug':
             transform_train = [
                    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ]

        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = imagenet_mini.ImageNetMini(root=args.dataset_path, train=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_set),
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        test_set = imagenet_mini.ImageNetMini(root=args.dataset_path, train=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 100
        in_channel = 3

    model1 = get_resnet_contrastive_big_pretrained_cifar(args.model, num_classes,pretrained=False)
#     model1 = get_resnet_contrastive_big_pretrained(args.model, num_classes,pretrained=False)
    
    if args.gpu>1 and torch.cuda.device_count() > 1:
        print("We have available ", torch.cuda.device_count(), "GPUs!")
        model1 = nn.DataParallel(model1, device_ids=[i for i in range(args.gpu)])
    model1 = model1.to(args.device)

    cudnn.benchmark = True


    if args.training_mode == "contrastive":
        train_contrastive_transform = DuplicateSampleTransform(transform_train)
        if args.dataset == "cifar10":
            train_set_contrastive = datasets.CIFAR10(
                root=args.dataset_path,
                train=True,
                download=True,
                transform=train_contrastive_transform,
            )
        elif args.dataset == "cifar100":
            train_set_contrastive = datasets.CIFAR100(
                root=args.dataset_path,
                train=True,
                download=True,
                transform=train_contrastive_transform,
            )
        elif args.dataset == "cub2011":
            train_set_contrastive = cub2011.Cub2011(
                root=args.dataset_path,
                train=True,
                download=True,
                transform=train_contrastive_transform,
            )
        elif args.dataset == "FMNIST":
            train_set_contrastive = datasets.FashionMNIST(
                root=args.dataset_path,
                train=True,
                download=True,
                transform=train_contrastive_transform,
            )
        elif args.dataset == "ImageNet-100":
            train_set_contrastive = imagenet_mini.ImageNetMini(
                root=args.dataset_path,
                train=True,
                transform=train_contrastive_transform,
            )

        train_loader_contrastive = torch.utils.data.DataLoader(
            train_set_contrastive,
            batch_size=args.batch_size,
            shuffle=True,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        model1 = model1.to(args.device)
        optimizer1 = optim.SGD(
            model1.parameters(),
            lr=args.lr_contrastive,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        criterion1=None
        if args.run_type=='tcl':
            criterion1 = TCL(temperature=args.temperature,k1=args.k1,k2=args.k2)
        else:
            criterion1 = SUPCON(temperature=args.temperature)

        criterion1.to(args.device)

        train_contrastive(model1, train_loader_contrastive, train_loader, test_loader, criterion1,
                          optimizer1, args)
        print('******Contrastive Training Completed******')


        model1.freeze_projection()

        optimizer1 = optim.SGD(
            model1.parameters(),
            lr=args.lr_cross_entropy,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        criterion1 = nn.CrossEntropyLoss()
        criterion1.to(args.device)

        args.best_acc = 0.0
  
        global curEpoch
        curEpoch=-100
        train_cross_entropy(model1, train_loader, test_loader, criterion1, optimizer1, args, 'supconNew')

        print("Finished Training")


    else:
        optimizer = optim.SGD(
            model1.parameters(),
            lr=args.lr_cross_entropy,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(args.device)

        args.best_acc = 0.0
        train_cross_entropy(model1, train_loader, test_loader, criterion, optimizer, args, 'xent')
           


if __name__ == "__main__":
    main()
    
    
