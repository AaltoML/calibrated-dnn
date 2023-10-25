import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True, download=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False, download=True,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True, download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.data == 'caltech256':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        train_set = datasets.Caltech256(args.data_root, download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        trans,
                                        normalize
                                      ]))
        val_set = datasets.Caltech256(args.data_root, download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        trans,
                                        normalize
                                    ]))
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
        
        
    print('Number of training samples: ', len(train_set))
    print('Number of test samples: ', len(val_set))
    if args.use_valid:
        test_set_index = torch.arange(len(train_set))
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        elif args.data.startswith('caltech'):
            num_sample_test = 5000
            num_sample_valid = 2500
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            if args.data.startswith('caltech'):
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[:(-num_sample_valid-num_sample_test)]),
                    num_workers=args.workers, pin_memory=True)
            else:
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[:-num_sample_valid]),
                    num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            if args.data.startswith('caltech'):
                val_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[(-num_sample_valid-num_sample_test):-num_sample_test]),
                    num_workers=args.workers, pin_memory=True)
            else:
                val_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[-num_sample_valid:]),
                    num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            if args.data.startswith('caltech'):
                test_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, 
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[-num_sample_test:]),
                    shuffle=False,
                    num_workers=args.workers, pin_memory=True)
            else:
                test_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
