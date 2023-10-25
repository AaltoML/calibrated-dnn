import os
import glob
import time
import argparse

model_names = ['msdnet']

arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', default=None, type=str, help='Name of latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default=None, choices=['anytime', 'dynamic'], help='which mode to evaluate')
exp_group.add_argument('--evaluate-from', default=None, type=str, metavar='PATH', help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default=None, type=str, help='GPU available.')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet', 'caltech256'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: msdnet)')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')

# msdnet config
arch_group.add_argument('--nBlocks', type=int, default=1)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int,default=4)
arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=1)
arch_group.add_argument('--growthRate', type=int, default=6)
arch_group.add_argument('--grFactor', default='1-2-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)


# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')

optim_group.add_argument('--epochs', default=300, type=int, metavar='N',
                         help='number of total epochs to run (default: 164)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')
                         
# new args
arg_parser.add_argument('--temperature', type=float, default=1., help="temperature scaling of softmax")
arg_parser.add_argument('--laplace_temperature', type=float, default=1.0, 
        help="temperature scaling of softmax for laplace predictions")

arg_parser.add_argument('--MIE', action='store_true', default=False, help='Use model-internal ensembling')

arg_parser.add_argument('--optimize_temperature', action='store_true', default=False, help='Use the validation set to optimize temperature scaling individually for each block')
arg_parser.add_argument('--optimize_var0', action='store_true', default=False, help='Use the validation set to optimize Laplace prior variance individually for each block')

# Laplace arguments
arg_parser.add_argument('--compute_only_laplace', action='store_true', default=False, help='skip training and only fit laplace approximation')
arg_parser.add_argument('--var0', type=float, default=5e-4)
arg_parser.add_argument('--laplace', action='store_true', default=False, help='test with MC integration and laplace approximation')
arg_parser.add_argument('--n_mc_samples', type=int, default=1, help='number of samples to draw from laplace')


