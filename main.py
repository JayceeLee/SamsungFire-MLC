from __future__ import print_function
import argparse
import time
import math
import matplotlib.pyplot as plt
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from engine import MultiLabelMAPEngine
from models import *#resnet101_wildcat

from dic import *
from utils import *

import torch.utils.data
import dataloader 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from loss import HardNegLoss
root = '../data/'
net_name ='samsung'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch SAMSUNGFIRE')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--eval', action='store_true', help='do eval?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--model_dir', default ='models',help = '')
parser.add_argument('--data_dir', default = root+'/trainval',help ='')
parser.add_argument('--eval_dir', default = root+'/eval',help ='')
parser.add_argument('--excel_file', default = root+ './Lexicon.xlsx',help ='')
parser.add_argument('--eval_excel_file', default = root+'./Lexicon.xlsx',help ='')
parser.add_argument('--ngpu', type = int, default=1, help ='gpus')

parser.add_argument('--image-size', '-i', default=256, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='../expes/models/'+net_name+'/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--k', default=1, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default= 0.7, type=float,
                    metavar='N', help='weight for the min regions (default: 0.7)')
parser.add_argument('--maps', default=1, type=int,
metavar='N', help='number of maps per class (default: 4)')
opt = parser.parse_args()
args = opt
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
	torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
	
print('===> Loading datasets')
train_set = dataloader.samsungMLC(root=opt.data_dir,
			 	
				  	excel_file = opt.excel_file
				 )
assert train_set


eval_set = dataloader.samsungMLC(root=opt.eval_dir,

					excel_file = opt.eval_excel_file
				 )
assert eval_set
model = resnet50_wildcat(ClassNum, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
print('classifier', model.classifier)
print('spatial pooling', model.spatial_pooling)

# define loss function (criterion)
criterion = HardNegLoss()

# define optimizer
optimizer = torch.optim.Adam(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay)

state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
     'evaluate': args.evaluate, 'resume': args.resume}
state['difficult_examples'] = False
state['save_model_path'] = '../expes/models/'+net_name

engine = MultiLabelMAPEngine(state)
engine.learning(model, criterion, train_set, eval_set, optimizer)		

