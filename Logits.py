from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from helpers.newloaders import *
from helpers.utils import adjust_learning_rate
from models import ResNet18
from trainer import test, train,train_L2,train_Logits,testSR

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass


# from helpers.loaders import *
parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')

parser.add_argument('--batch_size', default=100, type=int, help='the batch size')

parser.add_argument('--max_epochs', default=60, type=int, help='the maximum number of epochs')
parser.add_argument('--lradj', default=20, type=int, help='multiple the lr by 0.1 every n epochs')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--save_model', default='model.pth', help='model name')
parser.add_argument('--load_path', default='./checkpoint/ckpt.pth', help='the path to the pre-trained model, to be used with resume flag')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--log_dir', default='./chk', help='the path the log dir')
parser.add_argument('--runname', default='train', help='the exp name')
parser.add_argument('--lmbda_d', default=0, type=float,
                    help='value of regularization parameter to control total weight change')
parser.add_argument('--poison_train', action='store_true', help='train with poison image?')
parser.add_argument('--poison_p', default=0.5, type=float) 

parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')#./data/protecting_ipp/content/cifar10  or ./data/trigger_set
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--wm_batch_size', default=2, type=int, help='the batch size')
parser.add_argument('--method', help='watermarking method')
parser.add_argument('--trg_set_size', default=500, type=int, help='the batch size')                    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
LOG_DIR = os.path.join(args.log_dir, str(args.runname))
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
#args.save_dir = os.path.join(LOG_DIR,'chk')

logfile = os.path.join(LOG_DIR, 'log.txt')
confgfile = os.path.join(LOG_DIR, 'conf.txt')

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainloader, testloader, n_classes = getdataloader(args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

#get wmloader
transform = get_wm_transform(args.method, args.dataset)
trigger_set = get_trg_set(args.wm_path, args.wm_lbl, args.trg_set_size, transform)
wmloader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)



# create the model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(args.load_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    checkpoint2 = torch.load(args.load_path)
    teachernet = checkpoint2['net']
    acc2 = checkpoint2['acc']
    start_epoch2 = checkpoint2['epoch']
else:
    print('==> Building model..')
    net = ResNet18(num_classes=n_classes)

net = net.to(device)

teachernet = teachernet.to(device)

# support cuda
if device == 'cuda':
    print('Using CUDA')
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    teachernet = torch.nn.DataParallel(teachernet, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


lr_now = args.lr
lmbda_d = args.lmbda_d
poison_p = args.poison_p
B=args.batch_size

print("wm ori acc:")
test(net, criterion, logfile, wmloader, device)

# start training
for epoch in range(start_epoch, start_epoch + args.max_epochs):
   
    with open(logfile, 'a') as f:
        f.write('epoch : %d  lr: %f\n' % (epoch,lr_now))
    #train(epoch, net, criterion, optimizer, logfile,trainloader, device, wmloader)
    #train_look(epoch, net, teachernet,criterion, optimizer, logfile, trainloader, device, wmloader, tune_all=True)
    #train_L2(epoch, net, teachernet, criterion, optimizer, logfile, trainloader, device, weight_reg_param, wmloader, tune_all=True)

    train_Logits(epoch, net, teachernet, criterion, optimizer, logfile, trainloader, device,  lmbda_d , wmloader, tune_all=True)
   
    print("Test acc:")
    acc = test(net, criterion, logfile, testloader, device)

    print("wm acc:")
    test(net, criterion, logfile, wmloader, device)

    print('Saving..')
    state = {
        'net': net.module if device is 'cuda' else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if (epoch + 1) % 5 == 0:
        torch.save(state, os.path.join(args.save_dir, str(args.runname), 'epoch_' + str(epoch) +str(args.save_model)))