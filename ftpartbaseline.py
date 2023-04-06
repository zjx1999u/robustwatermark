"""ft watermark baseline model with use percent of trainset"""
from __future__ import print_function

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from helpers.newloaders import *
from helpers.utils import re_initializer_layer
from trainer import test, train, test_logit,ft_part,testSR
from models import ResNet18

parser = argparse.ArgumentParser(description='Fine-tune CIFAR10 models.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')#./data/protecting_ipp/content/cifar10  or ./data/trigger_set
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=50, type=int, help='the batch size')
parser.add_argument('--max_epochs', default=20, type=int, help='the maximum number of epochs')
parser.add_argument('--load_path', default='./checkpoint/model.t7', help='the path to the pre-trained model')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--save_model', default='finetune.t7', help='model name')
parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='finetune', help='the exp name')
parser.add_argument('--tunealllayers', action='store_true', help='fine-tune all layers')
parser.add_argument('--reinitll', action='store_true', help='re initialize the last layer')
parser.add_argument('--usepercent', default=0.1, type=float)
parser.add_argument('--trg_set_size', default=500, type=int, help='the batch size')
parser.add_argument('--method', help='watermarking method')
parser.add_argument('--classes_num', default=10, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
LOG_DIR = os.path.join(args.log_dir, str(args.runname))
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

logfile = os.path.join(LOG_DIR, 'log.txt')
confgfile = os.path.join(LOG_DIR, 'conf.txt')


# save configuration parameters
with open(confgfile, 'w') as f:
    f.write('ft with part dataset\n')
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainloader, testloader, n_classes = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

ori_classes = args.classes_num
#wmloader
transform = get_wm_transform(args.method, args.dataset)
trigger_set = get_trg_set(args.wm_path, 'labels.txt', args.trg_set_size, transform)
wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)


# Loading model.
print('==> loading model...')
net = ResNet18(num_classes=args.classes_num)
checkpoint = torch.load(args.load_path)

net.load_state_dict(checkpoint)

start_epoch = 0

net = net.to(device)
# support cuda
if device == 'cuda':
    print('Using CUDA')
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

usepercent=args.usepercent

print("Original WM acc:")
test(net, criterion, logfile, wm_loader, device)

# re initialize and re train the last layer
private_key = net.module.linear
if args.reinitll:
    net, _ = re_initializer_layer(net, n_classes)

if device is 'cuda':
    net.module.unfreeze_model()
else:
    net.unfreeze_model()


# start training
if not n_classes < ori_classes :
    print("reinitialze WM acc:")
    test(net, criterion, logfile, wm_loader, device)

B=args.batch_size



a=[i for i in range(500)]
number = int(500*usepercent)
sample_index =random.sample(a,number)
print(sample_index)
print(len(sample_index))
    
batch_train = (np.random.random(500) <= -1)

for i in range(number):
    vali_index = sample_index[i]
    batch_train[vali_index] = True

print(batch_train)
# start training
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    #train(epoch, net, criterion, optimizer, logfile,trainloader, device, wmloader=False, tune_all=args.tunealllayers)
    ft_part(epoch, net, criterion, optimizer, logfile, trainloader, device, batch_train,tune_all=args.tunealllayers)

    print("Test acc:")
    acc = test(net, criterion, logfile, testloader, device)

    new_layer = net.module.linear
    net, _ = re_initializer_layer(net, 0, private_key)
    print("poison acc with ori layer:")
    test(net, criterion, logfile, wm_loader, device)

    net, _ = re_initializer_layer(net, 0, new_layer)

    if not n_classes < ori_classes:
        print("poison acc with new layer:")
        test(net, criterion, logfile, wm_loader, device)


    print('Saving..')
    state = {
        'net': net.module if device is 'cuda' else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if (epoch + 1) % 5 == 0:
        torch.save(state, os.path.join(args.save_dir, str(args.runname), 'epoch_' + str(epoch + 1) +str(args.save_model)))



