import os

import numpy as np
import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass


def _getdatatransformsdb(datatype):
    transform_train, transform_test = None, None
    if datatype.lower() == CIFAR10 or datatype.lower() == CIFAR100 :#CALTECH GTSRB TINY
        # Data preperation
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif datatype.lower() == CALTECH:
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),#224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif datatype.lower() == TINY:
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),#224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif datatype.lower() ==GTSRB:
        transform_train = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    
    #elif datatype.lower() == GTSRB:


    return transform_train, transform_test


def _getdatatransformswm():
    transform_wm = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_wm


def getdataloader(datatype, train_db_path, test_db_path, batch_size):
    # get transformations
    transform_train, transform_test = _getdatatransformsdb(datatype=datatype)
    n_classes = 0

    # Data loaders
    if datatype.lower() == CIFAR10:
        print("Using CIFAR10 dataset.")
        trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
    elif datatype.lower() == CIFAR100:
        print("Using CIFAR100 dataset.")
        trainset = torchvision.datasets.CIFAR100(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 100
    elif datatype.lower() == CALTECH:
        print("Using Caltech dataset.")
        trainset = ImageFolderCustomClass(train_db_path, transform_train)
        testset = ImageFolderCustomClass(test_db_path, transform_test)
        n_classes = 101
    elif datatype.lower() == GTSRB:
        print("Using GTSRB dataset.")
        trainset = ImageFolderCustomClass(train_db_path, transform_train)
        testset = ImageFolderCustomClass(test_db_path, transform_test)
        n_classes = 43
    elif datatype.lower() == TINY:
        print("Using TINY-imagenet dataset.")
        trainset = ImageFolderCustomClass(train_db_path, transform_train)
        testset = ImageFolderCustomClass(test_db_path, transform_test)
        n_classes = 200

    else:
        print("Dataset is not supported.")
        return None, None, None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader, n_classes


def getotherdataloader(datatype, train_db_path, test_db_path, batch_size):
    # get transformations
    transform_train = _getdatatransformswm()
    transform_test = _getdatatransformswm()
    n_classes = 0

    if datatype.lower() == CALTECH:
        print("Using Caltech dataset.")
        trainset = ImageFolderCustomClass(train_db_path, transform_train)
        testset = ImageFolderCustomClass(test_db_path, transform_test)
        n_classes = 102

    else:
        print("Dataset is not supported.")
        return None, None, None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader, n_classes


def getwmloader(wm_path, batch_size, labels_path):
    transform_wm = _getdatatransformswm()
    # load watermark images
    wmloader = None

    wmset = ImageFolderCustomClass(
        wm_path,
        transform_wm)
    img_nlbl = []
    wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
    for idx, (path, target) in enumerate(wmset.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
    wmset.imgs = img_nlbl

    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    return wmloader

def getwmloader2(wm_path, batch_size, labels_path):
    transform_wm = _getdatatransformswm()
    # load watermark images
    wmloader = None

    wmset = ImageFolderCustomClass(
        wm_path,
        transform_wm)
    img_nlbl = []
    wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
    for idx, (path, target) in enumerate(wmset.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
    wmset.imgs = img_nlbl

    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return wmloader


def get_wm_transform(method, dataset):
    if method == 'WeaknessIntoStrength':    #对于modelwatermarking来说
        if dataset == "cifar10" :#if datatype.lower() == CIFAR10 or datatype.lower() == CIFAR100 :
            transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist":
            transform = transforms.Compose([
                transforms.CenterCrop(28),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        elif dataset == "caltech" or dataset == "cifar100":
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
        elif dataset == "tiny":
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
        elif dataset == "gtsrb":
            transform = transforms.Compose([
                transforms.Resize(36),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])

    else:
        if dataset == "cifar10" or dataset == "cifar100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist":
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

    return transform


def get_trg_set(path, labels, size, transform=None):
    wm_set = ImageFolderCustomClass(
        path,
        transform)
    img_nlbl = list()
    wm_targets = np.loadtxt(os.path.join(path, labels))
    for idx, (path, target) in enumerate(wm_set.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))

        if idx == (size - 1):
            break

    wm_set.imgs = img_nlbl
    print(wm_targets)#to delete
    return wm_set