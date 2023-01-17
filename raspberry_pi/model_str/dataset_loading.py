# -*- coding: utf -8-*-
# Author: YOBUWEN
# Date:  22:12
# File: dataset_loading.py
import os
import json
import sys

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split
import random
import matplotlib.pyplot as plt


data_transform = {
    "train": transforms.Compose([
                                # transforms.Pad(4),
                                transforms.RandomCrop(32, padding=4),
                                # transforms.RandomResizedCrop(224),
                                # transforms.Resize(224), #image net
                                # transforms.RandomResizedCrop(224), #image net
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #cifar-10
                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #image net
    ]),
    "val": transforms.Compose([
                                # transforms.Resize(256),
                                # transforms.CenterCrop(224),
                                # transforms.RandomCrop(32, padding=4),
                                # transforms.RandomHorizontalFlip(0.5),
                                # transforms.RandomResizedCrop(224),
                                # transforms.Resize(224), #image net
                                # transforms.RandomResizedCrop(224), #image net
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #cifar-10
                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #image net
    ])
}


def loading_data(num_workers = None, data_type = 'CIFAR10', batch_size = 10,
                    path = '.', data_plot = False, datasetsplit = True, train_split_ratio=0.6):

    nw = min([os.cpu_count(), num_workers if num_workers > 1 else 1])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

# +++++++++++++loading pytorch dataset++++++++++++
    if data_type == 'CIFAR10' :
        print('Using {} loading data.'.format(data_type))
        train_dataset = torchvision.datasets.CIFAR10(root=path,
                                                     train=True, download=True, transform=data_transform["train"])
        test_dataset = torchvision.datasets.CIFAR10(root=path,
                                                    train=False, download=True, transform=data_transform["val"])
# +++++++++++++loading SVHN dataset++++++++++++
    elif data_type == 'SVHN':
        print('Using {} loading data.'.format(data_type))
        train_dataset = torchvision.datasets.SVHN(root=path,
                                                  split="train", download=True, transform=data_transform["train"])
        test_dataset = torchvision.datasets.SVHN(root=path,
                                                 split="test", download=True, transform=data_transform["val"])
# +++++++++++++loading imagenet dataset++++++++++++
    elif data_type == 'ImageNet':
        print('Using {} loading data.'.format(data_type))
        train_dataset = torchvision.datasets.ImageNet(root=path,
                                                    split= "train", transform=data_transform["train"])
        test_dataset = torchvision.datasets.ImageNet(root=path,
                                                    split="val", transform=data_transform["val"])
# +++++++++++++loading imagenet dataset++++++++++++
    elif data_type == 'MNIST':
        print('Using {} loading data.'.format(data_type))
        train_dataset = torchvision.datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root=path,  train=False, transform=transforms.ToTensor(), download=True)
    else :
        print("\033[31mWarning: No selected loading data.\033[0m")
        sys.exit(1)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_num = len(test_dataset)

    if datasetsplit is True:
        train_size = int(round(len(train_dataset) * train_split_ratio))
        val_size = len(train_dataset) - train_size
        train_ds, val_ds = random_split(train_dataset, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(0))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        train_num = len(train_ds)
        validate_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        val_num = len(val_ds)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        train_num = len(train_dataset)


    #write data targets to .json
    if data_type == 'CIFAR10':
        imagenet_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in imagenet_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        file_class_indices = data_type + 'class_indices.json'
        with open(file_class_indices, 'w') as json_file:
            json_file.write(json_str)
    else:
        file_class_indices = None

    if data_plot == True:
        datas = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,transform=transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(datas, batch_size=4, shuffle=False, num_workers=1)
        plot_data_loader_image(data_loader, file_class_indices)

    if datasetsplit is True:
        return train_num, val_num, test_num, train_loader, validate_loader, test_loader, file_class_indices
    else:
        return train_num, None, test_num, train_loader, None, test_loader, file_class_indices


def plot_data_loader_image(data_loader, file_class_indices):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    # json_path = './class_indices.json'
    json_path = './' + file_class_indices
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = img * 255
            # img = (img * [0.4914, 0.4822, 0.4465] + [0.2023, 0.1994, 0.2010]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            # plt.xticks([])  # 去掉x轴的刻度
            # plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

