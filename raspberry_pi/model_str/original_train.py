import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader as Dataloader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import random
from dataset_loading import loading_data
from evaluate import validate
from model import LeNet

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def or_train(network, model_name, train_data, test_data, val_num,
          loss_function, optimizer, device, epochs, best_acc=0, scheduler=None, path="."):
    mean_loss = torch.zeros(1).to(device)
    for epoch in range(epochs):
        network.train()
        running_loss = 0.0
        data_bar = tqdm(enumerate(train_data), total=len(train_data))
        for step, (input, target) in data_bar:
            image, lable = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(image.to(device))
            loss = loss_function(output, lable)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            data_bar.set_description(f'train epoch[{epoch + 1}/{epochs}]')
            data_bar.set_postfix({'loss': '{:.3f}'.format(loss), 'mean_loss': '{:.3f}'.format(mean_loss.item()),
                                  'lr': '{:.5f}'.format(optimizer.param_groups[0]["lr"])})
        if scheduler is not None:
            scheduler.step()

        # print("step {}".format(count), "loss: {:.6f}".format(loss))
        top1_acc, top5_acc, valid_loss = validate(test_data, val_num, network, loss_function, device)
        # top1_acc, top5_acc = test_step(test_data, network, device)
        crt_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        if top1_acc > best_acc:
            best_acc = top1_acc
            best_epoch = epoch + 1
            # torch.save({'epoch': epoch + 1,
            #             'state_dict': network.module.state_dict(),
            #             'best_loss': mean_loss,
            #             'optimizer': optimizer.state_dict(),
            #             'top1_acc': top1_acc,
            #             'train_time': crt_time,
            #             'mean_loss': mean_loss},
            #            path + str(model_name) + "-best" + '.pth.tar')
            torch.save(network, path + str(model_name) + '.pth')
            print("save model epoch: {}".format(epoch + 1))



if __name__ == '__main__':
    setup_seed(1)

    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    train_num, val_num, test_num, train_loader, validate_loader, test_loader, file_class_indices \
        = loading_data(data_type='MNIST', num_workers=32, batch_size=128, datasetsplit=False,
                       path=r'//model/dataset')
    net = LeNet()

    # net = nn.DataParallel(net.cuda(), device_ids=[0, 1])
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    or_train(network=net, model_name='lenet',
             train_data=train_loader,
             test_data=test_loader,
             # best_acc=best_prec1,
             val_num=val_num,
             loss_function=loss_function,
             optimizer=optimizer,
             scheduler=None,
             device=device,
             epochs=50,
             path='/')