import random
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import HPQ.utils as utils
# import HPQ.regularizer as regularizer

#
def train_accuracy(output, target, images):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    # losses.update(loss.item(), images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))

    return top1.avg, top5.avg


def validate(val_loader, val_num, model, criterion=None, device=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    valid_losses = []
    valid_loss = 0
    acc = 0.0
    test_time = 0
    # switch to evaluate mode
    # 不启用 Batch Normalization 和 Dropout。
    # model.eval()
    model.eval()
    # model.module.train()
    # 不会track梯度
    with torch.no_grad():
        end = time.time()
        data_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (images, target) in data_bar:
            images, target = images.to(device), target.to(device)
            # compute output
            output = model(images.to(device))

            loss = criterion(output, target)
            valid_losses.append(loss.item())
            # predict = torch.max(output, dim=1)[1]
            # acc += torch.eq(predict, target).sum().item()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # if i % 100 == 0:
            #     progress.display(i)
            data_bar.set_description(f'evaluate epoch[{i + 1}/{len(val_loader)}]')


        # val_accurate = acc / val_num
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '
              .format(top1=top1, top5=top5))

    valid_loss = np.average(valid_losses)
    # print("valid_loss: {}".format(valid_loss))

    return top1.avg, top5.avg, valid_loss

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

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
     
  
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(10)

    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    validate_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                    transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1500, shuffle=False, num_workers=35,  pin_memory=True)

    model = mmodels.MobileNetV2_GN_INF32()
    pretrain_dict = torch.load("./saves/regularization_model/mov2-inf32-uniform-test-best.pth.tar")
    model.load_state_dict(pretrain_dict['state_dict'])

    num = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            num = num + 1
            if num == 50:
                w = np.full((m.weight.data.shape[0], m.weight.data.shape[1], m.weight.data.shape[2], m.weight.data.shape[3]),
                            fill_value=2.0, dtype=np.float32)
                # w = np.full(m.weight.data.shape[0], fill_value=1e-5, dtype=np.float32)
                tensorw = torch.nn.Parameter(torch.from_numpy(w))
                print(type(tensorw), tensorw.dtype, type(m.weight), m.weight.dtype)
                a = m.weight * tensorw
                m.weight.data = a.data
                sns.distplot(m.weight.data, hist=True, kde=False, bins='auto')
                plt.show()
                break
            # print(m, ':', m.weight)
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         sns.distplot(m.weight.data, hist=True, kde=False, bins='auto')
    #         plt.show()
    # print(num)
    net = nn.DataParallel(model.cuda(), device_ids=[0, 1])
    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    top1acc, top5acc, valloss= validate(validate_loader, val_num=0, model=net, criterion=loss_function, device=device)
    # print(top1acc, top5acc, valloss)
