import random
import time
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_loading import loading_data
from model import LeNet

def test(test_loader, test_num, model, criterion=None, device=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
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
        data_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (images, target) in data_bar:
            images, target = images.to(device), target.to(device)

            # compute output
            output = model(images.to(device))
            # output = model.module.quant_interface(images.to(device))
            # output = model.module.cal_quantize_inference(images.to(device))

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
            data_bar.set_description(f'test epoch[{i + 1}/{len(test_loader)}]')
            data_bar.set_postfix({'loss': '{:.3f}'.format(loss.item()),})


        # val_accurate = acc / val_num
        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '
        #       .format(top1=top1, top5=top5))

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

    # validate_dataset = torchvision.datasets.CIFAR10(root='/home/yobuwen/PycharmProjects/deepcompression/data', train=False, download=True,
    #                                                 transform=transforms.Compose([transforms.ToTensor(),
    #                                                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    # validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1024, shuffle=False, num_workers=16,  pin_memory=True)
    #
    # test_num = len(validate_dataset)
    # train_num, val_num, test_num, train_loader, validate_loader, test_loader, file_class_indices \
    #     = loading_data(data_type='MNIST', datasetsplit=False, train_split_ratio=0.8, data_plot=True,
    #                    num_workers=32, batch_size=128)
    """=================================================================================================================
        step1:instantiated model.
        step2:calculate weight of quantization to dequantization, and loading weight parameters.
        step3:calibrating data for activation using calibration dateset, 
              and output scale and zero point of activation by model forward.
        step4:calculate quantize of bias and activate value when inference.
    ================================================================================================================="""
    # model = LeNet()
    # pretrain_dict = torch.load('./lenet-best.pth.tar')
    # model.load_state_dict(pretrain_dict['state_dict'])

    model = torch.load('./lenet.pth')
    model.to(device)
    print(model)

    # loss_function = nn.CrossEntropyLoss()
    # top1_acc, top5_acc, test_loss = test(test_loader, test_num, model, loss_function, device)
    # print(' * test Acc@1 {:.3f} Acc@5 {:.3f} '.format(top1_acc, top5_acc))
    """=============================================================================================================="""
    data_transform = transforms.Compose([
                                transforms.Grayscale(1),#mnist data
                                transforms.Resize((28,28)), #cifar10 32*32/ mnist 28*28
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307),(0.3081))
    ])
    img = Image.open(r'./image/2.jpg')
    img = data_transform(img)
    data = torch.unsqueeze(img, dim=0)
    # plt.imshow(img[0])
    # plt.show()
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(data.to(device))).cpu()
        pre = torch.argmax(torch.softmax(output, dim=0)).numpy()

    print(pre)

    # """****************************************************************************************************************
    # *  calibration test function.                        START
    # **************************************************************************************************************"""
    # print('cal-'*10)
    # model = mmodels.LeNet()
    # path = '/home/yobuwen/PycharmProjects/SQS/HPQ/saves/baseline/sqs_lenet-best.pth.tar'
    # # utils.test(model, w_bit=[1, 2, 4], path=path, dequant=True)
    # # pretrain_dict = torch.load("/home/yobuwen/PycharmProjects/SQS/HPQ/saves/dequantize.pth")
    # model.load_state_dict(pretrain_dict)
    # # net = nn.DataParallel(model.cuda(), device_ids=[0, 1])
    # model.to(device)
    # loss_function = nn.CrossEntropyLoss()
    # # top1_acc, top5_acc, test_loss = test(test_loader, test_num, net, loss_function, device)
    # # print(' * test Acc@1 {:.3f} Acc@5 {:.3f} '.format(top1_acc, top5_acc))
    # """=============================================================================================================="""
    # model.cal_quantize()
    # data_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    # for i, (images, target) in data_bar:
    #     images, target = images.to(device), target.to(device)
    #     output = model.cal_quantize_forward(images.to(device))
    #     # if i%5 == 0:
    #     #     break
    # """=============================================================================================================="""
    # model.cal_freeze()
    # # for fm in model.modules():
    # #     if isinstance(fm, QConv2d) or isinstance(fm, QLinear):
    # #     # if hasattr(fm, 'qi') or hasattr(fm, 'qo') or hasattr(fm, 'qw'):
    # #         print(fm.qi,"/",fm.qo,"/",fm.qw)
    #
    # net = nn.DataParallel(model.cuda(), device_ids=[0, 1])
    # net.to(device)
    # top1_acc, top5_acc, test_loss = test(test_loader, test_num, net, loss_function, device)
    # print(' * test Acc@1 {:.3f} Acc@5 {:.3f} '.format(top1_acc, top5_acc))
    #
    # data_transform = transforms.Compose([
    #                             transforms.Grayscale(1),#mnist data
    #                             transforms.Resize((28,28)), #cifar10 32*32/ mnist 28*28
    #                             transforms.ToTensor()])
    # img = Image.open('image/2.jpg')
    # img = data_transform(img)
    # data = torch.unsqueeze(img, dim=0)
    # # plt.imshow(img[0])
    # # plt.show()
    # output = torch.squeeze(net.module.cal_quantize_inference(data.to(device))).cpu()
    # pre = torch.argmax(torch.softmax(output, dim=0))
    # print(pre)
    # """****************************************************************************************************************
    # *  calibration test function.                          END
    # **************************************************************************************************************"""
