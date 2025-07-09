import os

import torch
import torch.nn as nn
import torchvision

from architectures_cifar import wrn_28_2, wrn_28_8
from torchvision import transforms, datasets
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def predict(checkpoint):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./dataset/cifar10', train=False, download=True, transform=data_transform
    )

    # create model
    model = wrn_28_2(num_classes=10, ema=True, mode=checkpoint['mode'])
    model = nn.DataParallel(model).cuda()
    # load model weights
    model.load_state_dict(checkpoint['ema_state_dict'])
    # load image
    trueNum = 0
    pred_res = []
    ground_truth = []
    test_num = len(test_dataset)
    for step, data in enumerate(test_dataset, start=0):
        img, labels = data
        img = torch.unsqueeze(img, dim=0).cuda()
        model.eval()
        with torch.no_grad():
            # predict class
            output1 = model(img)['logits']
            output = torch.squeeze(output1)
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).cpu().numpy()
            pred_res.append(predict_cla)
            ground_truth.append(labels)
        print_res = "label: {}  class: {}   prob: {:.3}".format(labels, predict_cla,
                                                                predict[predict_cla].cpu().numpy())
        if int(labels) == int(predict_cla):
            trueNum = trueNum + 1
    print("trueNum: {}, allSample: {}, acc: {:.4}".format(trueNum, test_num, trueNum / test_num))

if __name__ == '__main__':
    checkpoint = torch.load("./results/CIFAR10-LT/cifar10-lt-seed0.pth", weights_only=False)
    predict(checkpoint)