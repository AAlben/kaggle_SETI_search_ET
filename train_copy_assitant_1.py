'''
1、借鉴老师代码V1
2、解读他的代码思路
 - trian_labels.csv 的读取
 -
3、我未曾用过的库
 - glob - 可使用相对路径的库？| 可按照Unix终端所使用的那般规则来查询文件等
'''


import os
import pdb
import cv2
import time
import codecs
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from loguru import logger
from colorama import Fore, Back, Style
r_ = Fore.WHITE

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from ignite.contrib.metrics import ROC_AUC
from efficientnet_pytorch import EfficientNet, utils

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import iplot
from plotly.subplots import make_subplots

# from skimage import color, data
# from skimage.util import random_noise
# from skimage.exposure import adjust_gamma
# from skimage.io import imshow, imread, imsave
# from skimage.transform import rotate, AffineTransform, warp, rescale, resize, downscale_local_mean

from snippets_dataset import SnippetsDataset, SnippetsDatasetTest, SimpleCustomBatch


logger.add('/home/alben/code/kaggle_SETI_search_ET/log/train.log', rotation="1 day")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed=123):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state
random_state = set_seed()


class EfficientNetV2(nn.Module):

    def __init__(self, backbone, out_dim):
        super(EfficientNetV2, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(backbone)
        fc_in_feature = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(fc_in_feature, out_dim)
        self.conv1_6_3 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1_6_3(x)
        x = self.efficientnet(x)
        return x


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]

    index = torch.randperm(batch_size)
    if use_cuda:
        index = index.cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.view(-1, 1)) + (1 - lam) * criterion(pred, y_b.view(-1, 1))


def train(epoch, model, train_loader, optimizer, loss_fn, lr_scheduler):
    losses_train, accuracy_train = [], []
    Y, y_pred = [], []
    model.train()
    for i, data in enumerate(train_loader):
        s_t = time.time()
        images, labels = data
        images, labels = images.type(torch.FloatTensor), labels.type(torch.FloatTensor)
        images, labels = images.to(device), labels.to(device)

        # batch_n, C, H, W = images.shape
        # images = images.view(-1, C, H, W)
        # labels = labels.repeat_interleave(crops_n)

        outputs = model(images).squeeze(1)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        Y.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(outputs.detach().cpu().numpy().tolist())

        if (i + 1) % PRINT_INTERVAL == 0:
            logger.info(f'Train - Epoch = {epoch:3}; Iteration = {i:3}; Len = {len(train_loader):3}; Loss = {np.mean(losses_train):8.4}; Acc = {metrics.roc_auc_score(Y, y_pred):8.4}; Interval = {time.time() - s_t:8.4}')
    lr_scheduler.step()
    return losses_train, metrics.roc_auc_score(Y, y_pred)


def valid(epoch, model, valide_loader):
    losses_valid, accuracy_valid = [], []
    Y, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valide_loader):
            s_t = time.time()
            images, labels = data
            images, labels = images.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            images, labels = images.to(device), labels.to(device)

            # batch_n, crops_n, C, H, W = images.shape
            # images = images.view(-1, C, H, W)

            outputs = model(images).squeeze(1)
            # outputs_avg = outputs.view(batch_n, crops_n).mean(1)
            loss = loss_fn(outputs, labels)

            losses_valid.append(loss.item())
            Y.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(outputs.detach().cpu().numpy().tolist())
        logger.info(f'Valid - Epoch = {epoch:3}; Iteration = {i:3}; Len = {len(valide_loader):3}; Loss = {np.mean(losses_valid):8.4}; Acc = {metrics.roc_auc_score(Y, y_pred):8.4}; Interval = {time.time() - s_t:8.4}')
    return losses_valid, metrics.roc_auc_score(Y, y_pred)


def test(epoch, model, test_loader, writer):
    flag = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data
            images = images.type(torch.FloatTensor)
            images = images.to(device)

            batch_n, crops_n, C, H, W = images.shape
            grid_images = images[:, 0, :, :, :]
            images = images.view(-1, C, H, W)
            outputs = model(images)
            outputs_avg = outputs.view(batch_n, crops_n).mean(1)

            if flag == 0:
                grid_images = grid_images.contiguous().view(-1, 1, grid_images.shape[-2], grid_images.shape[-1])
                grid = torchvision.utils.make_grid(grid_images, nrow=6)
                writer.add_image(f'test_{epoch}_{i}', grid)
                flag = 1
            logger.info(f'Test - Epoch = {epoch:3}; predict = {outputs_avg}')


def transforms_1(data):
    data = torch.from_numpy(data)
    ten_datas = transforms.TenCrop(224, vertical_flip=False)(data)
    return torch.stack(ten_datas)


def transforms_0(data):
    data = torch.from_numpy(data)
    data = transforms.RandomCrop(224)(data)
    return data.unsqueeze(0)


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver',
                        type=str,
                        help='procedure ver',
                        default='0')
    parser.add_argument('--comment',
                        type=str,
                        help='version comment',
                        default='first_version')
    args = parser.parse_args()
    ver = args.ver
    comment = args.comment
    logger.info(f'{"*" * 25} version = {ver}; comment = {comment} {"*" * 25}')

    EPOCH = 30
    BATCH_SIZE = 25
    PRINT_INTERVAL = 1000
    TRAIN_DATA_PATH = '/home/alben/data/cv_listen_2021_06/train'
    TEST_DATA_PATH = '/home/alben/data/cv_listen_2021_06/test'
    LABELS_CSV = '/home/alben/data/cv_listen_2021_06/train_labels.csv'
    BOARD_PATH = '/home/alben/code/kaggle_SETI_search_ET/board/train'
    MODEL_SAVE_PATH = '/home/alben/code/kaggle_SETI_search_ET/model_state'
    LR = 0.001
    LR_DECAY_STEP = 6
    NUM_CLASSES = 1
    baseline_name = 'efficientnet-b3'
    normalize_mean, normalize_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    IMG_H_W = (273, 256)
    TRAIN_VALID_RATE = 0.6

    writer = SummaryWriter(log_dir=BOARD_PATH, flush_secs=60)

    label_0_data = SnippetsDataset(TRAIN_DATA_PATH, LABELS_CSV, 0, transforms_0)
    label_0_ids = range(len(label_0_data))
    label_0_train = random.sample(label_0_ids, int(len(label_0_data) * TRAIN_VALID_RATE))
    label_0_valid = list(set(label_0_ids) - set(label_0_train))

    label_1_data = SnippetsDataset(TRAIN_DATA_PATH, LABELS_CSV, 1, transforms_1)
    label_1_ids = range(len(label_1_data))
    label_1_train = random.sample(label_1_ids, int(len(label_1_data) * TRAIN_VALID_RATE))
    label_1_valid = list(set(label_1_ids) - set(label_1_train))

    train_data = ConcatDataset([Subset(label_0_data, label_0_train), Subset(label_1_data, label_1_train)])
    train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_wrapper,
                              pin_memory=True)

    valid_data = ConcatDataset([Subset(label_0_data, label_0_valid), Subset(label_1_data, label_1_valid)])
    valid_loader = DataLoader(dataset=valid_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_wrapper,
                              pin_memory=True)

    test_data = SnippetsDatasetTest(TEST_DATA_PATH, transforms_1)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=len(test_data))

    # model = EfficientNet.from_pretrained(baseline_name)
    # fc_in_feature = model._fc.in_features
    # model._fc = nn.Linear(fc_in_feature, NUM_CLASSES, bias=True)
    model = EfficientNetV2(baseline_name, NUM_CLASSES)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.5)

    for epoch in tqdm(range(EPOCH)):
        losses_train, acc_train = train(epoch, model, train_loader, optimizer, loss_fn, lr_scheduler)
        losses_valid, acc_valid = valid(epoch, model, valid_loader)
        writer.add_scalars(f'loss_{ver}', {'train': np.mean(losses_train),
                                           'valid': np.mean(losses_valid)}, epoch)
        writer.add_scalars(f'accuracy_{ver}', {'train': acc_train,
                                               'valid': acc_valid}, epoch)
        test(epoch, model, test_loader, writer)
        torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/vggnet_cat_dog_0525_{epoch}.pth')
        logger.info(f'Summary - Epoch = {epoch:3}; loss_train = {np.mean(losses_train):8.4}; loss_valid = {np.mean(losses_valid):8.4}; acc_train = {np.mean(acc_train):8.4}; acc_valid = {np.mean(acc_valid):8.4}')
    writer.close()
