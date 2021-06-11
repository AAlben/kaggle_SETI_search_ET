'''
1、加载模型
2、读取test数据集
3、进行预测
4、然后输出到submission.csv中
'''


import os
import pdb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from efficientnet_pytorch import EfficientNet, utils

from snippets_dataset import SnippetsDataset, SnippetsDatasetTest, SimpleCustomBatch


logger.add('/home/alben/code/kaggle_SETI_search_ET/log/inference.log', rotation="1 day")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def test(model, test_loader):
    indexes, predictions = [], []
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            images, files = data
            images = images.type(torch.FloatTensor)
            images = images.to(device)

            batch_n, crops_n, C, H, W = images.shape
            grid_images = images[:, 0, :, :, :]
            images = images.view(-1, C, H, W)
            outputs = model(images)
            outputs_avg = outputs.view(batch_n, crops_n).mean(1)
            predictions.append(outputs_avg)
            # outputs_avg = outputs_avg.detach().cpu().numpy().tolist()
            # predictions.extend(outputs_avg)
            indexes.extend(list(files))
    return predictions, indexes


def transforms_1(data):
    data = torch.from_numpy(data)
    ten_datas = transforms.TenCrop(224, vertical_flip=False)(data)
    return torch.stack(ten_datas)


def transforms_0(data):
    data = torch.from_numpy(data)
    data = transforms.RandomCrop(224)(data)
    return data.unsqueeze(0)

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

    BATCH_SIZE = 25
    TEST_DATA_PATH = '/home/alben/data/cv_listen_2021_06/test'
    MODEL_SAVE_PATH = '/home/alben/code/kaggle_SETI_search_ET'
    MODEL_FILE = 'efficientnet_SETI_0609_4.pth'
    SUBMISSION_PATH = '/home/alben/code/kaggle_SETI_search_ET/submissions'
    LR = 0.005
    LR_DECAY_STEP = 6
    NUM_CLASSES = 1
    baseline_name = 'efficientnet-b2'
    normalize_mean, normalize_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    IMG_H_W = (273, 256)
    TRAIN_VALID_RATE = 0.9

    test_data = SnippetsDatasetTest(TEST_DATA_PATH, 'inference', transforms_1)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE)

    model_path = os.path.join(MODEL_SAVE_PATH, MODEL_FILE)
    # model = EfficientNetV2(baseline_name, NUM_CLASSES)
    # model.load_state_dict(torch.load(model_path))
    # model.to(device)
    model = EfficientNet.from_pretrained(baseline_name)
    fc_in_feature = model._fc.in_features
    model._fc = nn.Linear(fc_in_feature, NUM_CLASSES, bias=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions, indexes = test(model, test_loader)
    predictions = torch.cat(predictions)
    outputs = torch.nn.Sigmoid()(predictions)
    outputs = outputs.detach().cpu().numpy()
    files = [os.path.splitext(os.path.basename(file))[0] for file in indexes]

    df = pd.DataFrame({'id': files, 'target': outputs})
    df.to_csv(os.path.join(SUBMISSION_PATH, f'submission_{ver}.csv'), index=False)
