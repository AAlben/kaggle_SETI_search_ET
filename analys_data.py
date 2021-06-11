'''
Q：
1、每个文件夹都有多少个文件 - 是否匹配
2、一个文件长什么样子？
'''


import os
import ipdb
import numpy as np
import pandas as pd


import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision


DATA_PATH = '/home/alben/data/cv_listen_2021_06/train'
BOARD_PATH = '/home/alben/code/cv_seti_listen/board/data'
writer = SummaryWriter(log_dir=BOARD_PATH)


def check_count():
    for folder in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(path):
            continue
        l = os.listdir(path)
        print(f'{folder}"s files length = {len(l)}')

'''
f"s files length = 3151
a"s files length = 3087
d"s files length = 3048
8"s files length = 3086
c"s files length = 3167
2"s files length = 3145
7"s files length = 3161
9"s files length = 3139
5"s files length = 3126
3"s files length = 3115
1"s files length = 3144
4"s files length = 3123
0"s files length = 3145
b"s files length = 3127
e"s files length = 3170
6"s files length = 3231
'''


def show_data():
    file = '/home/alben/data/cv_listen_2021_06/train/f/fff66bbc51db.npy'
    data = np.load(file)  # [6, 273, 256]
    ipdb.set_trace()
    data = torch.from_numpy(data)  # [6, 273, 256]
    data = data.unsqueeze(1)  # [6, 1, 273, 256]
    grid = torchvision.utils.make_grid(data, nrow=2)
    writer.add_image(f'plot_0', grid)
    writer.close()
show_data()


def analys_labels():
    '''
    [得出的结论]
    1、label = 1 的占比 = 0.093571；不到10% | 类别不均衡
    2、一共50165条 - 5W条数据
    3、数据无空值 - 也没有空值的必要
    '''

    labels_file = '/home/alben/data/cv_listen_2021_06/train_labels.csv'
    df = pd.read_csv(labels_file)
    target_desc = df['target'].value_counts()
    '''
    0    45471
    1     4694
    Name: target, dtype: int64
    '''
    df.info()
    '''
    RangeIndex: 50165 entries, 0 to 50164
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   id      50165 non-null  object
     1   target  50165 non-null  int64
    dtypes: int64(1), object(1)
    '''
    df.describe()
    '''
                 target
    count  50165.000000
    mean       0.093571
    std        0.291234
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        0.000000
    max        1.000000
    '''
