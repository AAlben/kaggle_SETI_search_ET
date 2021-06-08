import os
import pdb
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SnippetsDataset(Dataset):

    def __init__(self, data_path, labels_csv_file, mode, transform=None, train_valid_rate=0.8, seed=123):
        random.seed(seed)
        self.seed = seed
        self.data_path = data_path
        self.mode = mode
        self.train_valid_rate = train_valid_rate
        self.transform = transform
        self.files, self.labels = self.load(labels_csv_file)

    def __getitem__(self, index):
        file, label = self.files[index], self.labels[index]
        data = np.load(file).astype(float)  # before - dtype('float16'); after - dtype('float64')
        data = data / np.array([np.abs(data).max() for i in range(6)]).reshape(6, 1, 1)

        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.files)

    def load(self, labels_csv_file):
        df = pd.read_csv(labels_csv_file)
        df['file'] = df['id'].apply(lambda x: os.path.join(self.data_path, x[0], f'{x}.npy'))
        if self.mode == 'train':
            df = df.sample(frac=self.train_valid_rate, random_state=self.seed)
        elif self.mode == 'valid':
            df = df.sample(frac=1 - self.train_valid_rate, random_state=self.seed)
        else:
            raise Exception('', '')
        return df.file.tolist(), df.target.tolist()


class SnippetsDatasetTest(Dataset):

    def __init__(self, data_path, transform=None, seed=123):
        random.seed(seed)
        self.data_path = data_path
        self.transform = transform
        self.files = self.load()

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file).astype(float)
        data = data / np.array([np.abs(data).max() for i in range(6)]).reshape(6, 1, 1)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.files)

    def load(self):
        files = []
        for folder in os.listdir(self.data_path):
            if not os.path.isdir(os.path.join(self.data_path, folder)):
                continue
            folder_path = os.path.join(self.data_path, folder)
            folder_files = os.listdir(folder_path)
            random_file = random.choice(folder_files)
            files.append(os.path.join(folder_path, folder_files[0]))
            files.append(os.path.join(folder_path, random_file))
        return files
