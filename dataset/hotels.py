import os
import pickle

import PIL.Image
import numpy as np
import pandas as pd
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.train_lbl2id = {}

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1:
                im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        if self.train_lbl2id.get(self.ys[index]):
            target = self.train_lbl2id.get(self.ys[index])
        else:
            target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class Hotels(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.mode = mode
        self.root = root
        with open(os.path.join(self.root, '/v5_splits/train_lbl2id.pkl'), 'rb') as f:
            self.train_lbl2id = pickle.load(f)

        if mode == 'train':
            self.config_file = pd.read_csv(os.path.join(self.root, '/v5_splits/train_small.csv'))
        elif self.mode == 'eval':
            self.config_file = pd.read_csv(os.path.join(self.root, '/v5_splits/val1_small.csv'))

        self.transform = transform
        print('getting classes')
        self.classes = np.unique(self.config_file.label)
        # if self.mode == 'train':
        #     self.classes = range(0, 100)
        # elif self.mode == 'eval':
        #     self.classes = range(100, 200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        self.ys = list(self.config_file.label)
        self.I = [i for i in range(len(self.ys))]
        relative_im_paths = list(self.config_file.image)
        self.im_paths = [os.path.join(self.root, i) for i in relative_im_paths]
