import gzip
import json
import os
import pickle

import torch
from torch_geometric.data import Data, Dataset


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return sorted(list(os.listdir(self.processed_dir)), key=lambda fn: os.path.getsize(self.processed_dir + "/" + fn))

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)
        return data
