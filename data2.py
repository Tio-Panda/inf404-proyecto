from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

class PtDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        root = Path(root_dir)
        self.file_paths = sorted(
            [p for p in root.iterdir() if p.is_file()],
            key=lambda p: p.stat().st_size
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])

        reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
        data.edge_index = torch.cat([data.edge_index, reverse], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)

        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        data.edge_attr = data.edge_attr.float()

        if data.y != None:
            data.y = data.y.long()

        return data
