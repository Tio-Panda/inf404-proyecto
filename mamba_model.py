import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_batch
from torch.nn import LayerNorm
from mamba_ssm import Mamba

class NodePrioritization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, batch):
        node_degrees = degree(edge_index[0], x.size(0), dtype=x.dtype)

        if self.training:
            noise = torch.rand_like(node_degrees)
            score = node_degrees + noise
        else:
            score = node_degrees

        x_dense, mask = to_dense_batch(x, batch)
        scores_dense, _ = to_dense_batch(score, batch)

        scores_dense[~mask] = -1e9

        sorted_indices = torch.argsort(scores_dense, dim=1, descending=True)

        batch_idx = torch.arange(x_dense.size(0), device=x.device).unsqueeze(1)
        x_sorted = x_dense[batch_idx, sorted_indices]
        
        return x_sorted, mask, sorted_indices

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super().__init__()

        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        out_fwd = self.mamba_fwd(x)

        x_rev = torch.flip(x, dims=[1])
        out_rev = self.mamba_bwd(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])

        combined = torch.cat([out_fwd, out_rev], dim=-1)
        out = self.output_proj(combined)

        return self.norm(out + self.dropout(x))

class NeuroBackMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.norm_in = LayerNorm(hidden_dim)

        self.prioritizer = NodePrioritization()

        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.embedding(x)
        h = self.norm_in(h)

        h_seq, mask, sort_idx = self.prioritizer(h, edge_index, batch)

        for layer in self.layers:
            h_seq = layer(h_seq)

        undo_sort_indices = torch.argsort(sort_idx, dim=1)
        batch_idx = torch.arange(h_seq.size(0), device=x.device).unsqueeze(1)

        h_restored = h_seq[batch_idx, undo_sort_indices]

        _, mask_orig = to_dense_batch(torch.zeros(x.size(0), device=x.device), batch)

        final_h = h_restored[mask_orig]

        out = self.classifier(final_h)
        return out
