import gzip
import json
import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Sampler

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

class SortedBucketSampler(Sampler):
    """
    Sampler que cachea los tamaños de los grafos en el directorio especificado.
    Respeta la jerarquía fija: ./data/pt/pretrain/ o ./data/pt/validation/
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_path = "./cache/cache_file.pt"

        # --- Lógica de Caché ---
        if os.path.exists(self.cache_path):
            print(f"⚡ Caché detectado en: {self.cache_path}")
            self.lengths = torch.load(self.cache_path, weights_only=False)
            
            # Validación de integridad
            if len(self.lengths) != len(dataset):
                print("⚠️ El caché no coincide con el dataset actual. Recalculando...")
                self.lengths = self._calculate_lengths()
        else:
            print(f"🐢 Generando caché de metadatos en: {self.cache_path}")
            self.lengths = self._calculate_lengths()
        
        # Índices ordenados por tamaño (para agrupar grafos similares)
        self.sorted_indices = np.argsort(self.lengths)

    def _calculate_lengths(self):
        # Leemos el tamaño de cada grafo UNO por UNO (esto tarda, pero solo la primera vez)
        lengths = []
        for i in tqdm(range(len(self.dataset)), desc="Indexando tamaños"):
            lengths.append(self.dataset[i].num_nodes)
        
        lengths = np.array(lengths)
        # Guardamos en tu directorio fijo
        torch.save(lengths, self.cache_path)
        print(f"💾 Caché guardado exitosamente.")
        return lengths

    def __iter__(self):
        # 1. Agrupar índices por tamaño (Bucketing)
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        # 2. Barajar los grupos (Shuffle) si es training
        if self.shuffle:
            np.random.shuffle(batches)
            
        # 3. Entregar índices planos
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)
