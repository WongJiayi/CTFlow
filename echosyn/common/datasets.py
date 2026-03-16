import os
import random

import torch
from torch.utils.data import Dataset


class LatentBlockDataset(Dataset):
    """
    Dataset for auto-regressive block-wise training.

    Loads pre-encoded latent volumes and text embeddings.
    Returns consecutive block pairs (current block as condition, next block as target).
    """

    def __init__(self, root_dir, embedding_dir, block_size=16):
        self.root_dir = root_dir
        self.embedding_dir = embedding_dir
        self.block_size = block_size

        self.file_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])

        self.embedding_paths = [
            os.path.join(embedding_dir, os.path.basename(f))
            for f in self.file_paths
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        latent_path = self.file_paths[idx]
        embed_path = self.embedding_paths[idx]

        latent = torch.load(latent_path, map_location="cpu")   # [C, T, H, W]
        embedding = torch.load(embed_path, map_location="cpu") # [N, D]
        embedding = embedding[0].unsqueeze(0)
        embedding = embedding / (embedding.norm(p=2) + 1e-6)

        C, T, H, W = latent.shape
        max_start = T - 2 * self.block_size

        # 50% chance to sample from the start (bias toward beginning of volume)
        if random.random() < 0.5:
            t = 0
        else:
            t = random.randint(0, max_start)

        block_curr = latent[:, t:t + self.block_size]
        block_next = latent[:, t + self.block_size:t + 2 * self.block_size]

        return {
            "image": block_curr,     # condition: [C, T, H, W]
            "video": block_next,     # target:    [C, T, H, W]
            "embedding": embedding,  # text embedding: [1, D]
        }


def instantiate_dataset(configs, split=None):
    datasets = []
    for cfg in configs:
        if not cfg.get("active", False):
            continue
        name = cfg.name
        params = dict(cfg.params)

        if name == "LatentBlock":
            dataset = LatentBlockDataset(**params)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(datasets)
