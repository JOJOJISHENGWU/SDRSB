from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    adj = torch.maximum(adj, adj.transpose(0, 1))
    adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    degree = adj.sum(-1).clamp_min(1e-8)
    return adj / torch.sqrt(degree[:, None] * degree[None, :])


def spectral_communities(adj: np.ndarray, number_of_communities: int, seed: int = 42) -> np.ndarray:
    adj = np.maximum(adj, adj.T)
    degree = adj.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(adj.shape[0]) - inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)[1:number_of_communities + 1]
    embedding = eigenvectors[:, order]
    return _kmeans(embedding, number_of_communities, seed)


def _kmeans(features: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(features.shape[0], clusters, replace=False)].copy()
    for _ in range(100):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = distances.argmin(axis=1)
        updated = np.stack([
            features[labels == cluster].mean(0) if np.any(labels == cluster) else centers[cluster]
            for cluster in range(clusters)
        ])
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels


def load_or_build_communities(adj_path: str | Path, community_path: str | Path,
                               number_of_communities: int, seed: int = 42) -> torch.Tensor:
    community_path = Path(community_path)
    if community_path.exists():
        communities = np.load(community_path)
    else:
        communities = spectral_communities(np.load(adj_path), number_of_communities, seed)
        community_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(community_path, communities)
    return torch.as_tensor(communities, dtype=torch.long)


class SoftCommunityBoundary(torch.nn.Module):
    def __init__(self, number_of_communities: int, init_inter: float = -3.0):
        super().__init__()
        self.g_inter = torch.nn.Parameter(torch.full((number_of_communities,), init_inter))

    def forward(self, communities: torch.Tensor, hard: bool = False) -> torch.Tensor:
        same = (communities[:, None] == communities[None, :]).to(self.g_inter.dtype)
        if hard:
            return same
        permeability = torch.sigmoid(self.g_inter)[communities]
        return same + (1 - same) * permeability[:, None] * permeability[None, :]
