from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * std + mean


class TrafficWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, tod: np.ndarray, dow: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.tod = torch.from_numpy(tod.astype(np.int64))
        self.dow = torch.from_numpy(dow.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.tod[index], self.dow[index]


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        archive = np.load(path)
        for key in ("data", "x"):
            if key in archive:
                return np.asarray(archive[key])
        return np.asarray(archive[archive.files[0]])
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported array format: {path}")


def make_windows(raw: np.ndarray, history: int, feature_index: int,
                 slots_per_day: int, days_per_week: int):
    if raw.ndim == 2:
        values = raw
    elif raw.ndim == 3:
        values = raw[:, :, feature_index]
    else:
        raise ValueError(f"Expected [time,nodes] or [time,nodes,features], got {raw.shape}")
    x, y, tod, dow = [], [], [], []
    for start in range(values.shape[0] - history):
        forecast_time = start + history
        x.append(values[start:forecast_time])
        y.append(values[forecast_time])
        tod.append(forecast_time % slots_per_day)
        dow.append((forecast_time // slots_per_day) % days_per_week)
    return np.stack(x), np.stack(y), np.asarray(tod), np.asarray(dow)


def prepare_data(config: dict):
    x, y, tod, dow = make_windows(
        load_array(config["data_path"]), config["history"], config["feature_index"],
        config["slots_per_day"], config["days_per_week"])
    n = len(x)
    train_end = int(config["split"][0] * n)
    val_end = train_end + int(config["split"][1] * n)
    mean = x[:train_end].mean(axis=(0, 1))
    std = np.maximum(x[:train_end].std(axis=(0, 1)), 1e-6)
    scaler = Standardizer(mean, std)
    x = scaler.transform(x)
    y = scaler.transform(y)
    return (TrafficWindowDataset(x[:train_end], y[:train_end], tod[:train_end], dow[:train_end]),
            TrafficWindowDataset(x[train_end:val_end], y[train_end:val_end], tod[train_end:val_end], dow[train_end:val_end]),
            TrafficWindowDataset(x[val_end:], y[val_end:], tod[val_end:], dow[val_end:]), scaler)
