from __future__ import annotations

import torch
import torch.nn.functional as F


def spectral_consistency(x_history: torch.Tensor, target: torch.Tensor,
                         prediction: torch.Tensor) -> torch.Tensor:
    true_sequence = torch.cat([x_history, target[:, None, :]], dim=1)
    predicted_sequence = torch.cat([x_history, prediction[:, None, :]], dim=1)
    return (torch.fft.rfft(true_sequence, dim=1).abs() -
            torch.fft.rfft(predicted_sequence, dim=1).abs()).abs().mean()


def orthogonality(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    high_vector = high.mean(0).flatten()
    low_vector = low.mean(0).flatten()
    return F.cosine_similarity(high_vector[None], low_vector[None]).pow(2).squeeze()


def composite_loss(model, x, target, prediction, diagnostics,
                   lambda_spec=0.02, lambda_sep=0.05, lambda_orth=0.01):
    prediction_term = F.l1_loss(prediction, target) + 0.3 * F.smooth_l1_loss(prediction, target)
    spectral_term = spectral_consistency(x, target, prediction)
    separation_term = -diagnostics["alpha_community"].var(dim=1, unbiased=False).mean()
    orthogonal_term = orthogonality(model.gcn.high, model.gcn.low)
    total = (prediction_term + lambda_spec * spectral_term +
             lambda_sep * separation_term + lambda_orth * orthogonal_term)
    return total, {"prediction": prediction_term.detach(), "spectral": spectral_term.detach(),
                   "separation": separation_term.detach(), "orthogonality": orthogonal_term.detach()}


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    valid = torch.isfinite(prediction) & torch.isfinite(target) & (target >= 10)
    mape = (error[valid].abs() / target[valid].abs().clamp_min(1e-8)).mean() * 100 if valid.any() else torch.tensor(0.0)
    return {"MAE": error.abs().mean().item(),
            "RMSE": error.square().mean().sqrt().item(),
            "MAPE": mape.item()}
