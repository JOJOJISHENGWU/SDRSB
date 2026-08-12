from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .graph import SoftCommunityBoundary, normalize_adjacency


class SpectralPeriodSelector(nn.Module):
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor, communities: torch.Tensor, node_weights: torch.Tensor):
        spectrum = torch.fft.rfft(x, dim=1).abs()              # [B,F,N]
        community_count = int(communities.max()) + 1
        score = torch.zeros(spectrum.shape[1], device=x.device, dtype=x.dtype)
        community_weights = torch.zeros(
            x.shape[0], community_count, spectrum.shape[1], device=x.device, dtype=x.dtype)
        for community in range(community_count):
            nodes = communities == community
            if not nodes.any():
                continue
            weights = node_weights[nodes]
            weights = weights / weights.sum().clamp_min(1e-8)
            community_weights[:, community] = (spectrum[:, :, nodes] * weights[None, None, :]).sum(-1)
            score = score + community_weights[:, community].mean(0)
        score[0] = -torch.inf
        count = min(self.top_k, score.numel() - 1)
        _, indices = torch.topk(score, count)
        periods = (x.shape[1] / indices.clamp_min(1)).long()
        if count < self.top_k:
            periods = F.pad(periods, (0, self.top_k - count), value=1)
            node_weights_fft = spectrum[:, indices, :].permute(0, 2, 1)
            node_weights_fft = F.pad(node_weights_fft, (0, self.top_k - count), mode="replicate")
            community_weights = F.pad(community_weights[:, :, indices], (0, self.top_k - count), mode="replicate")
        else:
            node_weights_fft = spectrum[:, indices, :].permute(0, 2, 1)
            community_weights = community_weights[:, :, indices]
        return periods.tolist(), node_weights_fft, community_weights


class InceptionBlock(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel, padding=kernel // 2)
            for kernel in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([conv(x) for conv in self.convs], dim=-1).mean(-1)


class SCPE(nn.Module):
    def __init__(self, history: int, top_k: int, d_model: int, communities: int,
                 layers: int = 3, dropout: float = 0.1, use_modulation: bool = True):
        super().__init__()
        self.history = history
        self.top_k = top_k
        self.use_modulation = use_modulation
        self.selector = SpectralPeriodSelector(top_k)
        self.input_proj = nn.Linear(1, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.period_linear = nn.Linear(history, history + 1)
        self.blocks = nn.ModuleList([InceptionBlock(d_model) for _ in range(layers)])
        self.community_embedding = nn.Embedding(communities, d_model // 4)
        self.modulation = nn.Sequential(
            nn.Linear(d_model // 4, d_model), nn.GELU(), nn.Linear(d_model, 2 * d_model)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(history + 1, 1)

    def forward(self, x: torch.Tensor, communities: torch.Tensor, node_weights: torch.Tensor):
        periods, node_period_weights, community_period_weights = self.selector(x, communities, node_weights)
        batch, _, nodes = x.shape
        hidden = self.input_norm(self.input_proj(x.permute(0, 2, 1).reshape(batch * nodes, -1, 1)))
        extended = self.period_linear(hidden.transpose(1, 2)).transpose(1, 2)
        context_ids = communities.repeat(batch)
        branches = []
        for period in periods:
            length = extended.shape[1]
            padded_length = ((length + period - 1) // period) * period
            padded = F.pad(extended, (0, 0, 0, padded_length - length))
            grid = padded.reshape(batch * nodes, padded_length // period, period, -1)
            grid = grid.permute(0, 3, 1, 2)
            for block in self.blocks:
                grid = F.gelu(block(grid))
            if self.use_modulation:
                params = self.modulation(self.community_embedding(context_ids))
                gamma, beta = params.chunk(2, -1)
                grid = grid * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
            branches.append(grid.permute(0, 2, 3, 1).reshape(batch * nodes, -1, grid.shape[1])[:, :length])
        features = torch.stack(branches, -1)
        weights = F.softmax(node_period_weights.reshape(batch * nodes, self.top_k), -1)
        features = (features * weights[:, None, None, :]).sum(-1) + extended
        features = self.dropout(self.output_norm(features))
        output = self.output_linear(features.transpose(1, 2)).transpose(1, 2)
        return output.reshape(batch, nodes, 1, -1), periods, community_period_weights


class RoutingNetwork(nn.Module):
    def __init__(self, top_k: int, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(top_k, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1)
        )
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(self, community_weights: torch.Tensor, communities: torch.Tensor):
        logits = self.mlp(F.softmax(community_weights, -1)).squeeze(-1)
        alpha_community = torch.sigmoid(self.temperature.clamp(0.1, 20) * logits)
        return alpha_community, alpha_community[:, communities]


class DualStreamGCN(nn.Module):
    def __init__(self, d_in: int, d_out: int, communities: int, k_high: int = 1, k_low: int = 3):
        super().__init__()
        self.high = nn.Parameter(torch.randn(k_high, d_in, d_out) * 0.02)
        self.low = nn.Parameter(torch.randn(k_low, d_in, d_out) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.boundary = SoftCommunityBoundary(communities)

    @staticmethod
    def cheb(x, adj, weight):
        result = x @ weight[0]
        if weight.shape[0] == 1:
            return result
        previous, current = x, torch.einsum("nm,bmf->bnf", adj, x)
        result = result + current @ weight[1]
        for order in range(2, weight.shape[0]):
            nxt = 2 * torch.einsum("nm,bmf->bnf", adj, current) - previous
            result = result + nxt @ weight[order]
            previous, current = current, nxt
        return result

    def forward(self, x, adjacency, communities, alpha, ablation="none"):
        global_adj = normalize_adjacency(adjacency)
        boundary = self.boundary(communities, hard=ablation == "hard_boundary")
        local_adj = normalize_adjacency(adjacency * boundary)
        high = self.cheb(x, local_adj, self.high)
        low = self.cheb(x, global_adj, self.low)
        if ablation == "single_kernel_global":
            alpha = torch.zeros_like(alpha)
        elif ablation == "fixed_routing":
            alpha = torch.full_like(alpha, 0.5)
        output = low + alpha[:, :, None] * (high - low) + self.bias
        return output, high, low


class SDRSB(nn.Module):
    def __init__(self, nodes: int, communities: int, history: int = 12, top_k: int = 3,
                 d_model: int = 96, d_gcn: int = 32, inception_layers: int = 3,
                 beta_warmup_steps: int = 200, use_modulation: bool = True,
                 k_high: int = 1, k_low: int = 3):
        super().__init__()
        self.nodes = nodes
        self.beta_warmup_steps = beta_warmup_steps
        self.global_step = 0
        self.scpe = SCPE(history, top_k, d_model, communities, inception_layers,
                         use_modulation=use_modulation)
        self.temporal_readout = nn.Linear(d_model, 1)
        self.adapter = nn.Linear(d_model, d_gcn)
        self.routing = RoutingNetwork(top_k)
        self.gcn = DualStreamGCN(d_gcn, d_gcn, communities, k_high, k_low)
        self.spatial_readout = nn.Linear(d_gcn, 1)
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.b2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adjacency, communities, ablation="none"):
        self.global_step += 1
        if ablation == "no_community":
            communities = torch.zeros_like(communities)
        degree = adjacency.sum(-1)
        node_weights = degree / degree.sum().clamp_min(1e-8)
        temporal, periods, community_weights = self.scpe(x, communities, node_weights)
        temporal = temporal[:, :, 0, :]
        temporal_prediction = self.temporal_readout(temporal).squeeze(-1)
        alpha_community, alpha_node = self.routing(community_weights, communities)
        spatial_hidden, high, low = self.gcn(
            self.adapter(temporal), adjacency, communities, alpha_node, ablation)
        spatial = self.spatial_readout(spatial_hidden).squeeze(-1)
        warmup = min(1.0, self.global_step / max(self.beta_warmup_steps, 1))
        injected = temporal_prediction + self.beta.clamp(0, 5) * warmup * spatial
        gate = torch.sigmoid(self.w1 * temporal_prediction + self.b1 + self.w2 * injected + self.b2)
        prediction = gate * temporal_prediction + (1 - gate) * injected
        diagnostics = {"periods": periods, "alpha_community": alpha_community,
                       "alpha_node": alpha_node, "gate": gate, "high": high, "low": low}
        return prediction, diagnostics
