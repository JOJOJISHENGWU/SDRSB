# End-to-end architecture trace

## 1. Inputs and preprocessing

A sample is a normalized tensor `X[B, 12, N]`, its one-step target `Y[B, N]`, time-of-day/day-of-week indices, the physical adjacency `A[N, N]`, and offline spectral community IDs `c[N]`. Splitting is chronological. Normalization statistics come from the training history only.

## 2. SCPE temporal path

1. RFFT is applied along the 12-step temporal dimension.
2. DC is excluded and spectral magnitude is aggregated using node and community importance.
3. Top-`k` frequency indices are converted to integer periods.
4. Each node is treated as an independent temporal sample and projected to `d_model`.
5. Every selected period forms a 2-D period grid. Parallel odd-sized Inception kernels extract multi-scale patterns.
6. Community embeddings generate FiLM scale/shift parameters for each periodic branch.
7. Node-specific spectral energy softmaxes weight the `k` branches.
8. A residual, normalization, dropout, and projection produce the temporal embedding and temporal prediction.
9. The same FFT produces `P_comm[B,C,k]`, which is passed to DSCR-GCN routing. This is the explicit spectral-spatial coupling.

## 3. SCB graph path

Spectral clustering is performed once on the normalized Laplacian. During training, SCB maps learnable community logits to permeability values. Same-community edge multipliers are one; cross-community multipliers are products of the two community permeabilities. Multiplication by the physical adjacency preserves graph support.

## 4. DSCR-GCN spatial path

- The local/high-frequency stream uses `A_local = A ⊙ B_soft` and order `K_high=1`.
- The global/low-frequency stream uses full `A` and order `K_low=3`.
- A routing MLP consumes the normalized community spectral weights and generates `alpha_comm[B,C]`.
- `alpha_comm[:, c]` expands routing to nodes.
- `H = H_global + alpha * (H_local - H_global)` makes routing semantics unambiguous: one selects local; zero selects global.

## 5. Prediction fusion

The spatial stream is a correction to the temporal prediction. Its coefficient is warmed from zero to the learned `beta`, then temporal and injected predictions are gated. The reference training path optionally refines this result with time-of-day/day-of-week decomposition and adds the latest observed value as a persistence residual.

## 6. Optimization

The objective combines prediction accuracy, spectral consistency, routing diversity, and stream complementarity. Validation MAE selects the checkpoint. Test metrics are computed once after restoring the best checkpoint and denormalizing predictions.
