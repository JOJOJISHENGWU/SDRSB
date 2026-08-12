# Paper-first alignment notes

This file exposes—not hides—the points at which a concise manuscript equation and the supplied experimental implementation have different levels of detail.

## Naming

The manuscript uses **SCPE**, **SCB**, and **DSCR-GCN**. Some source docstrings retain earlier labels such as MSSE, CACM, CRN, or DCR-GCN. This repository consistently uses the manuscript names and treats the older labels as implementation subcomponents.

## Community spectral aggregation

Paper Eq. (2) is compact and its indexing can be read ambiguously. The implementation computes an FFT magnitude `[B,F,N]`, aggregates nodes inside each community, and sums community contributions into a frequency score. The pseudocode uses this shape-correct interpretation.

## Spectral clustering eigenvectors

Paper Eq. (12) denotes the first `C` smooth eigenvectors. The supplied builder explicitly skips the trivial zero-eigenvalue constant vector and clusters the next `C` vectors. The pseudocode records the executable builder behavior and notes the paper convention.

## Prediction horizon

The methodology defines general `T_p`; the reference `STModel` sets `pred_len=1` and creates one-step targets. The repository retains a horizon parameter in SCPE contracts but the complete forward/training path clearly selects the one-step implementation. Multi-horizon experiments should either use a recursive/direct extension or the separate experiment scripts; no unsupported mechanism is invented here.

## Gated fusion parameter sharing

Eqs. (30)-(31) describe shared scalar gate parameters, which preserve permutation equivariance. The reference `gatedFusion(num_nodes)` uses node-mixing linear layers of width `N`. Because the paper is the primary authority, the main pseudocode specifies the shared-scalar gate and flags the source implementation here. A reproduction of the exact current source can replace it with `Linear(N,N)` layers.

## Persistence and calendar refinement

The reference `STModel.forward` adds the latest normalized observation after temporal/spatial fusion and, when calendar indices are supplied, applies trend/season refinement. These operations are not separately formalized in Eqs. (1)-(31), but they are part of the actual training path. They are included explicitly in `06_sdrsb_forward.pseudo` and labeled implementation-level operations.

## Graph-convolution orientation

Paper equations write `T_k(A) X W`. The implementation propagates using `adj_norm.T` inside `einsum`. For symmetric traffic adjacency and symmetric normalization these are equivalent. The pseudocode uses the paper notation while requiring a symmetric normalized graph.

## Loss coefficients

The source `spectral_consistency_loss(..., alpha=0.02)` returns an already weighted value. The pseudocode defines the unweighted spectral discrepancy and multiplies it by `0.02` once. Separation loss similarly includes its configured `sep_weight`; it must not be multiplied a second time.

## Dataset ratios

The experiment settings text states `7:2:1`, whereas the supplied `train.py` uses `0.7/0.1/0.2`. The reviewer pseudocode follows the executable reference split and records the discrepancy rather than claiming both are identical. Before public release, the authors should choose one protocol and make the manuscript, scripts, and reported results consistent.

## Learning rate and batch size

The manuscript states learning rate `1e-4` and batch size `16`; the inspected reference `train.py` defaults to AdamW learning rate `1e-3` and batch size `32`. These are experiment-protocol differences, not architectural differences. `07_objective_and_training.pseudo` follows the inspected reference and this note makes the difference auditable.
