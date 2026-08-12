# Equation and implementation traceability

| Paper location | Technical meaning | Pseudocode | Reference code |
|---|---|---|---|
| Eqs. (1)-(3) | RFFT, community-aware energy, top-k period selection | `03_scpe.pseudo::select_spectral_periods` | `timesnet.py::FFT_for_Period` |
| Eqs. (4)-(5) | padding, period-grid reshape, Inception convolutions | `03_scpe.pseudo::inception_period_branch` | `timesnet.py::TimesBlock._forward_global`, `Inception_Block_V1` |
| Eqs. (6)-(7) | community embedding and FiLM modulation | `03_scpe.pseudo::community_modulation` | `timesnet.py::ContextModulation` |
| Eqs. (8)-(9) | energy-weighted branch fusion and projection | `03_scpe.pseudo::SCPE` | `timesnet.py::TimesBlock` |
| Eqs. (10)-(12) | normalized Laplacian and spectral K-means | `02_spectral_community_discovery.pseudo` | `build_community_spectral.py` |
| Eqs. (13)-(17) | learnable cross-community permeability | `04_scb.pseudo` | `soft_community_boundary.py` |
| Eqs. (18)-(20) | graph normalization and Chebyshev recursion | `00_symbols_and_contracts.pseudo`, `05_dscr_gcn.pseudo` | `fcd_gc.py::_normalize_adj`, `_cheb_conv` |
| Eqs. (21)-(22) | local `K=1` and global `K=3` streams | `05_dscr_gcn.pseudo::DSCR_GCN` | `fcd_gc.py::FrequencyCommunityDualKernelGC` |
| Eqs. (23)-(26) | community MLP routing and node expansion | `05_dscr_gcn.pseudo::community_routing` | `community_routing_net.py`, `train.py::STModel.forward` |
| Eqs. (27)-(28) | dual-stream interpolation and spatial readout | `05_dscr_gcn.pseudo::DSCR_GCN` | `fcd_gc.py::forward`, `train.py::spatial_readout` |
| Eqs. (29)-(31) | residual injection and gated fusion | `06_sdrsb_forward.pseudo` | `train.py::STModel.forward`, `gatedFusion` |
| Eqs. (32)-(34) | MAE, SmoothL1, spectral consistency | `07_objective_and_training.pseudo` | `train.py::spectral_consistency_loss` and training loop |
| Eqs. (35)-(36) | routing separation and kernel orthogonality | `05_dscr_gcn.pseudo` | `community_routing_net.py::separation_loss`, `fcd_gc.py::orthogonality_loss` |
| Eq. (37) | weighted total objective | `07_objective_and_training.pseudo::total_loss` | `train.py::train_and_evaluate` |
| Algorithm 1 | complete train/inference flow and complexity | `01` through `08` | `train.py`, `timesnet.py`, `fcd_gc.py` |
