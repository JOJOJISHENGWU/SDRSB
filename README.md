
# 🚦 SDRSB: Spectral-Community Dual-Stream Routing with Soft Boundaries for Traffic Prediction



SDRSB is a spectral-community dual-stream framework for traffic prediction.  
The proposed framework aims to capture complex urban traffic dynamics by jointly modeling:

- ⏱️ heterogeneous temporal periodic patterns;
- 🌐 community-level spatial interactions;
- 📡 frequency-dependent traffic propagation behaviors.

The framework is designed for intelligent transportation systems (ITS) applications, providing accurate and robust traffic forecasting under diverse traffic conditions.

---

# ✨ Overview

Traffic networks exhibit complex spatio-temporal characteristics, including:

- recurrent daily traffic patterns;
- short-term fluctuations;
- heterogeneous interactions among different urban regions;
- multi-scale congestion propagation.

To address these challenges, SDRSB introduces three complementary modules:

### 🔹 Spectral-Community Periodic Encoder (SCPE)

SCPE captures heterogeneous traffic periodicity by combining:

- spectral temporal representations;
- community-aware temporal modulation.

It enables the model to learn different periodic behaviors across functional traffic regions.

---

### 🔹 Soft Community Boundary (SCB)

SCB models traffic interactions among neighboring communities through adaptive boundary learning.

Different from rigid community separation, SCB:

- preserves intra-community traffic coherence;
- enables flexible cross-community information exchange;
- better captures regional traffic interactions.

---

### 🔹 Dual-Stream Spectral-Community Routing GCN (DSCR-GCN)

DSCR-GCN adaptively integrates:

- local spatial dependencies;
- global traffic evolution patterns.

It dynamically adjusts spatial propagation according to different traffic states.

---

# 🛠️ Environment

Recommended environment:

- 🐍 Python >= 3.10
- 🔥 PyTorch >= 2.0
- 🎮 CUDA-enabled GPU environment

Install required packages:

```bash
pip install -r requirements.txt
````

---

# 📊 Dataset

SDRSB is evaluated on widely used traffic forecasting benchmarks:

* 📍 PeMS03
* 📍 PeMS04
* 📍 PeMS07
* 📍 PeMS08
* 📍 METR-LA
* 📍 PEMS-BAY

Please download the corresponding datasets and organize them as:

```
data/
├── PeMS03/
├── PeMS04/
├── PeMS07/
├── PeMS08/
├── METR-LA/
└── PEMS-BAY/
```

---

# 🚀 Training

To train SDRSB:

```bash
python train.py --dataset PeMS04
```

The configuration can be adjusted according to:

* dataset;
* input sequence length;
* prediction horizon;
* hidden dimension;
* community number;
* learning rate.

---

# 🔍 Evaluation

Evaluate a trained model:

```bash
python test.py --dataset PeMS04 --checkpoint <checkpoint_path>
```

The model is evaluated using standard traffic forecasting metrics:

📌 MAE
📌 RMSE
📌 MAPE

---

# 🧩 Model Components

The main components of SDRSB include:

| Module   | Function                                   |
| -------- | ------------------------------------------ |
| SCPE     | Community-aware spectral temporal modeling |
| SCB      | Adaptive community interaction modeling    |
| DSCR-GCN | Local-global spatial routing               |

Together, these modules provide a unified framework for modeling heterogeneous traffic dynamics.

---

# 📈 Experiments

The experiments include:

✅ Comparison with statistical forecasting methods
✅ Comparison with GNN-based approaches
✅ Comparison with Transformer-based approaches
✅ Comparison with physics-informed traffic prediction methods
✅ Ablation studies
✅ Parameter sensitivity analysis
✅ Computational efficiency evaluation
✅ Traffic dynamics and interpretability analysis

---

# 🧪 Ablation Studies

We evaluate the contribution of each component using:

### ❌ w/o SCPE

Remove spectral-community temporal modeling.

### ❌ w/o Community

Remove community discovery to evaluate the contribution of spatial organization.

### ❌ Hard Boundary

Replace adaptive soft boundaries with rigid community separation.

### ❌ w/o DSCR-GCN

Remove adaptive spectral-community routing.

These experiments verify the effectiveness of each proposed component.

---

# ⚡ Computational Efficiency

To evaluate practical applicability, SDRSB is compared with representative baselines in terms of:

* 📦 model parameters;
* 🔢 FLOPs;
* ⏳ training time;
* 🚀 inference latency;
* 💾 GPU memory consumption.

The results demonstrate that SDRSB achieves a favorable balance between forecasting accuracy and computational cost.

---

# 🔎 Interpretability Analysis

To better understand the learned traffic representations, we conduct:

* 🚦 traffic evolution analysis;
* 🕸️ spatio-temporal receptive field analysis;
* 📊 prediction behavior visualization;
* 🌐 frequency-topology propagation analysis.

These analyses provide insights into how SDRSB captures:

* heterogeneous temporal patterns;
* regional interactions;
* traffic propagation behaviors.

---

# 📚 Citation

If you find SDRSB useful, please cite:

```bibtex
@article{SDRSB,
  title={SDRSB: Spectral-Community Dual-Stream Routing with Soft Boundaries for Traffic Prediction},
  author={},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}
```

---

# 🙏 Acknowledgement

We thank the traffic forecasting community for providing public datasets and open-source implementations that support reproducible research.

---


