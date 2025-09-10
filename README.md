# GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents
Research code for the paper "GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents".

Paper link: [https://arxiv.org/abs/2505.12842](https://arxiv.org/abs/2505.12842)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Wuzheng02/GEM-OODforGUIagents
cd GEM-OODforGUIagents
```

### 2. Run Evaluation

#### Example: AITZ (ID) vs. OmniAct-Desktop (OOD)

To evaluate GEM on the AITZ train set (ID) and test using AITZ test (ID) and OmniAct-Desktop test (OOD):

1. **Extract input scores** (for both ID and OOD datasets):

   ```bash
   python run.py
   ```

2. **Fit GMM and perform OOD detection**:

   ```bash
   python GEM.py
   ```

> 🔍 Note: Baseline methods (e.g., MSP, Energy, Mahalanobis) are also available in `run.py` (see commented sections).


## 📋 Citation

```bibtex
@article{wu2025gem,
  title={GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents},
  author={Wu, Zheng and Cheng, Pengzhou and Wu, Zongru and Dong, Lingzhong and Zhang, Zhuosheng},
  journal={arXiv preprint arXiv:2505.12842},
  year={2025}
}
```
