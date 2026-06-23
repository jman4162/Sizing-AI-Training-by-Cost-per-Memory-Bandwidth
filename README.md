# Sizing AI Training by **Cost per Memory Bandwidth**

*A practical, first-order model (math + Python) to tell if you’re compute-, memory-, or network-bound—and how to pick the cheapest TB/s that hits your tokens/sec target.*

> Notebook: **Sizing\_AI\_Training\_by\_Cost\_per\_Memory\_Bandwidth.ipynb** (this repo). ([GitHub][1])

## Why this exists

Large transformer training is often shaped by data movement through **HBM/GDDR**, which can matter as much as peak TFLOPs. The binding constraint depends on the workload, though: some phases are compute-bound, some memory-bound, some network-bound. This project provides a compact model—both in math and code—to tell which:

* Diagnose whether a run is **compute**, **memory**, or **network** bound
* Estimate **tokens/sec per GPU**, GPUs needed for a target throughput, and cluster **TB/s**
* Compare hardware using **\$/TB/s/hour** (cost per memory bandwidth), which often tracks throughput/\$ better than TFLOPs/\$ for large LLM training

## What’s inside

* 📓 **Notebook** with the derivation + reference implementation
* 🧮 **Equations** for FLOPs/token, bytes/token (optimizer + activations), arithmetic intensity, and network-bound checks
* 🧰 **Tunable knobs** for FlashAttention, activation checkpointing, optimizer precision, global tokens/step, etc.
* 🧪 **Example catalog** entries for common GPUs (editable to your pricing/specs)

---

## Quickstart

```bash
# 1) Clone
git clone https://github.com/jman4162/Sizing-AI-Training-by-Cost-per-Memory-Bandwidth
cd Sizing-AI-Training-by-Cost-per-Memory-Bandwidth

# 2) (Recommended) Create an environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install minimal deps for running the notebook
python -m pip install --upgrade pip jupyterlab

# 4) Launch and open the notebook
jupyter lab
```

> The notebook uses only the standard library (`dataclasses`, `math`). If you add plots, install `matplotlib` too.

---

## Usage pattern

1. **Fill in your run**

* Model size $N$, layers $L$, hidden size $d_{\text{model}}$
* Global tokens per step $B_g$ (global batch × sequence length)
* Optimizer traffic $\alpha_{\text{opt}}$ (e.g., Adam bf16 ≈ 16–20 B/param/step)
* Activation traffic coefficient $c_{\text{act}}$ (lower with FlashAttention/fused kernels)
* Recompute multiplier $\gamma$ (1.1–1.4 with activation checkpointing)

2. **Set hardware entries**
   Usable TFLOPs (bf16/fp16), HBM TB/s, NIC Gb/s, and your **\$/GPU-hr**.

3. **Ask the two key questions**

* What’s the **bottleneck**? (`compute`, `memory`, or `network`)
* Among configs that aren’t network-bound, which gives the lowest **\$/TB/s·hr** while meeting your tokens/sec target?

---

## Minimal code snippet (from the notebook)

```python
from dataclasses import dataclass
from math import ceil
from typing import Optional

@dataclass
class Hardware:
    name: str
    peak_flops_tflops: float
    hbm_tbps: float
    nic_gbps: float
    price_per_gpu_hr: float
    utilization: float = 0.75            # default per-resource efficiency
    compute_eff: Optional[float] = None  # override per resource; falls back to utilization
    memory_eff: Optional[float] = None
    network_eff: Optional[float] = None

@dataclass
class Model:
    n_params: float; layers: int; d_model: int; bytes_per_elem: int = 2

@dataclass
class TrainingCfg:
    k_flops_per_token: float = 6.0
    recompute_mult: float = 1.0
    alpha_opt_bytes_per_param: float = 16.0
    c_act: float = 6.0
    global_tokens_per_step: int = 512_000
    bytes_per_grad_elem: int = 2

# ...functions for per_token_flops, per_token_hbm_bytes, per_token_net_bytes...

def tokens_per_sec_per_gpu(hw, model, train, dp_world_size=1):
    # returns r_gpu, r_comp, r_mem, r_net, bound, intensity, machine_balance
    ...

def plan_cluster(hw, model, train, tokens_per_sec_target, dp_world_size=1):
    # returns per-GPU rate, GPUs needed, $/hr, cluster HBM TB/s, $/TB/s·hr
    ...
```

---

## Interpreting results

* **`bound == "memory"`** → You’re memory-bandwidth bound.

  * Reduce bytes/token: FlashAttention, fused kernels, 8-bit optimizers, bigger $B_g$ (if stable).
  * Prefer hardware with **better \$/TB/s·hr** (e.g., higher HBM BW per \$).

* **`bound == "network"`** → All-reduce is the choke point.

  * Increase $B_g$, reduce pure DP (add TP/PP/ZeRO), overlap comms, or raise effective NIC BW (EFA/IB).

* **`bound == "compute"`** → Great! Improve utilization and ensure you’re not secretly I/O-constrained.

---

## Examples to try

* Compare **H100 vs H200 vs L4** for a 70B model at target 200k tokens/sec.
* Flip to **inference** by setting $\kappa\approx2$, $\alpha_{\text{opt}}=0$, and modeling **KV-cache** bytes/token instead of activations.
* Test the effect of **global tokens/step** on the network bound (watch `r_net`).

---

## Roadmap

* [ ] Helper CLI: `python plan.py --model 70b --target-tps 2e5 --hw h100,h200`
* [ ] Plotting helpers (roofline view; \$/TB/s vs design points)
* [ ] Inference variant (KV cache), MoE variant (active params), long-context attention presets
* [ ] Optional YAML config for reproducible comparisons

---

## Contributing

PRs and issues welcome! Ideas:

* Add measured bandwidth/utilization from your cluster
* Additional hardware profiles and real **\$/TB/s·hr** snapshots
* Verified presets for FlashAttention, 8-bit optimizers, ZeRO, etc.

---

## Project Files

* [Sizing_AI_Training_by_Cost_per_Memory_Bandwidth.ipynb](./Sizing_AI_Training_by_Cost_per_Memory_Bandwidth.ipynb) — Main notebook with model and code.
* [The KV Cache: What It Is, Why It Matters, and How to Size It for Modern LLMs](./The_KV_Cache_What_It_Is,_Why_It_Matters,_and_How_to_Size_It_for_Modern_LLMs.ipynb) — Deep dive notebook on KV cache sizing and implications for LLM inference.

---

## References & further reading

* Roofline model (compute vs memory bound) — Williams et al., *CACM* (2009)
* FlashAttention (I/O-aware attention) — Dao et al., *arXiv:2205.14135*
* Megatron-LM scaling & comms patterns — Shoeybi et al., *arXiv:1909.08053*
* ZeRO optimizer sharding — Rajbhandari et al., *SC’20* / arXiv:1910.02054
* 8-bit optimizers — Dettmers et al., *arXiv:2110.02861*
* NCCL collectives, EFA/libfabric plugin — NVIDIA & AWS docs

*(See the blog post for a longer, linked bibliography.)*

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).

---

## Citation

If this helped your team ship or save money, feel free to cite the repo/blog post or drop a star ⭐.

```bibtex
@misc{cost_per_memory_bandwidth,
  title  = {Sizing AI Training by Cost per Memory Bandwidth},
  author = {Hodge, John},
  year   = {2025},
  url    = {https://github.com/jman4162/Sizing-AI-Training-by-Cost-per-Memory-Bandwidth}
}
```

---

[1]: https://github.com/jman4162/Sizing-AI-Training-by-Cost-per-Memory-Bandwidth/blob/main/Sizing_AI_Training_by_Cost_per_Memory_Bandwidth.ipynb "Sizing-AI-Training-by-Cost-per-Memory-Bandwidth/Sizing_AI_Training_by_Cost_per_Memory_Bandwidth.ipynb"
