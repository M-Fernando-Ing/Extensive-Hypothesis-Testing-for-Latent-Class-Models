# SearchLibrium — Latent Class Specification Search

Reproducibility code for the manuscript:

> **Capturing behavioural heterogeneity through extensive hypothesis testing for latent class choice models**
> Submitted to *Transportation Research Part C: Emerging Technologies*, Special Issue in Memory of Professor Hani S. Mahmassani.

This repository is a reduced version of [SearchLibrium](https://pypi.org/project/SearchLibrium/) (Ahern et al., 2026) containing only the modules required to reproduce the results reported in the manuscript. It is intended for peer review and reproducibility, not as a general-purpose release. For the full package and its documentation, see the upstream project.

---

## Requirements

- Python ≥ 3.10
- Linux, Windows or macOS

GPU acceleration is available on **Linux with an NVIDIA GPU** only (including WSL2). On Windows and macOS the package runs on CPU.

| Platform | JAX backend installed |
|---|---|
| Linux (incl. HPC, WSL2) | `jax[cuda12]` — uses the GPU when present, otherwise falls back to CPU |
| Windows, macOS | `jax` (CPU) |

All dependencies are declared in `pyproject.toml` and installed automatically.

---

## Installation

```bash
git clone https://github.com/M-Fernando-Ing/Extensive-Hypothesis-Testing-for-Latent-Class-Models.git
cd Extensive-Hypothesis-Testing-for-Latent-Class-Models
bash install.sh
```

Run `bash install.sh --help` for options (custom environment name, venv instead of conda, HPC temp directory).

### Notes

- **First installation on Linux** downloads approximately 3–4 GB of NVIDIA CUDA libraries. On network file systems (e.g. HPC scratch) this step may take 5–10 minutes.
- **HPC users:** installation can be performed on a login node. At runtime, JAX detects the GPU on the compute node automatically.
- **`PYTHONNOUSERSITE=1`** prevents Python from loading packages from `~/.local`, which may otherwise shadow the environment's versions (e.g. an older `SearchLibrium` or an incompatible `jaxlib`).
- **Existing installations:** if the PyPI release of `SearchLibrium` is installed in the same environment, uninstall it first (`pip uninstall SearchLibrium`), since both share the same import name.

### Verify

```bash
python -c "import SearchLibrium, jax; print(SearchLibrium.__file__); print(jax.devices())"
```

The path must point to `src/SearchLibrium/` in this repository. `jax.devices()` returns `CudaDevice` when a GPU is in use and `CpuDevice` otherwise.

---

## Main modules

| Module | Role |
|---|---|
| `latent_class_final.py` | Latent class MNL estimator with multistart and per-chain degeneracy flags |
| `separation_screen.py` | Quasi-perfect separation diagnostics for latent class models |
| `search.py` | Search framework: solution representation and perturbation operators |
| `harmony.py` | Harmony Search |
| `siman.py` | Simulated Annealing |
| `call_meta.py` | Entry point for metaheuristic runs |
| `_choice_model.py` | Base discrete choice model class |

---

## Data

The public datasets used in the manuscript are included in this repository, except MasS which is private.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```text
[Manuscript citation]

Ahern, Z., Taco Morales, M.F., Paz, A., Beeramole, P., & Burdett, R. (2026).
SearchLibrium: Automated discrete choice model search.
https://pypi.org/project/SearchLibrium/
```
