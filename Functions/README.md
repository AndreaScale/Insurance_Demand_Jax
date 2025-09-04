# Functions: CARA + Lognormal (Truncated) Demand Lab (JAX)

A tiny, didactic package that simulates and plots demand for full insurance under **CARA utility** and **lognormal risk** with an **out-of-pocket cap** (to keep expected utility finite). Everything core runs in **JAX** (`jax.numpy`), while plotting uses `matplotlib`.

## Install deps

```bash
pip install jax jaxlib matplotlib
```

> On Apple Silicon: choose the right `jax/jaxlib` wheels per the official docs.

## Module map

- `demand.py`: parameters, Halton draws, uninsured utility, reservation price, demand.
- `simulation.py`: risk draws and simple dataset simulator (choice/price/sex/age).
- `graphs.py`: plotting helpers for a single demand curve and multiple curves across `alpha`.

All key functions are re-exported in `__init__.py` for convenience.

## Minimal example

```python
from Functions import default_params, simulate_dataset, plot_demand_curves_by_alpha
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1) Params + data
params = default_params()
data = simulate_dataset(N=3000, params=params, seed=42)

# 2) Compare curves across alphas (shades of gray)
alphas = jnp.array([5e-5, 1e-4, 2e-4, 4e-4])
ax = plot_demand_curves_by_alpha(params, alphas, data["sex"], data["age"])
plt.show()
```

## Teaching notes

- **Why the OOP cap?** CARA + *uncapped* lognormal implies `E[exp(alpha*L)] = +∞`; truncation mimics realistic out-of-pocket limits and makes the integral finite.
- **Why Halton?** It reduces Monte Carlo noise, producing smooth curves with relatively few draws.
- **Vectorization:** All numerics are `jax.numpy` and broadcasted; you can later `@jit` the heavy functions if you wish.