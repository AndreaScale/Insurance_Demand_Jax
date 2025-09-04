# Functions: CARA + Lognormal (Truncated) Demand Lab (JAX)

A tiny exaplme that simulates and plots demand for full insurance under **CARA utility** and **lognormal risk** with an **out-of-pocket cap** (to keep expected utility finite). Everything core runs in **JAX** (`jax.numpy`), while plotting uses `matplotlib`.

## Install deps

```bash
pip install jax jaxlib matplotlib
```

## Module map

- `demand.py`: parameters, Halton draws, uninsured utility, reservation price, demand.
- `simulation.py`: risk draws and simple dataset simulator (choice/price/sex/age).
- `graphs.py`: plotting helpers for a single demand curve and multiple curves across `alpha`.

All key functions are re-exported in `__init__.py` for convenience.


## Notes

- **Why the OOP cap?** CARA + *uncapped* lognormal implies `E[exp(alpha*L)] = +∞`; truncation mimics realistic out-of-pocket limits and makes the integral finite.
- **Why Halton?** It reduces Monte Carlo noise, producing smooth curves with relatively few draws.
- **Vectorization:** All numerics are `jax.numpy` and broadcasted; you can later `@jit` the heavy functions if you wish. 