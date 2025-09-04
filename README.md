

# Insurance Demand Model with `JAX`

A tiny exaplme that simulates and plots demand for full insurance under **CARA utility** and **lognormal risk** with an **out-of-pocket cap** (to keep expected utility finite). Everything core runs in **JAX** (`jax.numpy`), while plotting uses `matplotlib`.

## Model description

We study demand for full insurance under **CARA utility** and **lognormal health expenditure risk**:

- Each individual faces medical loss

$$Exp \sim \text{LogNormal}(\mu(\text{sex}, \text{age}), \sigma^2),$$

where $\mu(\text{sex}, \text{age}) = \mu_0 + \mu_a * age + \mu_s * sex$.

- Utility is **CARA**:

$$u(c) = -\tfrac{1}{\alpha} \exp(-\alpha c)$$

with risk aversion $\alpha>0$.

- Full insurance costs premium \(p\) and covers all expenditure (truncated at an out-of-pocket maximum \(M\)).
- Without insurance: $c = -\min(Exp,M)$; with insurance: $c = -p$.
- Insurance is purchased iff the premium is below the **certainty equivalent** of risk, that is

$$p \leq p^{\ast} = \frac{1}{\alpha} log(E(exp(\alpha min(L,M)))),$$

This yields individual demand $D_i(p)=1[p \leq p^{\ast}_i]$, and aggregate demand is

$$Q = \int D_i(p) di.$$

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
