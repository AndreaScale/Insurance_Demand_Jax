# ============================================================
# demand.py
# ============================================================
# Core economics + utilities:
# - Parameters container (default_params)
# - Linear log-mean for lognormal losses mu(sex, age)
# - Halton sequence (1D) + inverse-CDF to Normal
# - Average uninsured utility via Monte Carlo (with OOP cap)
# - Reservation price p*
# - Individual & market demand
# - Demand curve on a grid
#
# Everything is written with jax.numpy (jnp) for vectorization.
# This is a didactic implementation: extremely commented and explicit.
# ============================================================

from typing import Dict, Any
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, erfinv


# --------------------------
# 0) Parameters container
# --------------------------
def default_params() -> Dict[str, Any]:
    """
    Build a small parameters dictionary with reasonable defaults.
    The mean of the lognormal depends on sex and age (linear). Variance is fixed
    at sigma=1 by design (per project requirement). We also include an out-of-pocket
    cap 'oop_max' to truncate losses so CARA expected utility remains finite.

    Returns
    -------
    params : dict
        {
          "alpha": float,  # CARA risk aversion (>0)
          "mu": {          # linear log-mean coefficients
              "intercept": float,
              "sex": float,
              "age": float,
              "age_center": float,
          },
          "sigma": float,      # fixed at 1.0
          "oop_max": float,    # truncation level for losses (in $)
        }
    """
    return {
        "alpha": 2e-4,    # try {5e-5, 1e-4, 2e-4, 4e-4} to see shifts
        "mu": {
            # Choose intercept so that E[L] ~ 1500 when sex=0, age ~ age_center, sigma=1
            "intercept": jnp.log(1500.0) - 0.5,
            # ~10% higher mean loss if sex==1
            "sex": jnp.log(1.10),
            # ~+25% per +10 years (log scale), centered at age_center
            "age": jnp.log(1.25) / 10.0,
            "age_center": 40.0,
        },
        "sigma": 1.0,
        "oop_max": 10000.0,
    }


# --------------------------
# 1) Linear mu(sex, age)
# --------------------------
def mu_linear(sex: jnp.ndarray, age: jnp.ndarray, mu_params: Dict[str, Any]) -> jnp.ndarray:
    """
    Compute log-mean 'mu' of the lognormal L ~ LogNormal(mu, sigma^2)
    as a linear function of (sex, age).

    Parameters
    ----------
    sex : array of int in {0,1}
    age : array of int/float (years)
    mu_params : dict with keys {"intercept", "sex", "age", "age_center"}

    Returns
    -------
    mu : array, same shape as inputs
    """
    age_centered = (age - mu_params["age_center"])
    return (
        mu_params["intercept"]
        + mu_params["sex"] * sex
        + mu_params["age"] * age_centered
    )


# --------------------------
# 2) Halton sequence (1D)
# --------------------------
def halton_1d(n: int, base: int = 2, start_index: int = 1) -> jnp.ndarray:
    """
    Generate n points from the 1D Halton sequence in (0,1).

    Notes
    -----
    - The Halton sequence is a low-discrepancy (quasi-random) sequence.
      It often reduces Monte Carlo noise compared to pseudo-random draws.
    - We skip index 0 by default (start_index=1) following common practice.
    - This implementation uses small Python control flow to figure out
      how many base-`base` digits are needed. It's perfectly fine for teaching.
      If you later want to JIT this, precompute the digit count externally.

    Returns
    -------
    u : jnp.ndarray, shape (n,), values strictly in (0,1)
    """
    idx = jnp.arange(start_index, start_index + n, dtype=jnp.int32)
    max_idx = int(start_index + n - 1)

    # Determine number of base digits to cover the largest index
    K = 1
    bpow = base
    while bpow <= max_idx:
        bpow *= base
        K += 1

    pow_base = jnp.array([base**k for k in range(K)], dtype=jnp.int64)        # (K,)
    frac_wts = jnp.array([base**(-(k+1)) for k in range(K)], dtype=jnp.float32)  # (K,)

    # Extract base-`base` digits for each index
    digits = (idx[:, None] // pow_base[None, :]) % base  # (n, K)

    u = (digits * frac_wts[None, :]).sum(axis=1)
    eps = 1e-12
    return jnp.clip(u, eps, 1.0 - eps)


# --------------------------
# 3) Uniform -> Normal via inverse CDF
# --------------------------
def normal_from_uniform(u: jnp.ndarray) -> jnp.ndarray:
    """
    Map U(0,1) to standard Normal N(0,1) using the exact inverse-CDF:
        Z = Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1).
    """
    return jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)


# --------------------------
# 4) Average uninsured utility (Monte Carlo + OOP cap)
# --------------------------
def avg_uninsured_utility(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    n_draws: int = 2048,
    halton_base: int = 2,
    halton_start: int = 1,
) -> jnp.ndarray:
    """
    Approximate E[u_uninsured] with CARA utility and truncated lognormal losses.

    Model
    -----
    - CARA utility: u(c) = -(1/alpha) * exp(-alpha c), alpha>0.
    - Uninsured consumption (wealth normalized to 0): c = -min(L, M),
      where L ~ LogNormal(mu(sex, age), sigma^2), M=oop_max (truncation).
    - Therefore u_uninsured = -(1/alpha) * exp(alpha * min(L, M)).

    Implementation
    --------------
    - Draw Z_s ~ N(0,1) via Halton in U then inverse CDF.
    - Construct L_i^s = exp(mu_i + sigma * Z_s), then cap at M.
    - Compute log-mean-exp for numerical stability:
        log E[exp(alpha * min(L,M))] ≈ logsumexp(alpha * L_cap) - log(S).
    - Finally multiply by -(1/alpha) and exponentiate.

    Returns
    -------
    avg_u0 : jnp.ndarray of shape (N,), per-person expected uninsured utility.
    """
    alpha = params["alpha"]
    mu_params = params["mu"]
    sigma = params["sigma"]    # fixed at 1.0 in this project
    M = params["oop_max"]

    # Person-specific log-means
    mu = mu_linear(sex, age, mu_params)  # (N,)

    # Halton -> Normal draws shared across all individuals
    u = halton_1d(n_draws, base=halton_base, start_index=halton_start)  # (S,)
    z = normal_from_uniform(u)                                          # (S,)

    # Broadcast to (N, S)
    L = jnp.exp(mu[:, None] + sigma * z[None, :])
    L_cap = jnp.minimum(L, M)

    # log E[exp(alpha * L_cap)] per person
    log_mean_exp = logsumexp(alpha * L_cap, axis=1) - jnp.log(n_draws)

    # Average uninsured utility
    avg_u0 = -(1.0 / alpha) * jnp.exp(log_mean_exp)  # (N,)
    return avg_u0


# --------------------------
# 5) Reservation price (certainty equivalent)
# --------------------------
def reservation_price(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    n_draws: int = 2048,
    halton_base: int = 2,
    halton_start: int = 1,
) -> jnp.ndarray:
    """
    Compute reservation price p*_i for each person:
        p*_i = (1/alpha) * log E[exp(alpha * min(L_i, M))].

    This follows directly from comparing insured vs uninsured expected utility
    under CARA with full insurance and price p.
    """
    alpha = params["alpha"]
    mu_params = params["mu"]
    sigma = params["sigma"]
    M = params["oop_max"]

    mu = mu_linear(sex, age, mu_params)  # (N,)

    u = halton_1d(n_draws, base=halton_base, start_index=halton_start)  # (S,)
    z = normal_from_uniform(u)                                          # (S,)

    L = jnp.exp(mu[:, None] + sigma * z[None, :])   # (N, S)
    L_cap = jnp.minimum(L, M)

    # log E[exp(alpha * L_cap)]
    log_mean_exp = logsumexp(alpha * L_cap, axis=1) - jnp.log(n_draws)

    # p* = (1/alpha) * log E[...]
    p_star = (1.0 / alpha) * log_mean_exp
    return p_star


# --------------------------
# 6) Individual & Market demand at a price
# --------------------------
def individual_demand(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    price: jnp.ndarray,
    n_draws: int = 2048,
    halton_base: int = 2,
    halton_start: int = 1,
) -> jnp.ndarray:
    """
    Individual demand indicator: 1 if price <= p*_i, else 0.
    `price` can be a scalar or array broadcastable to (N,).
    """
    p_star = reservation_price(params, sex, age,
                               n_draws=n_draws,
                               halton_base=halton_base,
                               halton_start=halton_start)  # (N,)

    price = jnp.asarray(price)
    if price.ndim == 0:
        price = jnp.full_like(p_star, fill_value=price.item())

    return (price <= p_star).astype(jnp.int32)


def market_demand_at_price(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    price: float,
    n_draws: int = 2048,
    halton_base: int = 2,
    halton_start: int = 1,
) -> jnp.ndarray:
    """
    Fraction of individuals who buy at price p (scalar): mean_i 1{p <= p*_i}.
    Returns a scalar in [0,1].
    """
    d = individual_demand(params, sex, age, price,
                          n_draws=n_draws, halton_base=halton_base, halton_start=halton_start)
    return d.mean()


# --------------------------
# 7) Demand curve on a price grid
# --------------------------
def demand_curve(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    p_grid: jnp.ndarray,
    n_draws: int = 4096,
    halton_base: int = 2,
    halton_start: int = 1,
) -> jnp.ndarray:
    """
    Compute q(p) on p_grid efficiently by computing p*_i once.
    q(p) = mean_i 1{p <= p*_i}.
    """
    p_star = reservation_price(params, sex, age,
                               n_draws=n_draws, halton_base=halton_base, halton_start=halton_start)  # (N,)
    p_grid = jnp.asarray(p_grid)  # (P,)

    buys = (p_grid[:, None] <= p_star[None, :]).astype(jnp.float32)  # (P, N)
    q = buys.mean(axis=1)  # (P,)
    return q