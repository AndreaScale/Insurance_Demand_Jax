# ============================================================
# simulation.py
# ============================================================
# Dataset and risk draws utilities:
# - risk_draws: sample (truncated) lognormal losses per person
# - simulate_dataset: produce (choice, price, sex, age)
#
# Uses functions from demand.py (mu_linear, halton_1d, normal_from_uniform,
# reservation_price, individual_demand). Kept in a separate file to illustrate
# modular design.
# ============================================================

from typing import Dict, Any
import jax
import jax.numpy as jnp

from .demand import (
    mu_linear, halton_1d, normal_from_uniform,
    reservation_price, individual_demand,
)


def risk_draws(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    n_draws: int = 1000,
    halton_base: int = 2,
    halton_start: int = 1,
) -> jnp.ndarray:
    """
    Sample n_draws losses per person from L = exp(mu + sigma*Z), Z~N(0,1),
    then truncate at oop_max for consistency with the model.

    Returns
    -------
    L_cap : jnp.ndarray of shape (N, n_draws)
    """
    mu = mu_linear(sex, age, params["mu"])  # (N,)
    sigma = params["sigma"]
    M = params["oop_max"]

    # Halton -> Normal draws
    u = halton_1d(n_draws, base=halton_base, start_index=halton_start)  # (S,)
    z = normal_from_uniform(u)                                          # (S,)

    L = jnp.exp(mu[:, None] + sigma * z[None, :])  # (N, S)
    return jnp.minimum(L, M)


def simulate_dataset(
    N: int,
    params: Dict[str, Any],
    p_min: float = None,
    p_max: float = None,
    seed: int = 0,
    n_draws: int = 4096,
    halton_base: int = 2,
    halton_start: int = 1,
) -> Dict[str, jnp.ndarray]:
    """
    Simulate a dataset with fields:
        - choice (0/1), price (float), sex (0/1), age (int)

    Design
    ------
    - sex ~ Bernoulli(0.5)
    - age ~ Uniform{20,...,64}
    - price ~ Uniform[p_min, p_max], independent of (sex, age)
      If p_min/p_max missing, derive p_max from ~99th percentile of p* (reference).

    Choice is deterministic in this educational model: choice = 1{price <= p*}.
    """
    key = jax.random.PRNGKey(seed)

    # Draw covariates
    key, k1, k2 = jax.random.split(key, 3)
    sex = jax.random.bernoulli(k1, p=0.5, shape=(N,)).astype(jnp.int32)
    age = jax.random.randint(k2, shape=(N,), minval=20, maxval=65).astype(jnp.int32)

    # Set price range from reservation prices if not provided
    if (p_min is None) or (p_max is None):
        p_star_ref = reservation_price(params, sex, age,
                                       n_draws=n_draws,
                                       halton_base=halton_base,
                                       halton_start=halton_start)
        if p_min is None:
            p_min = 0.0
        if p_max is None:
            p_max = jnp.quantile(p_star_ref, 0.99)

    key, k3 = jax.random.split(key)
    price = jax.random.uniform(k3, shape=(N,), minval=p_min, maxval=p_max)

    # Deterministic choice from the model
    choice = individual_demand(params, sex, age, price,
                               n_draws=n_draws,
                               halton_base=halton_base,
                               halton_start=halton_start)

    return {
        "choice": choice.astype(jnp.int32),
        "price": price,
        "sex": sex,
        "age": age,
    }