from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(2,))  # S è static per efficienza/JIT
def draw_capped_lognormal(key, risk, S, params, K):
    """
    key:    jax.random.PRNGKey
    risk:   array shape (N,) o (N,1) con i r_i
    S:      int, numero di draw per individuo
    params: dict con chiavi {'mu_0', 'mu'} per μ = μ_0 + μ * r
    K:      float o array broadcastable (cap superiore)

    Ritorna: array (N, S) con draw lognormali cappati a K.
    """
    risk = jnp.asarray(risk, dtype=jnp.float32).reshape(-1, 1)  # (N,1)
    mu = jnp.asarray(params["mu_0"]) + jnp.asarray(params["mu"]) * risk  # (N,1)
    # sigma = 1 fisso
    z = jax.random.normal(key, shape=(risk.shape[0], S))        # (N,S)
    draws = jnp.exp(mu + z)                                     # (N,S)
    return jnp.minimum(draws, jnp.asarray(K, dtype=draws.dtype))  # cap a K

from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def _log_mean_exp_alphaX(draws, alpha):
    """
    Numerically stable log( mean_s exp(alpha * X_{i,s}) ) per i.
    draws: (N, S)
    alpha: scalar
    returns: (N,)
    """
    draws = jnp.asarray(draws)
    S = draws.shape[1]
    return logsumexp(alpha * draws, axis=1) - jnp.log(S)

@jax.jit
def expected_utility_uninsured(draws, alpha):
    """
    Expected utility for being uninsured under CARA, no initial wealth.
    draws: (N, S) nonnegative expenditures X_{i,s}
    alpha: scalar (>0) risk aversion
    returns: (N,)  EU_i = - E_s[ exp(alpha * X_{i,s}) ]
    """
    log_mean_exp = _log_mean_exp_alphaX(draws, alpha)  # (N,)
    return -jnp.exp(log_mean_exp)

@jax.jit
def certainty_equivalent_uninsured(draws, alpha):
    """
    CARA certainty-equivalent loss implied by those draws.
    draws: (N, S)
    alpha: scalar
    returns: (N,)  CE_i = (1/alpha) * log E_s[ exp(alpha * X_{i,s}) ]
    Handles alpha ~ 0 by falling back to mean(X).
    """
    eps = 1e-12
    log_mean_exp = _log_mean_exp_alphaX(draws, alpha)  # (N,)
    ce_when_alpha = (1.0 / alpha) * log_mean_exp
    ce_when_zero  = jnp.mean(draws, axis=1)
    return jnp.where(jnp.abs(alpha) < eps, ce_when_zero, ce_when_alpha)

# Optional wrappers that keep the earlier signature you used:
@partial(jax.jit, static_argnums=(2,))
def uninsured_EU_with_signature(key, risk, S, params, K, draws, alpha):
    return expected_utility_uninsured(draws, alpha)

@partial(jax.jit, static_argnums=(2,))
def uninsured_CE_with_signature(key, risk, S, params, K, draws, alpha):
    return certainty_equivalent_uninsured(draws, alpha)

from functools import partial
import jax
import jax.numpy as jnp

# Assumes these are already defined from earlier messages:
# - draw_capped_lognormal(key, risk, S, params, K)
# - expected_utility_uninsured(draws, alpha)
# - certainty_equivalent_uninsured(draws, alpha)

@partial(jax.jit, static_argnums=(2,))
def insurance_demand_full_coverage(key, risk, S, params, K, price, alpha):
    """
    Fraction of individuals who buy full insurance under CARA utility.

    key:    jax.random.PRNGKey
    risk:   (N,) or (N,1) array of individual risk indexes (used to set μ_i)
    S:      int, number of expenditure draws per individual (static for JIT)
    params: dict with {'mu_0','mu'} defining μ_i = mu_0 + mu * risk_i
    K:      float or broadcastable cap for expenditure draws
    price:  scalar or (N,) premium for full insurance
    alpha:  scalar CARA risk-aversion coefficient

    Returns:
        demand: scalar in [0,1], the fraction of individuals who buy insurance.
    """
    # Normalize shapes
    risk = jnp.asarray(risk).reshape(-1)
    N = risk.shape[0]
    price = jnp.asarray(price)
    priceN = jnp.where(price.ndim == 0, jnp.full((N,), price), price.reshape(N))

    # 1) Simulate uninsured expenditure draws X_{i,s}
    draws = draw_capped_lognormal(key, risk, S, params, K)  # (N,S)

    # 2) Uninsured expected utility: EU_un = - E_s[ exp(alpha * X_{i,s}) ]
    EU_un = expected_utility_uninsured(draws, alpha)        # (N,)

    # 3) Insured expected utility with full coverage: certain loss = price_i
    #    EU_in = -exp(alpha * price_i)
    EU_in = -jnp.exp(alpha * priceN)                        # (N,)

    # 4) Decision rule:
    #    - For alpha ~= 0, EU_un ≈ EU_in ≈ -1, so compare certainty equivalents:
    #         buy if  P_i <= CE_un
    #    - Otherwise, buy if EU_in >= EU_un
    eps = 1e-12
    CE_un = certainty_equivalent_uninsured(draws, alpha)    # (N,)
    buy_mask = jnp.where(jnp.abs(alpha) < eps,
                         priceN <= CE_un,
                         EU_in >= EU_un)

    demand = jnp.mean(buy_mask.astype(jnp.float32))
    return demand

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_demand_vs_price(key, risk, S, params, K, alpha,
                         price_min=0.0, price_max=8000.0, num_points=60,
                         ax=None):
    """
    Plot demand (fraction buying) vs price under CARA and full coverage.

    key:        jax.random.PRNGKey
    risk:       (N,) or (N,1) risk array (used for μ_i construction)
    S:          int, number of expenditure draws per individual
    params:     dict {'mu_0','mu'} for μ_i = mu_0 + mu * risk_i, σ = 1
    K:          float (cap) or broadcastable
    alpha:      scalar CARA coefficient
    price_min, price_max: price range
    num_points: number of prices in the grid
    ax:         optional matplotlib axis to plot on

    Returns:
        prices: (P,) JAX array
        demand: (P,) JAX array of demand in [0,1]
    """
    # 1) Simulate uninsured expenditure draws once (shared across all prices)
    risk = jnp.asarray(risk).reshape(-1)
    N = risk.shape[0]
    draws = draw_capped_lognormal(key, risk, S, params, K)  # (N, S)

    # 2) Precompute uninsured expected utility and CE (only depend on draws, alpha)
    EU_un = expected_utility_uninsured(draws, alpha)         # (N,)
    CE_un = certainty_equivalent_uninsured(draws, alpha)     # (N,)

    # 3) Prices grid
    prices = jnp.linspace(price_min, price_max, num_points)  # (P,)

    # 4) Demand at each price (vectorized over price)
    eps = 1e-12
    if jnp.abs(alpha) < eps:
        # α ≈ 0 => buy if price <= CE_un
        buy_mask = prices[None, :] <= CE_un[:, None]         # (N,P)
    else:
        # General case => buy if EU_insured >= EU_uninsured
        EU_in = -jnp.exp(alpha * prices)[None, :]            # (1,P) -> (N,P) by broadcast
        buy_mask = EU_in >= EU_un[:, None]                   # (N,P)

    demand = jnp.mean(buy_mask.astype(jnp.float32), axis=0)  # (P,)

    # 5) Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(prices, demand, lw=2)
    ax.set_xlabel("Price")
    ax.set_ylabel("Demand (fraction buying)")
    ax.set_ylim(0, 1)
    ax.set_title("Full-coverage Insurance Demand (CARA)")
    ax.grid(True, linestyle="--", alpha=0.5)
    if ax is None:
        plt.show()

    return prices, demand

