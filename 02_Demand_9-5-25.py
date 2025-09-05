# Improt necessary libraries
import jax

# Import custom functions
from Functions import *

# -------------------------------
# Example usage with reasonable parameters
# -------------------------------
    
# Population & risk
N = 5000
master_key = jax.random.PRNGKey(0)
key_draws, key_risk = jax.random.split(master_key)
risk = jax.random.uniform(key_risk, shape=(N,), minval=0.0, maxval=1.0)

# Lognormal params (σ = 1 fixed, μ_i = mu_0 + mu * risk_i)
params = {"mu_0": 6.5, "mu": 1.0}   # mean losses roughly in the low-$thousands
K = 50_000.0                        # cap to avoid extreme tail explosions
S = 2000                            # draws per person (increase for smoother curve)

# CARA risk aversion
alpha = 0.0002

# Price range
price_min, price_max = 0.0, 8_000.0
_ = plot_demand_vs_price(key_draws, risk, S, params, K, alpha,
                         price_min=price_min, price_max=price_max, num_points=80)
