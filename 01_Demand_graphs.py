###################################################################################################
###################################################################################################

# GOAL: Plot Demand for Insurance
# INPUT: Simulated Dataset
# OUTPUT: Graphs in ...

###################################################################################################
###################################################################################################

# Demand for Full Insurance under CARA + (Truncated) Lognormal Risk
# ============================================
# This file shows how to:
#  1) Simulate a simple dataset (choice, price, sex, age)
#  2) Build a risk model: L ~ LogNormal(mu(sex, age), sigma=1)
#  3) Approximate uninsured expected utility via Monte Carlo using HALTON draws
#  4) Compute demand as an indicator: buy if p <= reservation price
#  5) Plot the demand curve q(p) and compare curves across risk aversion levels
#
# Libraries: JAX for arrays/math; Matplotlib for plotting only.
# Everything uses jax.numpy (jnp).

####################################################################################################################################
# Global Parameters and Imports
####################################################################################################################################

# Import necessary libraries
import os, jax, jax.numpy as jnp, matplotlib.pyplot as plt
from jax.scipy.special import logsumexp, erfinv

# Latex for figures
plt.rcParams.update({
    "text.usetex": True,    # Latex
    "font.family": "serif", # Font
    "figure.dpi": 150       # Quality images
    })

# Set JAX configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)  # Debug NaNs in JAX computations

# Import custom functions
from Functions import *

# Host-dependent paths
project_path  = os.environ["PROJECT_PATH"]
overleaf_path = os.environ["OVERLEAF_PATH"]
dropbox_path  = os.environ["DROPBOX_PATH"]

# Import simulation params
seed = int(os.environ["SEED"])
N = int(os.environ["N"])

################
# Demand Model #
################

def main():
    params = default_params()
    data = simulate_dataset(N=N, params=params, seed=seed) # Calling the parameters in signature

    # Single demand curve (black)
    plot_demand_curve(params, data["sex"], data["age"])
    plt.show()

    # Multiple alphas (shades of gray)
    alphas = jnp.array([5e-5, 1e-4, 2e-4, 4e-4])
    plot_demand_curves_by_alpha(params, alphas, data["sex"], data["age"])
    plt.show()

if __name__ == "__main__":
    main()