# ============================================================
# graphs.py
# ============================================================
# Plotting helpers:
# - plot_demand_curve
# - plot_demand_curves_by_alpha
#
# Uses matplotlib for visuals (kept separate from JAX math).
# ============================================================

from typing import Dict, Any, Sequence, Optional
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .demand import reservation_price, demand_curve


def plot_demand_curve(
    params: Dict[str, Any],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    p_min: float = None,
    p_max: float = None,
    n_points: int = 100,
    n_draws: int = 4096,
    halton_base: int = 2,
    halton_start: int = 1,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot q(p) vs p as a black line.
    If p_min/p_max are not given, derive p_max from the 99th percentile of p*.
    """
    p_star = reservation_price(params, sex, age,
                               n_draws=n_draws,
                               halton_base=halton_base,
                               halton_start=halton_start)
    if p_min is None:
        p_min = 0.0
    if p_max is None:
        p_max = jnp.quantile(p_star, 0.99)

    p_grid = jnp.linspace(p_min, p_max, n_points)
    q_grid = demand_curve(params, sex, age, p_grid,
                          n_draws=n_draws,
                          halton_base=halton_base,
                          halton_start=halton_start)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    # Black line per requirement
    ax.plot(p_grid, q_grid, linewidth=2.0, color="black")
    ax.set_xlabel("Price p")
    ax.set_ylabel("Quantity q (fraction insured)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Demand for Full Insurance")
    return ax


def plot_demand_curves_by_alpha(
    base_params: Dict[str, Any],
    alphas: Sequence[float],
    sex: jnp.ndarray,
    age: jnp.ndarray,
    p_min: float = None,
    p_max: float = None,
    n_points: int = 100,
    n_draws: int = 4096,
    halton_base: int = 2,
    halton_start: int = 1,
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
):
    """
    For each alpha in `alphas`, plot q(p) using a different gray shade.
    Darker = higher alpha.
    """
    alphas = jnp.asarray(alphas)

    # Choose x-axis from a reference alpha if not provided
    if (p_min is None) or (p_max is None):
        ref_params = dict(base_params)
        ref_params["alpha"] = float(jnp.median(alphas))
        p_star_ref = reservation_price(ref_params, sex, age,
                                       n_draws=n_draws,
                                       halton_base=halton_base,
                                       halton_start=halton_start)
        if p_min is None:
            p_min = 0.0
        if p_max is None:
            p_max = jnp.quantile(p_star_ref, 0.99)

    p_grid = jnp.linspace(p_min, p_max, n_points)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    # Shades from light to dark, then assign by ordering of alphas
    shades = jnp.linspace(0.8, 0.0, len(alphas))
    order = jnp.argsort(alphas)  # plot in increasing alpha
    for rank, idx in enumerate(order):
        a = float(alphas[int(idx)])
        params = dict(base_params)
        params["alpha"] = a
        q_grid = demand_curve(params, sex, age, p_grid,
                              n_draws=n_draws,
                              halton_base=halton_base,
                              halton_start=halton_start)
        # Matplotlib grayscale as string: "0.0" (black) ... "1.0" (white)
        ax.plot(p_grid, q_grid, linewidth=2.0, color=str(shades[rank]),
                label=fr"$\alpha={a:.1e}$")

    ax.set_xlabel("Price p")
    ax.set_ylabel("Quantity q (fraction insured)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Demand Curves Across Risk Aversion (CARA)")
    if show_legend:
        ax.legend(frameon=False, loc="upper right")
    return ax