"""
Functions package: educational JAX lab for insurance demand under CARA utility.
Exposes key entry points for convenience.
"""

from .demand import (
    default_params,
    mu_linear,
    halton_1d,
    normal_from_uniform,
    avg_uninsured_utility,
    reservation_price,
    individual_demand,
    market_demand_at_price,
    demand_curve,
)

from .simulation import (
    risk_draws,
    simulate_dataset,
)

from .graphs import (
    plot_demand_curve,
    plot_demand_curves_by_alpha,
)

__all__ = [
    # demand/core
    "default_params", "mu_linear", "halton_1d", "normal_from_uniform",
    "avg_uninsured_utility", "reservation_price",
    "individual_demand", "market_demand_at_price", "demand_curve",
    # simulation
    "risk_draws", "simulate_dataset",
    # graphs
    "plot_demand_curve", "plot_demand_curves_by_alpha",
]