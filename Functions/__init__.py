"""
functions
=========

Convenience re-exports for the insurance-demand toy model.

Modules expected in this package:
- simulation.py  -> draw_capped_lognormal
- utility.py     -> expected_utility_uninsured, certainty_equivalent_uninsured,
                    uninsured_EU_with_signature, uninsured_CE_with_signature
- demand.py      -> insurance_demand_full_coverage
- plots.py       -> plot_demand_vs_price
"""

__version__ = "0.1.0"
__docformat__ = "google"

from .Functions_Nicole import (
    expected_utility_uninsured,
    certainty_equivalent_uninsured,
    uninsured_EU_with_signature,
    uninsured_CE_with_signature,
    draw_capped_lognormal,
    insurance_demand_full_coverage,
    plot_demand_vs_price
)

__all__ = [
    "draw_capped_lognormal",
    "expected_utility_uninsured",
    "certainty_equivalent_uninsured",
    "uninsured_EU_with_signature",
    "uninsured_CE_with_signature",
    "insurance_demand_full_coverage",
    "plot_demand_vs_price",
]

'''
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
'''