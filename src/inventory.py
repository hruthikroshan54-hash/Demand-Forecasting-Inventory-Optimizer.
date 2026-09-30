"""Classic inventory policy maths driven by the ML forecast."""
from __future__ import annotations

import math

from scipy.stats import norm


def safety_stock(sigma_daily: float, lead_time_days: float, service_level: float) -> float:
    """SS = z * sigma_daily * sqrt(lead time). service_level in (0, 1)."""
    z = norm.ppf(service_level)
    return max(0.0, z * sigma_daily * math.sqrt(lead_time_days))


def reorder_point(mean_daily: float, lead_time_days: float, ss: float) -> float:
    """ROP = demand during lead time + safety stock."""
    return mean_daily * lead_time_days + ss


def eoq(annual_demand: float, order_cost: float, unit_cost: float, holding_rate: float) -> float:
    """Economic Order Quantity = sqrt(2 D S / H), with H = unit_cost * holding_rate."""
    h = unit_cost * holding_rate
    if annual_demand <= 0 or order_cost <= 0 or h <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * order_cost / h)


def inventory_plan(mean_daily: float, sigma_daily: float, lead_time_days: float, service_level: float,
                   order_cost: float, unit_cost: float, holding_rate: float) -> dict:
    ss = safety_stock(sigma_daily, lead_time_days, service_level)
    rop = reorder_point(mean_daily, lead_time_days, ss)
    annual = mean_daily * 365
    q = eoq(annual, order_cost, unit_cost, holding_rate)
    h = unit_cost * holding_rate
    orders_per_year = annual / q if q else 0.0
    return {
        "safety_stock": ss,
        "reorder_point": rop,
        "eoq": q,
        "annual_demand": annual,
        "orders_per_year": orders_per_year,
        "cycle_days": 365 / orders_per_year if orders_per_year else 0.0,
        "annual_ordering_cost": orders_per_year * order_cost,
        "annual_holding_cost": (q / 2 + ss) * h,
    }
