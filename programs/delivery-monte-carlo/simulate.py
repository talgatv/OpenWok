#!/usr/bin/env python3
"""Monte Carlo simulator for delivery business profitability."""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


MetricMap = Dict[str, float]
CityMetricsMap = Dict[str, MetricMap]


METRIC_KEYS = [
    "attempted_orders",
    "successful_payments",
    "cancelled_orders",
    "delivered_orders",
    "protected_orders",
    "refund_orders",
    "dispute_orders",
    "late_orders",
    "fraud_incidents",
    "manual_reviews",
    "support_tickets",
    "gmv_delivered_usd",
    "gmv_processed_usd",
    "platform_base_revenue_usd",
    "protection_revenue_usd",
    "total_revenue_usd",
    "processor_fees_usd",
    "refund_loss_usd",
    "dispute_loss_gross_usd",
    "insurance_coverage_usd",
    "dispute_loss_net_usd",
    "infra_cost_usd",
    "support_cost_usd",
    "review_cost_usd",
    "insurance_premium_cost_usd",
    "operating_costs_total_usd",
    "total_losses_usd",
    "net_profit_usd",
    "reserve_hold_usd",
    "cash_after_reserve_usd",
]


CITY_INDEX_KEYS = [
    "density_index",
    "crime_index",
    "income_index",
    "economic_stress_index",
    "tourism_index",
    "traffic_congestion_index",
    "courier_reliability_index",
    "digital_fraud_pressure_index",
    "anti_fraud_maturity_index",
    "hotspot_inequality_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation for delivery business economics."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON configuration file (default: config.json)",
    )
    parser.add_argument(
        "--save-report",
        default="",
        help="Legacy option: optional path to save JSON report.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for full reporting bundle (JSON/MD/CSV). Empty value disables bundle output.",
    )
    parser.add_argument(
        "--report-name",
        default="simulation_report",
        help="Base filename for report bundle (default: simulation_report).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config: {path}: {exc}") from exc


def get_required(config: Dict[str, Any], *keys: str) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            joined = ".".join(keys)
            raise ValueError(f"Missing config key: {joined}")
        cursor = cursor[key]
    return cursor


def validate_range(name: str, value: float, min_value: float, max_value: float) -> None:
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be in [{min_value}, {max_value}], got {value}")


def clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def evaluate_risk_band(score: float) -> str:
    if score >= 0.67:
        return "HIGH"
    if score >= 0.34:
        return "MEDIUM"
    return "LOW"


def compute_city_multipliers(city: Dict[str, float]) -> Dict[str, float]:
    population_factor = clamp(
        safe_div(math.log1p(city["population_millions"]), math.log1p(15.0)),
        0.15,
        1.35,
    )

    demand_multiplier = clamp(
        0.55
        + 0.35 * population_factor
        + 0.20 * city["density_index"]
        + 0.20 * city["income_index"]
        + 0.15 * city["tourism_index"]
        - 0.10 * city["economic_stress_index"],
        0.45,
        2.20,
    )

    order_value_multiplier = clamp(
        0.60
        + 0.55 * city["income_index"]
        + 0.10 * city["density_index"]
        + 0.20 * city["tourism_index"]
        - 0.10 * city["economic_stress_index"],
        0.45,
        2.50,
    )

    payment_success_multiplier = clamp(
        0.90
        + 0.12 * city["income_index"]
        - 0.18 * city["economic_stress_index"]
        - 0.10 * city["digital_fraud_pressure_index"]
        + 0.06 * city["anti_fraud_maturity_index"],
        0.65,
        1.10,
    )

    cancel_rate_multiplier = clamp(
        0.80
        + 0.30 * city["traffic_congestion_index"]
        + 0.20 * city["economic_stress_index"]
        + 0.15 * city["hotspot_inequality_index"]
        - 0.15 * city["courier_reliability_index"],
        0.55,
        1.90,
    )

    refund_rate_multiplier = clamp(
        0.75
        + 0.25 * city["traffic_congestion_index"]
        + 0.20 * city["economic_stress_index"]
        + 0.20 * city["crime_index"]
        + 0.20 * city["hotspot_inequality_index"]
        - 0.20 * city["courier_reliability_index"],
        0.45,
        2.20,
    )

    dispute_rate_multiplier = clamp(
        0.65
        + 0.45 * city["crime_index"]
        + 0.30 * city["economic_stress_index"]
        + 0.30 * city["digital_fraud_pressure_index"]
        + 0.20 * city["hotspot_inequality_index"]
        - 0.35 * city["anti_fraud_maturity_index"],
        0.35,
        2.70,
    )

    fraud_incident_multiplier = clamp(
        0.55
        + 0.60 * city["crime_index"]
        + 0.40 * city["economic_stress_index"]
        + 0.45 * city["digital_fraud_pressure_index"]
        + 0.20 * city["hotspot_inequality_index"]
        - 0.45 * city["anti_fraud_maturity_index"],
        0.30,
        3.00,
    )

    late_delivery_multiplier = clamp(
        0.65
        + 0.55 * city["traffic_congestion_index"]
        + 0.25 * city["density_index"]
        + 0.15 * city["tourism_index"]
        - 0.35 * city["courier_reliability_index"],
        0.45,
        2.40,
    )

    support_tickets_multiplier = clamp(
        0.70
        + 0.25 * city["economic_stress_index"]
        + 0.25 * city["crime_index"]
        + 0.20 * city["traffic_congestion_index"]
        + 0.20 * city["hotspot_inequality_index"],
        0.50,
        2.20,
    )

    manual_review_multiplier = clamp(
        0.75
        + 0.30 * city["crime_index"]
        + 0.30 * city["digital_fraud_pressure_index"]
        + 0.20 * city["hotspot_inequality_index"],
        0.50,
        2.50,
    )

    protection_adoption_multiplier = clamp(
        0.80
        + 0.35 * city["income_index"]
        + 0.10 * city["crime_index"]
        + 0.10 * city["economic_stress_index"],
        0.55,
        1.80,
    )

    reserve_hold_multiplier = clamp(
        0.85
        + 0.25 * city["crime_index"]
        + 0.20 * city["digital_fraud_pressure_index"]
        + 0.15 * city["economic_stress_index"]
        - 0.20 * city["anti_fraud_maturity_index"],
        0.60,
        1.60,
    )

    return {
        "demand_multiplier": demand_multiplier,
        "order_value_multiplier": order_value_multiplier,
        "payment_success_multiplier": payment_success_multiplier,
        "cancel_rate_multiplier": cancel_rate_multiplier,
        "refund_rate_multiplier": refund_rate_multiplier,
        "dispute_rate_multiplier": dispute_rate_multiplier,
        "fraud_incident_multiplier": fraud_incident_multiplier,
        "late_delivery_multiplier": late_delivery_multiplier,
        "support_tickets_multiplier": support_tickets_multiplier,
        "manual_review_multiplier": manual_review_multiplier,
        "protection_adoption_multiplier": protection_adoption_multiplier,
        "reserve_hold_multiplier": reserve_hold_multiplier,
    }


def parse_city_model(city_model_cfg: Any) -> Dict[str, Any]:
    if city_model_cfg is None:
        return {"enabled": False, "cities": []}

    if not isinstance(city_model_cfg, dict):
        raise ValueError("city_model must be an object")

    enabled = bool(city_model_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "cities": []}

    raw_cities = city_model_cfg.get("cities")
    if not isinstance(raw_cities, list) or not raw_cities:
        raise ValueError("city_model.cities must be a non-empty list when city_model.enabled=true")

    parsed_cities: List[Dict[str, Any]] = []
    total_share = 0.0

    for raw_city in raw_cities:
        if not isinstance(raw_city, dict):
            raise ValueError("Each city entry in city_model.cities must be an object")

        name = str(raw_city.get("name", "")).strip()
        if not name:
            raise ValueError("Each city must have non-empty 'name'")

        market_share = float(raw_city.get("market_share", 0.0))
        if market_share < 0.0:
            raise ValueError(f"city '{name}' has negative market_share")

        population_millions = float(raw_city.get("population_millions", 0.0))
        if population_millions < 0.0:
            raise ValueError(f"city '{name}' has negative population_millions")

        city: Dict[str, Any] = {
            "name": name,
            "market_share": market_share,
            "population_millions": population_millions,
        }

        for key in CITY_INDEX_KEYS:
            value = float(raw_city.get(key, 0.5))
            validate_range(f"city_model.{name}.{key}", value, 0.0, 1.0)
            city[key] = value

        city["multipliers"] = compute_city_multipliers(city)
        total_share += market_share
        parsed_cities.append(city)

    if total_share <= 0.0:
        raise ValueError("Sum of city_model market_share values must be > 0")

    for city in parsed_cities:
        city["normalized_share"] = city["market_share"] / total_share

    return {"enabled": True, "cities": parsed_cities}


def parse_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    params = {
        "runs": int(get_required(config, "simulation", "runs")),
        "days": int(get_required(config, "simulation", "days")),
        "random_seed": get_required(config, "simulation", "random_seed"),
        "mean_orders_per_day": float(get_required(config, "demand", "mean_orders_per_day")),
        "daily_growth_rate": float(get_required(config, "demand", "daily_growth_rate")),
        "day_of_week_multipliers": get_required(config, "demand", "day_of_week_multipliers"),
        "order_value_mean_usd": float(get_required(config, "demand", "order_value_mean_usd")),
        "order_value_std_usd": float(get_required(config, "demand", "order_value_std_usd")),
        "payment_success_rate": float(get_required(config, "funnel", "payment_success_rate")),
        "cancel_rate_after_payment": float(get_required(config, "funnel", "cancel_rate_after_payment")),
        "refund_rate_after_delivery": float(get_required(config, "funnel", "refund_rate_after_delivery")),
        "avg_refund_fraction": float(get_required(config, "funnel", "avg_refund_fraction")),
        "dispute_rate_after_delivery": float(get_required(config, "funnel", "dispute_rate_after_delivery")),
        "avg_dispute_fraction": float(get_required(config, "funnel", "avg_dispute_fraction")),
        "late_delivery_rate": float(get_required(config, "funnel", "late_delivery_rate")),
        "fraud_incident_rate": float(get_required(config, "funnel", "fraud_incident_rate")),
        "platform_fee_per_order_usd": float(get_required(config, "revenue", "platform_fee_per_order_usd")),
        "protection_plan_adoption_rate": float(get_required(config, "revenue", "protection_plan_adoption_rate")),
        "protection_fee_per_order_usd": float(get_required(config, "revenue", "protection_fee_per_order_usd")),
        "insurance_premium_per_protected_order_usd": float(
            get_required(config, "insurance", "insurance_premium_per_protected_order_usd")
        ),
        "insurance_coverage_ratio_for_protected_disputes": float(
            get_required(config, "insurance", "insurance_coverage_ratio_for_protected_disputes")
        ),
        "variable_fee_rate": float(get_required(config, "payment_processor", "variable_fee_rate")),
        "fixed_fee_per_successful_payment_usd": float(
            get_required(config, "payment_processor", "fixed_fee_per_successful_payment_usd")
        ),
        "chargeback_fee_usd": float(get_required(config, "payment_processor", "chargeback_fee_usd")),
        "reserve_hold_rate": float(get_required(config, "payment_processor", "reserve_hold_rate")),
        "platform_refund_loss_share": float(get_required(config, "loss_allocation", "platform_refund_loss_share")),
        "platform_dispute_loss_share": float(get_required(config, "loss_allocation", "platform_dispute_loss_share")),
        "fixed_infrastructure_cost_per_day_usd": float(
            get_required(config, "operations", "fixed_infrastructure_cost_per_day_usd")
        ),
        "infrastructure_cost_per_attempted_order_usd": float(
            get_required(config, "operations", "infrastructure_cost_per_attempted_order_usd")
        ),
        "base_support_ticket_rate": float(get_required(config, "operations", "base_support_ticket_rate")),
        "extra_support_tickets_per_late_order": float(
            get_required(config, "operations", "extra_support_tickets_per_late_order")
        ),
        "extra_support_tickets_per_refund": float(
            get_required(config, "operations", "extra_support_tickets_per_refund")
        ),
        "extra_support_tickets_per_dispute": float(
            get_required(config, "operations", "extra_support_tickets_per_dispute")
        ),
        "support_ticket_cost_usd": float(get_required(config, "operations", "support_ticket_cost_usd")),
        "manual_review_ratio_from_fraud_incidents": float(
            get_required(config, "operations", "manual_review_ratio_from_fraud_incidents")
        ),
        "manual_review_cost_usd": float(get_required(config, "operations", "manual_review_cost_usd")),
    }

    if params["runs"] <= 0:
        raise ValueError("simulation.runs must be > 0")
    if params["days"] <= 0:
        raise ValueError("simulation.days must be > 0")
    if params["mean_orders_per_day"] < 0:
        raise ValueError("demand.mean_orders_per_day must be >= 0")
    if params["order_value_mean_usd"] < 0:
        raise ValueError("demand.order_value_mean_usd must be >= 0")
    if params["order_value_std_usd"] < 0:
        raise ValueError("demand.order_value_std_usd must be >= 0")

    day_multipliers = params["day_of_week_multipliers"]
    if not isinstance(day_multipliers, list) or len(day_multipliers) != 7:
        raise ValueError("demand.day_of_week_multipliers must be a list of 7 numbers")
    if any(float(x) <= 0 for x in day_multipliers):
        raise ValueError("All day_of_week_multipliers must be > 0")
    params["day_of_week_multipliers"] = [float(x) for x in day_multipliers]

    probability_keys = [
        "payment_success_rate",
        "cancel_rate_after_payment",
        "refund_rate_after_delivery",
        "avg_refund_fraction",
        "dispute_rate_after_delivery",
        "avg_dispute_fraction",
        "late_delivery_rate",
        "fraud_incident_rate",
        "protection_plan_adoption_rate",
        "insurance_coverage_ratio_for_protected_disputes",
        "variable_fee_rate",
        "reserve_hold_rate",
        "platform_refund_loss_share",
        "platform_dispute_loss_share",
        "base_support_ticket_rate",
        "manual_review_ratio_from_fraud_incidents",
    ]
    for key in probability_keys:
        validate_range(key, float(params[key]), 0.0, 1.0)

    non_negative_keys = [
        "daily_growth_rate",
        "platform_fee_per_order_usd",
        "protection_fee_per_order_usd",
        "insurance_premium_per_protected_order_usd",
        "fixed_fee_per_successful_payment_usd",
        "chargeback_fee_usd",
        "fixed_infrastructure_cost_per_day_usd",
        "infrastructure_cost_per_attempted_order_usd",
        "extra_support_tickets_per_late_order",
        "extra_support_tickets_per_refund",
        "extra_support_tickets_per_dispute",
        "support_ticket_cost_usd",
        "manual_review_cost_usd",
    ]
    for key in non_negative_keys:
        if float(params[key]) < 0:
            raise ValueError(f"{key} must be >= 0")

    city_model = parse_city_model(config.get("city_model"))
    params["city_model_enabled"] = city_model["enabled"]
    params["cities"] = city_model["cities"]

    return params


def build_day_segments(params: Dict[str, Any], expected_orders: float) -> List[Dict[str, Any]]:
    if not params["city_model_enabled"]:
        return [
            {
                "city_name": None,
                "expected_orders": expected_orders,
                "multipliers": {
                    "demand_multiplier": 1.0,
                    "order_value_multiplier": 1.0,
                    "payment_success_multiplier": 1.0,
                    "cancel_rate_multiplier": 1.0,
                    "refund_rate_multiplier": 1.0,
                    "dispute_rate_multiplier": 1.0,
                    "fraud_incident_multiplier": 1.0,
                    "late_delivery_multiplier": 1.0,
                    "support_tickets_multiplier": 1.0,
                    "manual_review_multiplier": 1.0,
                    "protection_adoption_multiplier": 1.0,
                    "reserve_hold_multiplier": 1.0,
                },
            }
        ]

    segments: List[Dict[str, Any]] = []
    for city in params["cities"]:
        segments.append(
            {
                "city_name": city["name"],
                "expected_orders": expected_orders
                * city["normalized_share"]
                * city["multipliers"]["demand_multiplier"],
                "multipliers": city["multipliers"],
            }
        )

    return segments


def sample_poisson(lmbda: float, rng: random.Random) -> int:
    if lmbda <= 0:
        return 0
    if lmbda < 40:
        threshold = math.exp(-lmbda)
        k = 0
        product = 1.0
        while product > threshold:
            k += 1
            product *= rng.random()
        return k - 1

    estimate = int(round(rng.gauss(lmbda, math.sqrt(lmbda))))
    return max(0, estimate)


def sample_binomial(n: int, p: float, rng: random.Random) -> int:
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n

    if n < 60:
        hits = 0
        for _ in range(n):
            if rng.random() < p:
                hits += 1
        return hits

    mean = n * p
    std = math.sqrt(n * p * (1.0 - p))
    estimate = int(round(rng.gauss(mean, std)))
    return min(n, max(0, estimate))


def sample_total_order_value(order_count: int, mean: float, std: float, rng: random.Random) -> float:
    if order_count <= 0:
        return 0.0
    if std <= 0:
        return order_count * mean

    total_mean = order_count * mean
    total_std = math.sqrt(order_count) * std
    return max(0.0, rng.gauss(total_mean, total_std))


def init_city_totals(params: Dict[str, Any]) -> CityMetricsMap:
    if not params["city_model_enabled"]:
        return {}

    return {city["name"]: {key: 0.0 for key in METRIC_KEYS} for city in params["cities"]}


def apply_fixed_infra_cost(
    params: Dict[str, Any],
    totals: MetricMap,
    city_totals: CityMetricsMap,
    city_attempted_orders: Dict[str, float],
    total_attempted_orders: float,
) -> None:
    fixed_cost = params["fixed_infrastructure_cost_per_day_usd"]
    if fixed_cost <= 0.0:
        return

    totals["infra_cost_usd"] += fixed_cost
    totals["operating_costs_total_usd"] += fixed_cost
    totals["net_profit_usd"] -= fixed_cost
    totals["cash_after_reserve_usd"] -= fixed_cost

    if not city_totals:
        return

    if total_attempted_orders > 0.0:
        for city in params["cities"]:
            city_name = city["name"]
            share = city_attempted_orders.get(city_name, 0.0) / total_attempted_orders
            allocated = fixed_cost * share
            city_totals[city_name]["infra_cost_usd"] += allocated
            city_totals[city_name]["operating_costs_total_usd"] += allocated
            city_totals[city_name]["net_profit_usd"] -= allocated
            city_totals[city_name]["cash_after_reserve_usd"] -= allocated
        return

    for city in params["cities"]:
        city_name = city["name"]
        allocated = fixed_cost * city["normalized_share"]
        city_totals[city_name]["infra_cost_usd"] += allocated
        city_totals[city_name]["operating_costs_total_usd"] += allocated
        city_totals[city_name]["net_profit_usd"] -= allocated
        city_totals[city_name]["cash_after_reserve_usd"] -= allocated


def simulate_one_run(params: Dict[str, Any], rng: random.Random) -> Tuple[MetricMap, CityMetricsMap]:
    totals: MetricMap = {key: 0.0 for key in METRIC_KEYS}
    city_totals = init_city_totals(params)

    for day in range(params["days"]):
        day_multiplier = params["day_of_week_multipliers"][day % 7]
        growth_factor = (1.0 + params["daily_growth_rate"]) ** day
        expected_orders = params["mean_orders_per_day"] * day_multiplier * growth_factor

        segments = build_day_segments(params, expected_orders)
        city_attempted_orders: Dict[str, float] = {}
        total_attempted_orders = 0.0

        for segment in segments:
            city_name = segment["city_name"]
            multipliers = segment["multipliers"]

            attempted_orders = sample_poisson(segment["expected_orders"], rng)
            total_attempted_orders += float(attempted_orders)
            if city_name is not None:
                city_attempted_orders[city_name] = city_attempted_orders.get(city_name, 0.0) + attempted_orders

            payment_success_rate = clamp(
                params["payment_success_rate"] * multipliers["payment_success_multiplier"],
                0.01,
                0.999,
            )
            cancel_rate_after_payment = clamp(
                params["cancel_rate_after_payment"] * multipliers["cancel_rate_multiplier"],
                0.0,
                0.95,
            )
            refund_rate_after_delivery = clamp(
                params["refund_rate_after_delivery"] * multipliers["refund_rate_multiplier"],
                0.0,
                0.90,
            )
            dispute_rate_after_delivery = clamp(
                params["dispute_rate_after_delivery"] * multipliers["dispute_rate_multiplier"],
                0.0,
                0.70,
            )
            late_delivery_rate = clamp(
                params["late_delivery_rate"] * multipliers["late_delivery_multiplier"],
                0.0,
                0.98,
            )
            fraud_incident_rate = clamp(
                params["fraud_incident_rate"] * multipliers["fraud_incident_multiplier"],
                0.0,
                0.95,
            )
            protection_plan_adoption_rate = clamp(
                params["protection_plan_adoption_rate"]
                * multipliers["protection_adoption_multiplier"],
                0.0,
                1.0,
            )
            manual_review_ratio = clamp(
                params["manual_review_ratio_from_fraud_incidents"]
                * multipliers["manual_review_multiplier"],
                0.0,
                1.0,
            )
            base_support_ticket_rate = clamp(
                params["base_support_ticket_rate"] * multipliers["support_tickets_multiplier"],
                0.0,
                1.0,
            )
            reserve_hold_rate = clamp(
                params["reserve_hold_rate"] * multipliers["reserve_hold_multiplier"],
                0.0,
                0.95,
            )

            order_value_multiplier = multipliers["order_value_multiplier"]
            order_value_mean = max(0.0, params["order_value_mean_usd"] * order_value_multiplier)
            order_value_std = max(0.0, params["order_value_std_usd"] * order_value_multiplier)

            successful_payments = sample_binomial(attempted_orders, payment_success_rate, rng)
            cancelled_orders = sample_binomial(successful_payments, cancel_rate_after_payment, rng)
            delivered_orders = successful_payments - cancelled_orders
            protected_orders = sample_binomial(delivered_orders, protection_plan_adoption_rate, rng)

            refund_orders = sample_binomial(delivered_orders, refund_rate_after_delivery, rng)
            dispute_orders = sample_binomial(delivered_orders, dispute_rate_after_delivery, rng)
            late_orders = sample_binomial(delivered_orders, late_delivery_rate, rng)
            fraud_incidents = sample_binomial(delivered_orders, fraud_incident_rate, rng)
            manual_reviews = sample_binomial(fraud_incidents, manual_review_ratio, rng)

            delivered_value_total = sample_total_order_value(
                delivered_orders,
                order_value_mean,
                order_value_std,
                rng,
            )
            cancelled_value_total = sample_total_order_value(
                cancelled_orders,
                order_value_mean,
                order_value_std,
                rng,
            )
            processed_value_total = delivered_value_total + cancelled_value_total

            avg_ticket = (
                delivered_value_total / delivered_orders
                if delivered_orders > 0
                else order_value_mean
            )
            refund_value = min(
                delivered_value_total,
                refund_orders * avg_ticket * params["avg_refund_fraction"],
            )
            dispute_value = min(
                delivered_value_total,
                dispute_orders * avg_ticket * params["avg_dispute_fraction"],
            )

            protected_disputes = 0
            if dispute_orders > 0 and delivered_orders > 0:
                protected_share = protected_orders / delivered_orders
                protected_disputes = sample_binomial(dispute_orders, protected_share, rng)

            protected_dispute_value = (
                dispute_value * (protected_disputes / dispute_orders)
                if dispute_orders > 0
                else 0.0
            )

            platform_base_revenue = delivered_orders * params["platform_fee_per_order_usd"]
            protection_revenue = protected_orders * params["protection_fee_per_order_usd"]
            total_revenue = platform_base_revenue + protection_revenue

            processor_fees = (
                successful_payments * params["fixed_fee_per_successful_payment_usd"]
                + processed_value_total * params["variable_fee_rate"]
            )
            refund_loss = refund_value * params["platform_refund_loss_share"]
            dispute_loss_gross = (
                dispute_value * params["platform_dispute_loss_share"]
                + dispute_orders * params["chargeback_fee_usd"]
            )
            insurance_coverage = (
                protected_dispute_value
                * params["platform_dispute_loss_share"]
                * params["insurance_coverage_ratio_for_protected_disputes"]
            )
            dispute_loss_net = max(0.0, dispute_loss_gross - insurance_coverage)

            base_support_tickets = sample_binomial(delivered_orders, base_support_ticket_rate, rng)
            support_tickets = (
                float(base_support_tickets)
                + late_orders * params["extra_support_tickets_per_late_order"]
                + refund_orders * params["extra_support_tickets_per_refund"]
                + dispute_orders * params["extra_support_tickets_per_dispute"]
            ) * multipliers["support_tickets_multiplier"]

            infra_cost = (
                attempted_orders * params["infrastructure_cost_per_attempted_order_usd"]
            )
            support_cost = support_tickets * params["support_ticket_cost_usd"]
            review_cost = manual_reviews * params["manual_review_cost_usd"]
            insurance_premium_cost = (
                protected_orders * params["insurance_premium_per_protected_order_usd"]
            )

            operating_costs_total = (
                processor_fees
                + infra_cost
                + support_cost
                + review_cost
                + insurance_premium_cost
            )
            total_losses = refund_loss + dispute_loss_net
            net_profit = total_revenue - operating_costs_total - total_losses

            reserve_hold = processed_value_total * reserve_hold_rate
            cash_after_reserve = net_profit - reserve_hold

            day_metrics: MetricMap = {
                "attempted_orders": float(attempted_orders),
                "successful_payments": float(successful_payments),
                "cancelled_orders": float(cancelled_orders),
                "delivered_orders": float(delivered_orders),
                "protected_orders": float(protected_orders),
                "refund_orders": float(refund_orders),
                "dispute_orders": float(dispute_orders),
                "late_orders": float(late_orders),
                "fraud_incidents": float(fraud_incidents),
                "manual_reviews": float(manual_reviews),
                "support_tickets": float(support_tickets),
                "gmv_delivered_usd": delivered_value_total,
                "gmv_processed_usd": processed_value_total,
                "platform_base_revenue_usd": platform_base_revenue,
                "protection_revenue_usd": protection_revenue,
                "total_revenue_usd": total_revenue,
                "processor_fees_usd": processor_fees,
                "refund_loss_usd": refund_loss,
                "dispute_loss_gross_usd": dispute_loss_gross,
                "insurance_coverage_usd": insurance_coverage,
                "dispute_loss_net_usd": dispute_loss_net,
                "infra_cost_usd": infra_cost,
                "support_cost_usd": support_cost,
                "review_cost_usd": review_cost,
                "insurance_premium_cost_usd": insurance_premium_cost,
                "operating_costs_total_usd": operating_costs_total,
                "total_losses_usd": total_losses,
                "net_profit_usd": net_profit,
                "reserve_hold_usd": reserve_hold,
                "cash_after_reserve_usd": cash_after_reserve,
            }

            for key, value in day_metrics.items():
                totals[key] += value

            if city_name is not None:
                for key, value in day_metrics.items():
                    city_totals[city_name][key] += value

        apply_fixed_infra_cost(
            params,
            totals,
            city_totals,
            city_attempted_orders,
            total_attempted_orders,
        )

    return totals, city_totals


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    idx = (len(ordered) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]

    fraction = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * fraction


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_number(value: float) -> str:
    return f"{value:,.2f}"


def build_city_distributions(
    run_city_results: List[CityMetricsMap], params: Dict[str, Any]
) -> Dict[str, Any]:
    if not params["city_model_enabled"]:
        return {}

    city_distributions: Dict[str, Any] = {}
    for city in params["cities"]:
        city_name = city["name"]
        city_runs = [run.get(city_name, {key: 0.0 for key in METRIC_KEYS}) for run in run_city_results]

        distributions: Dict[str, Dict[str, float]] = {}
        for key in METRIC_KEYS:
            distributions[key] = summarize([row[key] for row in city_runs])

        per_day_expected = {
            key: distributions[key]["mean"] / params["days"] for key in METRIC_KEYS
        }

        delivered_mean = distributions["delivered_orders"]["mean"]
        delivered_day = per_day_expected["delivered_orders"]

        kpis = {
            "refund_rate": safe_div(per_day_expected["refund_orders"], delivered_day),
            "dispute_rate": safe_div(per_day_expected["dispute_orders"], delivered_day),
            "fraud_rate": safe_div(per_day_expected["fraud_incidents"], delivered_day),
            "late_rate": safe_div(per_day_expected["late_orders"], delivered_day),
            "unit_net_profit_per_delivered_order_usd": safe_div(
                distributions["net_profit_usd"]["mean"], delivered_mean
            ),
            "unit_revenue_per_delivered_order_usd": safe_div(
                distributions["total_revenue_usd"]["mean"], delivered_mean
            ),
        }

        risk_score = clamp(
            0.35 * min(1.0, safe_div(kpis["dispute_rate"], 0.012))
            + 0.35 * min(1.0, safe_div(kpis["fraud_rate"], 0.010))
            + 0.30 * min(
                1.0,
                safe_div(max(0.0, -kpis["unit_net_profit_per_delivered_order_usd"]), 1.50),
            ),
            0.0,
            1.0,
        )
        kpis["risk_score"] = risk_score
        kpis["risk_band"] = evaluate_risk_band(risk_score)

        context = {"population_millions": city["population_millions"]}
        for key in CITY_INDEX_KEYS:
            context[key] = city[key]

        city_distributions[city_name] = {
            "context": context,
            "multipliers": city["multipliers"],
            "distributions": distributions,
            "per_day_expected": per_day_expected,
            "kpis": kpis,
        }

    return city_distributions


def build_city_risk_ranking(city_distributions: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranking: List[Dict[str, Any]] = []
    for city_name, payload in city_distributions.items():
        kpis = payload["kpis"]
        ranking.append(
            {
                "city": city_name,
                "risk_score": kpis["risk_score"],
                "risk_band": kpis["risk_band"],
                "dispute_rate": kpis["dispute_rate"],
                "fraud_rate": kpis["fraud_rate"],
                "unit_net_profit_per_delivered_order_usd": kpis[
                    "unit_net_profit_per_delivered_order_usd"
                ],
            }
        )

    ranking.sort(key=lambda row: row["risk_score"], reverse=True)
    return ranking


def build_report(
    run_results: List[MetricMap],
    params: Dict[str, Any],
    run_city_results: List[CityMetricsMap],
) -> Dict[str, Any]:
    metric_distributions = {}
    for key in METRIC_KEYS:
        metric_distributions[key] = summarize([row[key] for row in run_results])

    net_values = [row["net_profit_usd"] for row in run_results]
    cash_values = [row["cash_after_reserve_usd"] for row in run_results]

    loss_probability = (
        sum(1 for value in net_values if value < 0.0) / len(net_values)
        if net_values
        else 0.0
    )
    negative_cash_probability = (
        sum(1 for value in cash_values if value < 0.0) / len(cash_values)
        if cash_values
        else 0.0
    )

    per_day_expected = {}
    for key, stats in metric_distributions.items():
        per_day_expected[key] = stats["mean"] / params["days"]

    report: Dict[str, Any] = {
        "meta": {
            "runs": params["runs"],
            "days": params["days"],
            "random_seed": params["random_seed"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "city_model_enabled": bool(params["city_model_enabled"]),
        },
        "distributions": metric_distributions,
        "risk": {
            "loss_probability": loss_probability,
            "negative_cash_after_reserve_probability": negative_cash_probability,
        },
        "per_day_expected": per_day_expected,
    }

    city_distributions = build_city_distributions(run_city_results, params)
    if city_distributions:
        report["city_distributions"] = city_distributions
        report["city_risk_ranking"] = build_city_risk_ranking(city_distributions)
        report["meta"]["city_names"] = [city["name"] for city in params["cities"]]

    return report


def print_report(report: Dict[str, Any]) -> None:
    meta = report["meta"]
    dist = report["distributions"]
    risk = report["risk"]
    per_day = report["per_day_expected"]

    print("Monte Carlo Delivery Business Simulation")
    print("=" * 40)
    print(f"Runs: {meta['runs']:,}")
    print(f"Days per run: {meta['days']:,}")
    print(f"Random seed: {meta['random_seed']}")
    print(f"City model: {'enabled' if meta['city_model_enabled'] else 'disabled'}")
    print()

    print("Expected totals over the full horizon:")
    print(f"- GMV delivered:         {format_currency(dist['gmv_delivered_usd']['mean'])}")
    print(f"- Revenue total:         {format_currency(dist['total_revenue_usd']['mean'])}")
    print(f"- Operating costs:       {format_currency(dist['operating_costs_total_usd']['mean'])}")
    print(f"- Losses (refund+dispute): {format_currency(dist['total_losses_usd']['mean'])}")
    print(f"- Net profit:            {format_currency(dist['net_profit_usd']['mean'])}")
    print(f"- Reserve hold (cash):   {format_currency(dist['reserve_hold_usd']['mean'])}")
    print(f"- Cash after reserve:    {format_currency(dist['cash_after_reserve_usd']['mean'])}")
    print()

    print("Net profit distribution:")
    print(f"- Mean:   {format_currency(dist['net_profit_usd']['mean'])}")
    print(f"- Median: {format_currency(dist['net_profit_usd']['median'])}")
    print(f"- P05:    {format_currency(dist['net_profit_usd']['p05'])}")
    print(f"- P95:    {format_currency(dist['net_profit_usd']['p95'])}")
    print(f"- Min:    {format_currency(dist['net_profit_usd']['min'])}")
    print(f"- Max:    {format_currency(dist['net_profit_usd']['max'])}")
    print()

    print("Risk indicators:")
    print(f"- Probability of net loss: {risk['loss_probability'] * 100:.2f}%")
    print(
        "- Probability of negative cash after reserve: "
        f"{risk['negative_cash_after_reserve_probability'] * 100:.2f}%"
    )
    print()

    print("Expected daily operations:")
    print(f"- Attempted orders/day:  {format_number(per_day['attempted_orders'])}")
    print(f"- Delivered orders/day:  {format_number(per_day['delivered_orders'])}")
    print(f"- Refunds/day:           {format_number(per_day['refund_orders'])}")
    print(f"- Disputes/day:          {format_number(per_day['dispute_orders'])}")
    print(f"- Support tickets/day:   {format_number(per_day['support_tickets'])}")

    city_ranking = report.get("city_risk_ranking", [])
    if city_ranking:
        print()
        print("City risk ranking:")
        for row in city_ranking:
            print(
                f"- {row['city']}: {row['risk_band']} risk "
                f"(score {row['risk_score']:.2f}), "
                f"dispute {row['dispute_rate'] * 100:.2f}%, "
                f"fraud {row['fraud_rate'] * 100:.2f}%"
            )


def save_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def save_report_csv(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dist = report["distributions"]
    per_day = report["per_day_expected"]
    risk = report["risk"]
    meta = report["meta"]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "section",
                "metric",
                "mean",
                "median",
                "p05",
                "p95",
                "min",
                "max",
                "expected_per_day",
                "value",
            ]
        )

        writer.writerow(["meta", "runs", "", "", "", "", "", "", "", meta["runs"]])
        writer.writerow(["meta", "days", "", "", "", "", "", "", "", meta["days"]])
        writer.writerow(["meta", "random_seed", "", "", "", "", "", "", "", meta["random_seed"]])
        writer.writerow(
            ["meta", "generated_at_utc", "", "", "", "", "", "", "", meta["generated_at_utc"]]
        )
        writer.writerow(
            [
                "meta",
                "city_model_enabled",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                str(meta.get("city_model_enabled", False)),
            ]
        )

        for metric in METRIC_KEYS:
            stats = dist[metric]
            writer.writerow(
                [
                    "distribution",
                    metric,
                    f"{stats['mean']:.6f}",
                    f"{stats['median']:.6f}",
                    f"{stats['p05']:.6f}",
                    f"{stats['p95']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                    f"{per_day.get(metric, 0.0):.6f}",
                    "",
                ]
            )

        writer.writerow(
            [
                "risk",
                "loss_probability",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{risk['loss_probability']:.6f}",
            ]
        )
        writer.writerow(
            [
                "risk",
                "negative_cash_after_reserve_probability",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{risk['negative_cash_after_reserve_probability']:.6f}",
            ]
        )

        city_distributions = report.get("city_distributions", {})
        for city_name, payload in city_distributions.items():
            city_dist = payload["distributions"]
            city_per_day = payload["per_day_expected"]

            for metric in METRIC_KEYS:
                stats = city_dist[metric]
                writer.writerow(
                    [
                        "city_distribution",
                        f"{city_name}.{metric}",
                        f"{stats['mean']:.6f}",
                        f"{stats['median']:.6f}",
                        f"{stats['p05']:.6f}",
                        f"{stats['p95']:.6f}",
                        f"{stats['min']:.6f}",
                        f"{stats['max']:.6f}",
                        f"{city_per_day.get(metric, 0.0):.6f}",
                        "",
                    ]
                )

            for key, value in payload["kpis"].items():
                writer.writerow(
                    [
                        "city_kpi",
                        f"{city_name}.{key}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        value if isinstance(value, str) else f"{float(value):.6f}",
                    ]
                )


def render_report_markdown(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    dist = report["distributions"]
    risk = report["risk"]
    per_day = report["per_day_expected"]

    lines: List[str] = []
    lines.append("# Delivery Simulation Report")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- Runs: {meta['runs']:,}")
    lines.append(f"- Days per run: {meta['days']:,}")
    lines.append(f"- Random seed: {meta['random_seed']}")
    lines.append(f"- Generated at (UTC): {meta['generated_at_utc']}")
    lines.append(f"- City model enabled: {meta.get('city_model_enabled', False)}")
    if meta.get("city_names"):
        lines.append(f"- City profiles: {', '.join(meta['city_names'])}")
    lines.append("")
    lines.append("## Financial Summary")
    lines.append("")
    lines.append("| Metric | Mean |")
    lines.append("|---|---:|")
    lines.append(f"| GMV delivered | {format_currency(dist['gmv_delivered_usd']['mean'])} |")
    lines.append(f"| Revenue total | {format_currency(dist['total_revenue_usd']['mean'])} |")
    lines.append(
        f"| Operating costs | {format_currency(dist['operating_costs_total_usd']['mean'])} |"
    )
    lines.append(f"| Losses (refund+dispute) | {format_currency(dist['total_losses_usd']['mean'])} |")
    lines.append(f"| Net profit | {format_currency(dist['net_profit_usd']['mean'])} |")
    lines.append(f"| Reserve hold | {format_currency(dist['reserve_hold_usd']['mean'])} |")
    lines.append(
        f"| Cash after reserve | {format_currency(dist['cash_after_reserve_usd']['mean'])} |"
    )
    lines.append("")
    lines.append("## Risk")
    lines.append("")
    lines.append(f"- Probability of net loss: {risk['loss_probability'] * 100:.2f}%")
    lines.append(
        "- Probability of negative cash after reserve: "
        f"{risk['negative_cash_after_reserve_probability'] * 100:.2f}%"
    )
    lines.append("")
    lines.append("## Net Profit Distribution")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Mean | {format_currency(dist['net_profit_usd']['mean'])} |")
    lines.append(f"| Median | {format_currency(dist['net_profit_usd']['median'])} |")
    lines.append(f"| P05 | {format_currency(dist['net_profit_usd']['p05'])} |")
    lines.append(f"| P95 | {format_currency(dist['net_profit_usd']['p95'])} |")
    lines.append(f"| Min | {format_currency(dist['net_profit_usd']['min'])} |")
    lines.append(f"| Max | {format_currency(dist['net_profit_usd']['max'])} |")
    lines.append("")
    lines.append("## Expected Daily Operations")
    lines.append("")
    lines.append("| Metric | Expected/day |")
    lines.append("|---|---:|")
    lines.append(f"| Attempted orders | {format_number(per_day['attempted_orders'])} |")
    lines.append(f"| Delivered orders | {format_number(per_day['delivered_orders'])} |")
    lines.append(f"| Refunds | {format_number(per_day['refund_orders'])} |")
    lines.append(f"| Disputes | {format_number(per_day['dispute_orders'])} |")
    lines.append(f"| Support tickets | {format_number(per_day['support_tickets'])} |")
    lines.append("")

    city_ranking = report.get("city_risk_ranking", [])
    if city_ranking:
        lines.append("## City Risk Ranking")
        lines.append("")
        lines.append(
            "| City | Risk band | Risk score | Dispute rate | Fraud rate | Unit net profit/order |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in city_ranking:
            lines.append(
                "| "
                f"{row['city']} | "
                f"{row['risk_band']} | "
                f"{row['risk_score']:.2f} | "
                f"{row['dispute_rate'] * 100:.2f}% | "
                f"{row['fraud_rate'] * 100:.2f}% | "
                f"{format_currency(row['unit_net_profit_per_delivered_order_usd'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_report_markdown(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(render_report_markdown(report))


def save_report_bundle(report: Dict[str, Any], report_dir: Path, report_name: str) -> List[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_name}.json"
    md_path = report_dir / f"{report_name}.md"
    csv_path = report_dir / f"{report_name}.csv"

    save_report(json_path, report)
    save_report_markdown(md_path, report)
    save_report_csv(csv_path, report)

    return [json_path, md_path, csv_path]


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)

    try:
        config = load_config(config_path)
        params = parse_parameters(config)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    seed = params["random_seed"]
    rng = random.Random(seed)

    run_results: List[MetricMap] = []
    run_city_results: List[CityMetricsMap] = []
    for _ in range(params["runs"]):
        run_metrics, city_metrics = simulate_one_run(params, rng)
        run_results.append(run_metrics)
        run_city_results.append(city_metrics)

    report = build_report(run_results, params, run_city_results)
    print_report(report)

    bundle_paths: List[Path] = []
    if args.report_dir:
        bundle_paths = save_report_bundle(report, Path(args.report_dir), args.report_name)
        print()
        print("Saved reporting files:")
        for path in bundle_paths:
            print(f"- {path}")

    if args.save_report:
        output_path = Path(args.save_report)
        save_report(output_path, report)
        print()
        print(f"Saved JSON report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
