#!/usr/bin/env python3
"""Monte Carlo simulator for service cost per order and business models."""

import argparse
import csv
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def truncated_normal(mean: float, stddev: float, min_value: float, max_value: float) -> float:
    if stddev <= 0:
        return clamp(mean, min_value, max_value)
    for _ in range(8):
        value = random.gauss(mean, stddev)
        if min_value <= value <= max_value:
            return value
    return clamp(mean, min_value, max_value)


def sample_orders(mean: float, volatility: float) -> int:
    if mean <= 0:
        return 0
    stddev = max(1.0, mean * volatility)
    value = random.gauss(mean, stddev)
    return max(0, int(round(value)))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_scenarios(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenarios = config.get("scenarios")
    base = {k: v for k, v in config.items() if k != "scenarios"}
    if not scenarios:
        base["scenario"] = {"name": "base", "description": ""}
        return [base]
    result = []
    for scenario in scenarios:
        merged = deep_merge(base, scenario.get("overrides", {}))
        merged["scenario"] = {
            "name": scenario.get("name", "scenario"),
            "description": scenario.get("description", ""),
        }
        result.append(merged)
    return result


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    values_sorted = sorted(values)
    return {
        "mean": statistics.mean(values_sorted),
        "p10": percentile(values_sorted, 10),
        "p50": percentile(values_sorted, 50),
        "p90": percentile(values_sorted, 90),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def compute_fixed_costs(config: Dict[str, Any], days: int) -> float:
    fixed = config.get("fixed_costs", {})
    monthly = sum(fixed.get("monthly", {}).values())
    annual = sum(fixed.get("annual", {}).values())
    return monthly * (days / 30.0) + annual * (days / 365.0)


def compute_subscription_revenue(model: Dict[str, Any], cities: List[Dict[str, Any]], days: int) -> float:
    if model.get("type") != "subscription":
        return 0.0
    monthly_fee = float(model.get("monthly_fee_usd", 0.0))
    if monthly_fee <= 0:
        return 0.0
    subscribed = 0.0
    for city in cities:
        restaurants = float(city.get("restaurants_active", 0))
        share = float(city.get("restaurant_subscription_share", 0.0))
        subscribed += restaurants * share
    months = days / 30.0
    return subscribed * monthly_fee * months


def order_revenue(model: Dict[str, Any], order_value: float) -> float:
    model_type = model.get("type")
    if model_type == "fixed":
        revenue = float(model.get("fee_usd", 0.0))
    elif model_type == "percent":
        revenue = float(model.get("percent", 0.0)) * order_value
    elif model_type == "hybrid":
        revenue = float(model.get("fee_usd", 0.0)) + float(model.get("percent", 0.0)) * order_value
    elif model_type == "subscription":
        revenue = float(model.get("per_order_fee_usd", 0.0))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    min_fee = model.get("min_fee_usd")
    max_fee = model.get("max_fee_usd")
    if min_fee is not None:
        revenue = max(revenue, float(min_fee))
    if max_fee is not None:
        revenue = min(revenue, float(max_fee))
    return revenue


def compute_break_even_pricing(report: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    sim = report["simulation"]
    orders_mean = sim["orders"]["mean"]
    total_cost_mean = sim["variable_cost_usd"]["mean"] + sim["fixed_cost_usd"]["mean"]
    cost_per_order = total_cost_mean / orders_mean if orders_mean else 0.0
    avg_order_value = report["average_order_value_usd"]["mean"]

    fixed_fee = cost_per_order
    percent_fee = cost_per_order / avg_order_value if avg_order_value else 0.0

    hybrid_model = next((m for m in config["business_models"] if m.get("type") == "hybrid"), None)
    hybrid_fixed = 0.0
    hybrid_percent = 0.0
    hybrid_fixed_at_percent = 0.0
    hybrid_percent_at_fixed = 0.0
    if hybrid_model:
        hybrid_fixed = float(hybrid_model.get("fee_usd", 0.0))
        hybrid_percent = float(hybrid_model.get("percent", 0.0))
        hybrid_fixed_at_percent = cost_per_order - hybrid_percent * avg_order_value
        if avg_order_value:
            hybrid_percent_at_fixed = (cost_per_order - hybrid_fixed) / avg_order_value

    subscription_model = next((m for m in config["business_models"] if m.get("type") == "subscription"), None)
    monthly_fee = 0.0
    per_order_fee = 0.0
    monthly_needed = 0.0
    per_order_needed = 0.0
    if subscription_model:
        monthly_fee = float(subscription_model.get("monthly_fee_usd", 0.0))
        per_order_fee = float(subscription_model.get("per_order_fee_usd", 0.0))
        subscribed = 0.0
        for city in config["cities"]:
            restaurants = float(city.get("restaurants_active", 0))
            share = float(city.get("restaurant_subscription_share", 0.0))
            subscribed += restaurants * share
        months = config["simulation"]["days"] / 30.0
        if subscribed > 0 and months > 0:
            monthly_needed = (total_cost_mean - per_order_fee * orders_mean) / (subscribed * months)
        if orders_mean > 0:
            per_order_needed = (total_cost_mean - monthly_fee * subscribed * months) / orders_mean

    return {
        "assumptions": {
            "orders_mean": orders_mean,
            "avg_order_value_usd": avg_order_value,
            "total_cost_mean_usd": total_cost_mean,
        },
        "fixed_fee_usd": fixed_fee,
        "percent_of_order": percent_fee,
        "hybrid_fixed_usd_at_percent": {
            "percent": hybrid_percent,
            "fixed_fee_usd": hybrid_fixed_at_percent,
        },
        "hybrid_percent_at_fixed": {
            "fixed_fee_usd": hybrid_fixed,
            "percent": hybrid_percent_at_fixed,
        },
        "subscription_monthly_fee_usd_at_per_order": {
            "per_order_fee_usd": per_order_fee,
            "monthly_fee_usd": monthly_needed,
        },
        "subscription_per_order_fee_usd_at_monthly": {
            "monthly_fee_usd": monthly_fee,
            "per_order_fee_usd": per_order_needed,
        },
    }


def simulate_run(config: Dict[str, Any]) -> Dict[str, Any]:
    simulation = config["simulation"]
    demand = config["demand"]
    usage = config["usage"]
    usage_multipliers = config.get("usage_multipliers", {})
    failure_rates = config.get("failure_rates", {})
    costs = config["costs"]
    payment = config.get("payment_processor", {})
    cities = config["cities"]

    days = int(simulation["days"])
    weekday_multipliers = demand.get("weekday_multipliers", [1.0] * 7)
    order_volatility = float(demand.get("order_volatility", 0.2))

    tracking_updates_multiplier = float(usage_multipliers.get("tracking_updates_multiplier", 1.0))
    support_rate_multiplier = float(usage_multipliers.get("support_rate_multiplier", 1.0))
    map_requests_multiplier = float(usage_multipliers.get("map_requests_multiplier", 1.0))
    api_requests_multiplier = float(usage_multipliers.get("api_requests_multiplier", 1.0))
    ml_calls_multiplier = float(usage_multipliers.get("ml_calls_multiplier", 1.0))

    map_retry_rate = float(failure_rates.get("map_retry_rate", 0.0))
    api_retry_rate = float(failure_rates.get("api_retry_rate", 0.0))
    ml_retry_rate = float(failure_rates.get("ml_retry_rate", 0.0))
    notification_retry_rate = float(failure_rates.get("notification_retry_rate", 0.0))
    support_escalation_rate = float(failure_rates.get("support_escalation_rate", 0.0))

    payment_include = bool(payment.get("include_in_costs", False))
    payment_percent = float(payment.get("percent_fee", 0.0))
    payment_fixed = float(payment.get("fixed_fee_usd", 0.0))

    counts = {
        "orders": 0,
        "tracking_updates": 0,
        "map_requests": 0.0,
        "api_requests": 0.0,
        "ml_calls": 0.0,
        "push_count": 0.0,
        "sms_count": 0.0,
        "email_count": 0.0,
        "support_minutes": 0.0,
        "support_tickets": 0,
        "data_gb": 0.0,
        "order_value_usd": 0.0,
        "payment_fees_usd": 0.0,
    }
    city_stats: Dict[str, Dict[str, float]] = {}
    for city in cities:
        city_stats[city["name"]] = {
            "orders": 0,
            "variable_cost_usd": 0.0,
        }

    model_totals = {model["name"]: {"revenue": 0.0, "orders": 0} for model in config["business_models"]}

    for day in range(days):
        day_multiplier = weekday_multipliers[day % len(weekday_multipliers)]
        for city in cities:
            orders_mean = city["orders_per_day"] * day_multiplier
            orders_today = sample_orders(orders_mean, order_volatility)
            if orders_today == 0:
                continue
            city_name = city["name"]
            value_cfg = city["order_value_usd"]
            for _ in range(orders_today):
                order_value = truncated_normal(
                    value_cfg["mean"],
                    value_cfg["stddev"],
                    value_cfg["min"],
                    value_cfg["max"],
                )

                avg_minutes = float(city["avg_delivery_minutes"])
                updates_per_min = float(city["tracking_updates_per_min"]) * tracking_updates_multiplier
                traffic_multiplier = float(usage.get("tracking_traffic_multiplier", 0.0))
                jitter = float(usage.get("tracking_jitter", 0.0))
                updates_mean = avg_minutes * updates_per_min * (1.0 + city["traffic_index"] * traffic_multiplier)
                updates = max(1, int(round(truncated_normal(updates_mean, updates_mean * jitter, 1, updates_mean * 3))))

                complexity_multiplier = float(usage.get("map_complexity_multiplier", 0.0))
                map_requests = float(usage["base_map_requests"]) + (
                    updates * float(usage["map_requests_per_tracking_update"]) * (1.0 + city["complexity_index"] * complexity_multiplier)
                )
                map_requests *= map_requests_multiplier
                map_requests *= 1.0 + map_retry_rate

                api_requests = float(usage["base_api_requests"]) + updates * float(usage["api_requests_per_tracking_update"])
                api_requests *= api_requests_multiplier
                api_requests *= 1.0 + api_retry_rate

                support_ticket_rate = (
                    float(usage["support_ticket_rate"]) * float(city.get("support_rate_multiplier", 1.0)) * support_rate_multiplier
                )
                has_support = random.random() < support_ticket_rate
                support_minutes = 0.0
                if has_support:
                    support_cfg = usage["support_minutes"]
                    support_minutes = truncated_normal(
                        support_cfg["mean"],
                        support_cfg["stddev"],
                        support_cfg["min"],
                        support_cfg["max"],
                    )
                    support_minutes *= 1.0 + support_escalation_rate

                ml_calls = float(usage["base_ml_calls"])
                if has_support:
                    ml_calls += float(usage["ml_calls_per_support_ticket"])
                ml_calls *= ml_calls_multiplier
                ml_calls *= 1.0 + ml_retry_rate

                notification_cfg = usage["notification"]
                push_count = float(notification_cfg["push_count"]) * (1.0 + notification_retry_rate)
                sms_count = 1.0 if random.random() < float(notification_cfg["sms_rate"]) else 0.0
                email_count = 1.0 if random.random() < float(notification_cfg["email_rate"]) else 0.0
                sms_count *= 1.0 + notification_retry_rate
                email_count *= 1.0 + notification_retry_rate

                data_cfg = usage["data_mb"]
                data_mb = float(data_cfg["base_order_mb"]) + updates * float(data_cfg["tracking_update_mb"])
                data_gb = data_mb / 1024.0

                payment_fee = 0.0
                if payment_include:
                    payment_fee = order_value * payment_percent + payment_fixed

                counts["orders"] += 1
                counts["tracking_updates"] += updates
                counts["map_requests"] += map_requests
                counts["api_requests"] += api_requests
                counts["ml_calls"] += ml_calls
                counts["push_count"] += push_count
                counts["sms_count"] += sms_count
                counts["email_count"] += email_count
                counts["support_minutes"] += support_minutes
                counts["support_tickets"] += 1 if has_support else 0
                counts["data_gb"] += data_gb
                counts["order_value_usd"] += order_value
                counts["payment_fees_usd"] += payment_fee

                order_variable_cost = (
                    map_requests * costs["map_request_usd"]
                    + api_requests * costs["api_request_usd"]
                    + ml_calls * costs["ml_call_usd"]
                    + push_count * costs["push_usd"]
                    + sms_count * costs["sms_usd"]
                    + email_count * costs["email_usd"]
                    + support_minutes * costs["support_minute_usd"]
                    + data_gb * costs["storage_gb_usd"]
                    + payment_fee
                )
                city_stats[city_name]["variable_cost_usd"] += order_variable_cost

                for model in config["business_models"]:
                    revenue = order_revenue(model, order_value)
                    model_totals[model["name"]]["revenue"] += revenue
                    model_totals[model["name"]]["orders"] += 1

            city_stats[city_name]["orders"] += orders_today

    cost_breakdown = {
        "map_cost_usd": counts["map_requests"] * costs["map_request_usd"],
        "api_cost_usd": counts["api_requests"] * costs["api_request_usd"],
        "ml_cost_usd": counts["ml_calls"] * costs["ml_call_usd"],
        "push_cost_usd": counts["push_count"] * costs["push_usd"],
        "sms_cost_usd": counts["sms_count"] * costs["sms_usd"],
        "email_cost_usd": counts["email_count"] * costs["email_usd"],
        "support_cost_usd": counts["support_minutes"] * costs["support_minute_usd"],
        "storage_cost_usd": counts["data_gb"] * costs["storage_gb_usd"],
        "payment_fees_usd": counts["payment_fees_usd"],
    }

    variable_cost = sum(cost_breakdown.values())
    fixed_cost = compute_fixed_costs(config, days)

    orders_total = counts["orders"]
    average_order_value = counts["order_value_usd"] / orders_total if orders_total else 0.0
    usage_per_order = {
        "map_requests": counts["map_requests"] / orders_total if orders_total else 0.0,
        "api_requests": counts["api_requests"] / orders_total if orders_total else 0.0,
        "ml_calls": counts["ml_calls"] / orders_total if orders_total else 0.0,
        "tracking_updates": counts["tracking_updates"] / orders_total if orders_total else 0.0,
        "push_count": counts["push_count"] / orders_total if orders_total else 0.0,
        "sms_count": counts["sms_count"] / orders_total if orders_total else 0.0,
        "email_count": counts["email_count"] / orders_total if orders_total else 0.0,
        "support_minutes": counts["support_minutes"] / orders_total if orders_total else 0.0,
        "support_tickets": counts["support_tickets"] / orders_total if orders_total else 0.0,
        "data_gb": counts["data_gb"] / orders_total if orders_total else 0.0,
    }

    model_results: Dict[str, Dict[str, float]] = {}
    for model in config["business_models"]:
        model_name = model["name"]
        revenue = model_totals[model_name]["revenue"] + compute_subscription_revenue(model, cities, days)
        orders = model_totals[model_name]["orders"]
        net_profit = revenue - variable_cost - fixed_cost
        revenue_per_order = revenue / orders if orders else 0.0
        cost_per_order = (variable_cost + fixed_cost) / orders if orders else 0.0
        margin = (revenue - variable_cost - fixed_cost) / revenue if revenue else 0.0
        model_results[model_name] = {
            "revenue_usd": revenue,
            "orders": orders,
            "net_profit_usd": net_profit,
            "revenue_per_order_usd": revenue_per_order,
            "cost_per_order_usd": cost_per_order,
            "margin": margin,
        }

    return {
        "counts": counts,
        "variable_cost_usd": variable_cost,
        "fixed_cost_usd": fixed_cost,
        "cost_breakdown": cost_breakdown,
        "average_order_value_usd": average_order_value,
        "usage_per_order": usage_per_order,
        "models": model_results,
        "city_stats": city_stats,
    }


def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    simulation = config["simulation"]
    runs = int(simulation["runs"])
    costs = config["costs"]
    payment = config.get("payment_processor", {})

    model_names = [model["name"] for model in config["business_models"]]
    model_metrics = {
        name: {
            "net_profit_usd": [],
            "revenue_usd": [],
            "revenue_per_order_usd": [],
            "cost_per_order_usd": [],
            "margin": [],
        }
        for name in model_names
    }
    cost_breakdowns = []
    orders_list = []
    variable_costs = []
    fixed_costs = []
    avg_order_values = []
    usage_rollups: Dict[str, List[float]] = {
        "map_requests": [],
        "api_requests": [],
        "ml_calls": [],
        "tracking_updates": [],
        "push_count": [],
        "sms_count": [],
        "email_count": [],
        "support_minutes": [],
        "support_tickets": [],
        "data_gb": [],
    }
    city_rollups: Dict[str, Dict[str, List[float]]] = {}

    for city in config["cities"]:
        city_rollups[city["name"]] = {"orders": [], "cost_per_order_usd": []}

    for _ in range(runs):
        run = simulate_run(config)
        orders = run["counts"]["orders"]
        orders_list.append(orders)
        variable_costs.append(run["variable_cost_usd"])
        fixed_costs.append(run["fixed_cost_usd"])
        avg_order_values.append(run["average_order_value_usd"])
        cost_breakdowns.append(run["cost_breakdown"])

        for key in usage_rollups:
            usage_rollups[key].append(run["usage_per_order"][key])

        for model_name in model_names:
            result = run["models"][model_name]
            model_metrics[model_name]["net_profit_usd"].append(result["net_profit_usd"])
            model_metrics[model_name]["revenue_usd"].append(result["revenue_usd"])
            model_metrics[model_name]["revenue_per_order_usd"].append(result["revenue_per_order_usd"])
            model_metrics[model_name]["cost_per_order_usd"].append(result["cost_per_order_usd"])
            model_metrics[model_name]["margin"].append(result["margin"])

        for city_name, stats in run["city_stats"].items():
            orders_city = stats["orders"]
            cost_city = stats["variable_cost_usd"] + (run["fixed_cost_usd"] * (orders_city / orders if orders else 0))
            cost_per_order = cost_city / orders_city if orders_city else 0.0
            city_rollups[city_name]["orders"].append(orders_city)
            city_rollups[city_name]["cost_per_order_usd"].append(cost_per_order)

    cost_breakdown_avg = {}
    if cost_breakdowns:
        keys = cost_breakdowns[0].keys()
        for key in keys:
            cost_breakdown_avg[key] = statistics.mean([c[key] for c in cost_breakdowns])

    usage_summary = {key: summarize(values) for key, values in usage_rollups.items()}
    avg_order_value_summary = summarize(avg_order_values)
    unit_costs = dict(costs)
    unit_costs["payment_percent_fee"] = float(payment.get("percent_fee", 0.0))
    unit_costs["payment_fixed_fee_usd"] = float(payment.get("fixed_fee_usd", 0.0))
    unit_costs["payment_include_in_costs"] = bool(payment.get("include_in_costs", False))
    assumptions = {
        "usage_multipliers": config.get("usage_multipliers", {}),
        "failure_rates": config.get("failure_rates", {}),
        "payment_processor": payment,
    }

    model_summary = {}
    for model_name, metrics in model_metrics.items():
        net_profit_summary = summarize(metrics["net_profit_usd"])
        revenue_summary = summarize(metrics["revenue_usd"])
        cost_per_order_summary = summarize(metrics["cost_per_order_usd"])
        revenue_per_order_summary = summarize(metrics["revenue_per_order_usd"])
        margin_summary = summarize(metrics["margin"])
        profitable_runs = sum(1 for value in metrics["net_profit_usd"] if value > 0)
        profitability_rate = profitable_runs / runs if runs else 0.0
        model_summary[model_name] = {
            "net_profit_usd": net_profit_summary,
            "revenue_usd": revenue_summary,
            "cost_per_order_usd": cost_per_order_summary,
            "revenue_per_order_usd": revenue_per_order_summary,
            "margin": margin_summary,
            "profitability_rate": profitability_rate,
        }

    city_summary = {}
    for city_name, series in city_rollups.items():
        city_summary[city_name] = {
            "orders": summarize(series["orders"]),
            "cost_per_order_usd": summarize(series["cost_per_order_usd"]),
        }

    best_model = None
    best_profit = None
    for model_name, summary in model_summary.items():
        mean_profit = summary["net_profit_usd"]["mean"]
        if best_profit is None or mean_profit > best_profit:
            best_profit = mean_profit
            best_model = model_name

    report_stub = {
        "simulation": {
            "runs": runs,
            "days": config["simulation"]["days"],
            "orders": summarize(orders_list),
            "variable_cost_usd": summarize(variable_costs),
            "fixed_cost_usd": summarize(fixed_costs),
        },
        "average_order_value_usd": avg_order_value_summary,
    }
    break_even_pricing = compute_break_even_pricing(report_stub, config)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": config.get("scenario", {"name": "base", "description": ""}),
        "simulation": report_stub["simulation"],
        "cost_breakdown_avg_usd": cost_breakdown_avg,
        "usage_per_order": usage_summary,
        "average_order_value_usd": avg_order_value_summary,
        "unit_costs": unit_costs,
        "assumptions": assumptions,
        "models": model_summary,
        "city_summary": city_summary,
        "break_even_pricing": break_even_pricing,
        "recommendation": {
            "best_model_by_mean_profit": best_model,
            "best_model_mean_profit_usd": best_profit if best_profit is not None else 0.0,
        },
    }
    return report


def run_all_scenarios(config: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = get_scenarios(config)
    scenario_reports = {}
    for scenario_config in scenarios:
        scenario_report = run_simulation(scenario_config)
        scenario_name = scenario_report.get("scenario", {}).get("name", "scenario")
        scenario_reports[scenario_name] = scenario_report
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenario_reports,
    }


def write_reports(report: Dict[str, Any], report_dir: Path, report_name: str, save_report: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(save_report) if save_report else report_dir / f"{report_name}.json"
    md_path = report_dir / f"{report_name}.md"
    csv_path = report_dir / f"{report_name}.csv"

    json_path.write_text(json.dumps(report, indent=2))

    scenarios = report.get("scenarios")
    if scenarios is None:
        scenarios = {"base": report}

    model_rows = []
    for scenario_name, scenario_report in scenarios.items():
        for model_name, summary in scenario_report["models"].items():
            model_rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "net_profit_mean": summary["net_profit_usd"]["mean"],
                    "net_profit_p10": summary["net_profit_usd"]["p10"],
                    "net_profit_p90": summary["net_profit_usd"]["p90"],
                    "revenue_mean": summary["revenue_usd"]["mean"],
                    "cost_per_order_mean": summary["cost_per_order_usd"]["mean"],
                    "revenue_per_order_mean": summary["revenue_per_order_usd"]["mean"],
                    "margin_mean": summary["margin"]["mean"],
                    "profitability_rate": summary["profitability_rate"],
                }
            )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=model_rows[0].keys()) if model_rows else None
        if writer:
            writer.writeheader()
            writer.writerows(model_rows)

    lines = []
    lines.append("# Service Cost Simulation Report")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append("")

    for scenario_name, scenario_report in scenarios.items():
        scenario_info = scenario_report.get("scenario", {})
        description = scenario_info.get("description", "")
        lines.append(f"## Scenario: {scenario_name}")
        if description:
            lines.append(f"Description: {description}")
        lines.append("")

        sim = scenario_report["simulation"]
        lines.append("### Simulation Summary")
        lines.append("")
        lines.append(f"Runs: {sim['runs']}")
        lines.append(f"Days per run: {sim['days']}")
        lines.append(f"Orders per run (mean): {sim['orders']['mean']:.2f}")
        lines.append(f"Average order value (mean): ${scenario_report['average_order_value_usd']['mean']:.2f}")
        lines.append(f"Variable cost per run (mean): ${sim['variable_cost_usd']['mean']:.2f}")
        lines.append(f"Fixed cost per run (mean): ${sim['fixed_cost_usd']['mean']:.2f}")
        lines.append("")

        unit_costs = scenario_report.get("unit_costs", {})
        lines.append("### Unit Costs")
        lines.append("")
        lines.append("| Service | Unit Cost |")
        lines.append("| --- | --- |")
        lines.append(f"| Map request | ${unit_costs.get('map_request_usd', 0):.6f} |")
        lines.append(f"| API request | ${unit_costs.get('api_request_usd', 0):.6f} |")
        lines.append(f"| ML call | ${unit_costs.get('ml_call_usd', 0):.4f} |")
        lines.append(f"| Push notification | ${unit_costs.get('push_usd', 0):.6f} |")
        lines.append(f"| SMS | ${unit_costs.get('sms_usd', 0):.4f} |")
        lines.append(f"| Email | ${unit_costs.get('email_usd', 0):.6f} |")
        lines.append(f"| Support minute | ${unit_costs.get('support_minute_usd', 0):.2f} |")
        lines.append(f"| Storage (GB) | ${unit_costs.get('storage_gb_usd', 0):.4f} |")
        lines.append(\n            f\"| Payment fee | {unit_costs.get('payment_percent_fee', 0) * 100:.2f}% + ${unit_costs.get('payment_fixed_fee_usd', 0):.2f} |\"\n        )\n        lines.append(\"\")\n\n        usage = scenario_report[\"usage_per_order\"]\n        lines.append(\"### Usage Per Order (Mean/P10/P90)\")\n        lines.append(\"\")\n        lines.append(\"| Metric | Mean | P10 | P90 |\")\n        lines.append(\"| --- | --- | --- | --- |\")\n        for key, summary in usage.items():\n            lines.append(\n                \"| {key} | {mean:.2f} | {p10:.2f} | {p90:.2f} |\".format(\n                    key=key,\n                    mean=summary[\"mean\"],\n                    p10=summary[\"p10\"],\n                    p90=summary[\"p90\"],\n                )\n            )\n        lines.append(\"\")\n\n        lines.append(\"### Average Cost Breakdown\")\n        lines.append(\"\")\n        for key, value in scenario_report[\"cost_breakdown_avg_usd\"].items():\n            lines.append(f\"- {key}: ${value:.2f}\")\n        lines.append(\"\")\n\n        lines.append(\"### Business Model Comparison\")\n        lines.append(\"\")\n        lines.append(\"| Model | Net Profit Mean | Net Profit P10 | Net Profit P90 | Revenue Mean | Cost/Order Mean | Revenue/Order Mean | Margin Mean | Profitability Rate |\")\n        lines.append(\"| --- | --- | --- | --- | --- | --- | --- | --- | --- |\")\n        for row in [r for r in model_rows if r[\"scenario\"] == scenario_name]:\n            lines.append(\n                \"| {model} | ${net_profit_mean:.2f} | ${net_profit_p10:.2f} | ${net_profit_p90:.2f} | ${revenue_mean:.2f} | ${cost_per_order_mean:.2f} | ${revenue_per_order_mean:.2f} | {margin_mean:.2%} | {profitability_rate:.2%} |\".format(\n                    **row\n                )\n            )\n        lines.append(\"\")\n\n        break_even = scenario_report.get(\"break_even_pricing\", {})\n        if break_even:\n            lines.append(\"### Break-even Pricing\")\n            lines.append(\"\")\n            lines.append(f\"- Fixed fee break-even: ${break_even['fixed_fee_usd']:.2f}\")\n            lines.append(f\"- Percent break-even: {break_even['percent_of_order'] * 100:.2f}%\")\n            lines.append(\n                \"- Hybrid break-even at percent {percent:.2%}: fixed ${fixed_fee_usd:.2f}\".format(\n                    **break_even[\"hybrid_fixed_usd_at_percent\"]\n                )\n            )\n            lines.append(\n                \"- Hybrid break-even at fixed ${fixed_fee_usd:.2f}: percent {percent:.2%}\".format(\n                    **break_even[\"hybrid_percent_at_fixed\"]\n                )\n            )\n            lines.append(\n                \"- Subscription break-even at per-order ${per_order_fee_usd:.2f}: monthly ${monthly_fee_usd:.2f}\".format(\n                    **break_even[\"subscription_monthly_fee_usd_at_per_order\"]\n                )\n            )\n            lines.append(\n                \"- Subscription break-even at monthly ${monthly_fee_usd:.2f}: per-order ${per_order_fee_usd:.2f}\".format(\n                    **break_even[\"subscription_per_order_fee_usd_at_monthly\"]\n                )\n            )\n            lines.append(\"\")\n\n        lines.append(\"### City Cost Per Order\")\n        lines.append(\"\")\n        lines.append(\"| City | Orders Mean | Cost/Order Mean | Cost/Order P10 | Cost/Order P90 |\")\n        lines.append(\"| --- | --- | --- | --- | --- |\")\n        for city_name, summary in scenario_report[\"city_summary\"].items():\n            lines.append(\n                \"| {city} | {orders_mean:.2f} | ${cost_mean:.2f} | ${cost_p10:.2f} | ${cost_p90:.2f} |\".format(\n                    city=city_name,\n                    orders_mean=summary[\"orders\"][\"mean\"],\n                    cost_mean=summary[\"cost_per_order_usd\"][\"mean\"],\n                    cost_p10=summary[\"cost_per_order_usd\"][\"p10\"],\n                    cost_p90=summary[\"cost_per_order_usd\"][\"p90\"],\n                )\n            )\n        lines.append(\"\")\n\n        rec = scenario_report[\"recommendation\"]\n        lines.append(\"### Recommendation\")\n        lines.append(\"\")\n        if rec[\"best_model_by_mean_profit\"]:\n            lines.append(\n                f\"Best model by mean profit: {rec['best_model_by_mean_profit']} (${'{:.2f}'.format(rec['best_model_mean_profit_usd'])}).\"\n            )\n        else:\n            lines.append(\"No clear best model computed.\")\n        lines.append(\"\")\n\n    md_path.write_text(\"\\n\".join(lines))\n*** End Patch"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Service cost simulation and business model comparison.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--report-dir", default="reports", help="Directory for report outputs.")
    parser.add_argument("--report-name", default=None, help="Report base name (without extension).")
    parser.add_argument("--save-report", default="", help="Optional explicit JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())

    seed = config.get("simulation", {}).get("random_seed")
    if seed is not None:
        random.seed(seed)

    report_name = args.report_name or config.get("reporting", {}).get("report_name", "service_cost_report")
    report = run_simulation(config)

    write_reports(report, Path(args.report_dir), report_name, args.save_report)


if __name__ == "__main__":
    main()
