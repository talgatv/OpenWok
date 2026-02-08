# Calculation Notes (180 days)

This document explains (in words + formulas) how the 180-day results were calculated.

## What Was Run

- Config used: `config_180d.json`
- Runs: 2000
- Horizon: 180 days
- Seed: 42
- City model: enabled (3 city profiles)

Commands:

```bash
cd /Users/Talgat2/Documents/Projects/OpenWok/OpenWok/programs/delivery-monte-carlo
python3 simulate.py --config config_180d.json --report-dir reports --report-name simulation_report_180d
python3 analyze_report.py --input-report reports/simulation_report_180d.json --analysis-config analysis_config.json --output-dir reports --output-name analysis_report_180d
```

Output files (generated):
- `reports/simulation_report_180d.json`
- `reports/simulation_report_180d.md`
- `reports/simulation_report_180d.csv`
- `reports/analysis_report_180d.json`
- `reports/analysis_report_180d.md`
- `reports/analysis_report_180d.csv`

## 180-Day Headline Results (from reports)

From `simulation_report_180d.md`:
- GMV delivered: $8,686,149.43
- Revenue total: $344,211.13
- Operating costs: $535,323.43
- Losses (refund + dispute): $196,095.11
- Net profit: -$387,207.41
- Reserve hold: $254,730.01
- Cash after reserve: -$641,937.42

From `analysis_report_180d.md`:
- Unit revenue per delivered order: $1.07
- Unit total cost per delivered order: $2.28
- Unit net profit per delivered order: -$1.21
- Net profit margin: -112.49%
- Processor fee ratio vs revenue: 109.79%
- Refund rate: 3.47%
- Dispute rate: 1.42%
- Fraud rate: 1.26%

City risk ranking (rates are per delivered order):
- Hot_Risk_Urban_Profile: dispute 1.79%, fraud 1.66%, unit net/order -$1.33
- Los_Angeles: dispute 1.44%, fraud 1.28%, unit net/order -$1.33
- Calm_Mid_Size_Profile: dispute 0.97%, fraud 0.77%, unit net/order -$0.75

## Core Model: Monte Carlo Structure

The simulator runs `runs` independent simulations. Each simulation is `days` long.
For each day we sample:
- order demand
- funnel conversion
- operational incidents (refund/dispute/fraud/late)
- financial outcomes

At the end we aggregate daily totals into a run total.
Across runs we compute distribution stats: mean, median, P05/P95, min/max.

## Demand Model (Orders Per Day)

For day index `t = 0..days-1`:

1) Base expected orders:

```
expected_orders(t) = mean_orders_per_day
                    * day_of_week_multipliers[t % 7]
                    * (1 + daily_growth_rate) ** t
```

2) If city model enabled, split the day into city segments:

```
expected_orders_city(t) = expected_orders(t)
                        * normalized_city_share
                        * demand_multiplier(city)
```

Then sample actual attempted orders:

```
attempted_orders_city(t) ~ Poisson(expected_orders_city(t))
```

## City Parameters -> Probability Multipliers

Each city profile has indices in [0..1] + `population_millions`.
The simulator converts them into multipliers for key probabilities:
- payment success
- cancellations after payment
- refunds
- disputes
- fraud incidents
- late delivery
- reserve hold pressure

Important indices that increase abuse:
- `crime_index`
- `economic_stress_index`
- `digital_fraud_pressure_index`
- `hotspot_inequality_index`

Key index that reduces abuse:
- `anti_fraud_maturity_index`

(Details are in `CITY_PARAMETERS_GUIDE.md` and in `compute_city_multipliers()` inside `simulate.py`.)

For any base probability `p_base` and multiplier `m_city`:

```
p_city = clamp(p_base * m_city, min_allowed, max_allowed)
```

## Funnel + Incident Sampling (Per City Segment)

Given attempted orders `A`:

```
successful_payments ~ Binomial(A, payment_success_rate_city)
cancelled_orders    ~ Binomial(successful_payments, cancel_rate_after_payment_city)
delivered_orders    = successful_payments - cancelled_orders
protected_orders    ~ Binomial(delivered_orders, protection_plan_adoption_rate_city)

refund_orders  ~ Binomial(delivered_orders, refund_rate_after_delivery_city)
dispute_orders ~ Binomial(delivered_orders, dispute_rate_after_delivery_city)
late_orders    ~ Binomial(delivered_orders, late_delivery_rate_city)
fraud_incidents ~ Binomial(delivered_orders, fraud_incident_rate_city)
manual_reviews  ~ Binomial(fraud_incidents, manual_review_ratio_city)
```

## Order Value (GMV)

We approximate total delivered GMV for the segment using a normal approximation:

```
delivered_value_total ~ Normal(delivered_orders * mean_ticket_city,
                              sqrt(delivered_orders) * std_ticket_city)
```

Same for cancelled (paid then cancelled):

```
cancelled_value_total ~ Normal(cancelled_orders * mean_ticket_city,
                              sqrt(cancelled_orders) * std_ticket_city)

processed_value_total = delivered_value_total + cancelled_value_total
```

`mean_ticket_city` and `std_ticket_city` are scaled by the city `order_value_multiplier`.

## Revenue Model

Platform revenue (this is the model used in code):

```
platform_base_revenue = delivered_orders * platform_fee_per_order_usd
protection_revenue    = protected_orders * protection_fee_per_order_usd

total_revenue = platform_base_revenue + protection_revenue
```

## Processor Fees, Operating Costs, Losses

Processor fees are modeled as:

```
processor_fees = successful_payments * fixed_fee_per_successful_payment_usd
               + processed_value_total * variable_fee_rate
```

Operating costs include:
- processor fees
- infra variable: `attempted_orders * infrastructure_cost_per_attempted_order_usd`
- infra fixed: `fixed_infrastructure_cost_per_day_usd` (allocated per day)
- support: tickets * `support_ticket_cost_usd`
- manual review: reviews * `manual_review_cost_usd`
- insurance premium: `protected_orders * insurance_premium_per_protected_order_usd`

Refund/dispute losses for the platform:

```
refund_loss = refund_value * platform_refund_loss_share

dispute_loss_gross = dispute_value * platform_dispute_loss_share
                  + dispute_orders * chargeback_fee_usd

insurance_coverage = protected_dispute_value
                   * platform_dispute_loss_share
                   * insurance_coverage_ratio_for_protected_disputes

dispute_loss_net = max(0, dispute_loss_gross - insurance_coverage)

total_losses = refund_loss + dispute_loss_net
```

Net profit:

```
net_profit = total_revenue - operating_costs_total - total_losses
```

Cash after reserve:

```
reserve_hold = processed_value_total * reserve_hold_rate_city
cash_after_reserve = net_profit - reserve_hold
```

## Why It Is Negative In This Run (Interpretation)

Based on the report:
- Revenue per delivered order is about $1.07.
- Total costs + losses per delivered order are about $2.28.
- This yields about -$1.21 per delivered order.

The dominant reason is the assumption that the platform pays processing fees on full GMV,
while the platform only earns ~$1/order (+ small Protection revenue).
That makes `processor_fees_usd` larger than `total_revenue_usd` in this scenario.

This is not a statement about reality; it is a statement about the assumptions in `config_180d.json`.

## Notes / Simplifications

- City indices are scenario levers, not official city statistics.
- The model is not an anti-fraud bypass guide. It only models categories and frequencies.
- Real calibration needs pilot data (actual dispute rate, refund rate, fraud incident rate, fee structure, MoR).
