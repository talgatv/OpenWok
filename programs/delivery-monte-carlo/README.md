# Delivery Monte Carlo Simulator

Monte Carlo simulator for delivery business economics with an optional city risk model.

## Files
- `simulate.py` - simulation engine + reporting bundle (`json`, `md`, `csv`).
- `config.json` - core simulation config with city profiles.
- `analyze_report.py` - second-stage analyzer with forecasts and recommendations.
- `analysis_config.json` - scenarios and thresholds for analyzer.
- `CITY_PARAMETERS_GUIDE.md` - practical guide for city risk/fraud parameter tuning.
- `requirements.txt` - dependency list (standard library only).

## Run simulation
```bash
cd /Users/Talgat2/Documents/Projects/OpenWok/OpenWok/programs/delivery-monte-carlo
python3 simulate.py --config config.json
```

Default output files:
- `reports/simulation_report.json`
- `reports/simulation_report.md`
- `reports/simulation_report.csv`

Custom names:
```bash
python3 simulate.py --config config.json --report-dir reports --report-name simulation_q1
```

## Run analyzer (second program)
```bash
python3 analyze_report.py \
  --input-report reports/simulation_report.json \
  --analysis-config analysis_config.json \
  --output-dir reports \
  --output-name analysis_report
```

Analyzer output files:
- `reports/analysis_report.json`
- `reports/analysis_report.md`
- `reports/analysis_report.csv`

## What simulation calculates
- Funnel: attempts, successful payments, cancellations, delivered orders.
- Risk incidents: refunds, disputes, fraud events, late deliveries.
- Revenue: platform base fee and Protection Plan revenue.
- Costs: payment processor, infra, support, manual reviews, insurance premium.
- Losses: refund + dispute losses (with insurance offsets).
- Final economics: net profit and cash after reserve holds.
- City-level breakdown: risk ranking and unit economics per city profile.

## City model (new)
`config.json` has `city_model.enabled=true` and city profiles.
Each profile uses synthetic indices in `[0..1]` and `population_millions`.

Main city parameters:
- `crime_index` - raises dispute/fraud multipliers.
- `economic_stress_index` - increases fraud/refund/cancel pressure.
- `digital_fraud_pressure_index` - raises online abuse intensity.
- `hotspot_inequality_index` - models risky neighborhood concentration.
- `anti_fraud_maturity_index` - reduces fraud/dispute multipliers.
- `density_index` + `traffic_congestion_index` - increase delays and support load.
- `income_index` - increases average ticket size and payment stability.
- `courier_reliability_index` - reduces delays and service mistakes.
- `tourism_index` - raises volatility in demand and ticket size.
- `population_millions` + `market_share` - shape order volume split.

The simulator transforms these indices into probability multipliers for:
- `payment_success_rate`
- `cancel_rate_after_payment`
- `refund_rate_after_delivery`
- `dispute_rate_after_delivery`
- `fraud_incident_rate`
- `late_delivery_rate`
- support/manual review intensity
- reserve hold pressure

## Important modeling note
City indices are scenario parameters, not official statistics.
Use them for stress testing and strategy comparisons, then calibrate with pilot data.

## Tuning strategy for uncertain fraud frequency
1. Start with conservative baseline fraud/dispute rates.
2. Tune city indices and check report sensitivity.
3. Compare three runs: optimistic/base/stress.
4. Lock thresholds in `analysis_config.json` for alerts.
5. Recalibrate monthly with real incidents.
