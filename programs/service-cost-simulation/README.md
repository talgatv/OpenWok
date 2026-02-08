# Service Cost Simulation

Monte Carlo simulator for estimating platform service cost per order and comparing business models (fixed fee, percent, hybrid, subscription + per-order fee).

## Files
- `simulate.py` - simulation engine and report output.
- `config.json` - all tuning parameters.
- `requirements.txt` - dependency list (standard library only).
- `reports/` - output folder (JSON/MD/CSV).

## Run simulation
```bash
cd /Users/Talgat2/Documents/Projects/OpenWok/OpenWok/programs/service-cost-simulation
python3 simulate.py --config config.json
```

Custom report location:
```bash
python3 simulate.py --config config.json --report-dir reports --report-name service_cost_q1
```

## Outputs
- `reports/service_cost_report.json`
- `reports/service_cost_report.md`
- `reports/service_cost_report.csv`

## What it calculates
- Variable costs per order (maps, API, ML, notifications, support, storage).
- Payment processor fees (optional, configurable).
- Fixed costs allocated per order.
- Cost per order (min/avg/max via Monte Carlo runs).
- Business model revenue and net profit across scenarios (budget/standard/premium).
- Break-even pricing suggestions for fixed/percent/hybrid/subscription models.
- Fixed-fee guidance tiers (`self_cost`, `sustainable`, `growth`) and one portfolio-level fixed recommendation.
- City-level cost per order.

## Main config groups
- `simulation`: number of runs, days, random seed.
- `demand`: weekday multipliers and order volatility.
- `cities`: demand, traffic, complexity, order values, subscription participation.
- `usage`: tracking updates, map/API calls, support rates, notifications, data sizes.
- `usage_multipliers`: scenario-level multipliers for usage and support rates.
- `failure_rates`: retry/escalation rates (adds extra cost).
- `costs`: unit costs for each service.
- `payment_processor`: percent + fixed fee per transaction.
- `fixed_costs`: monthly and annual overhead.
- `business_models`: fixed, percent, hybrid, subscription rules.
- `scenarios`: override blocks for budget/standard/premium cost and usage profiles.
- `fixed_pricing_policy`: rounding, risk buffer, margin tiers, scenario weights, and recommended fixed tier.

## Notes
- Default numbers are placeholders for tuning.
- Use real provider pricing (maps, SMS, ML) for accurate results.
- For faster iteration, reduce `simulation.runs` or `simulation.days` in `config.json`.
