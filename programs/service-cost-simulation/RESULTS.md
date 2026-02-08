# Service Cost Simulation - Fixed Fee Focus

Source: `reports/service_cost_report.json` (generated at `2026-02-08T03:17:37.280949+00:00`).

## Portfolio Decision

- Recommended tier: `sustainable`
- Recommended single fixed fee for all orders: **$3.60**
- Why: covers mean service cost across mixed scenarios and leaves moderate buffer for team growth.

| Tier | Suggested Fixed Fee | Weighted Net Profit Mean |
| --- | --- | --- |
| self_cost | $3.35 | $4,500.79 |
| sustainable | $3.60 | $27,914.91 |
| growth | $3.95 | $60,694.68 |

## Scenario Cost Reality

| Scenario | Cost per Order (Mean) | Cost per Order (P90) | Fixed Tier Self-cost | Fixed Tier Sustainable | Fixed Tier Growth |
| --- | --- | --- | --- | --- | --- |
| budget | $2.68 | $2.71 | $2.70 | $2.95 | $3.20 |
| standard | $3.27 | $3.30 | $3.30 | $3.55 | $3.90 |
| premium | $4.18 | $4.20 | $4.20 | $4.55 | $4.95 |

## Main Cost Drivers

- `map_cost_usd` and `payment_fees_usd` are the largest cost items in all scenarios.
- `support_cost_usd` is the third largest recurring variable item.
- Fixed overhead per 30-day period is **$92,520.55** and already includes accounting/legal/tax items.

## Accounting and Governance Transparency

Fixed overhead currently included in simulation:
- Monthly: salaries, base infra, app maintenance, compliance, tools, accounting, legal, HR ops.
- Annual (allocated to period): year-end accounting, security audit, company registration, tax filing.

This setup is aligned with an open-book model because fixed overhead and variable service costs are explicit and traceable.

## Practical Pricing Policy (Fixed Only)

- Public base policy: **single fixed fee = $3.60**.
- If operating in low-cost mode only: reduce to `$3.35`.
- If service quality moves to premium profile for long periods: increase toward `$4.20+`.

## What This Means for a "Project for People"

- You can stay honest and independent from order size by using fixed fee only.
- With `$3.60`, model shows self-sustainability plus controlled reinvestment.
- This supports salary stability and gradual hiring without aggressive overpricing.

## Next Steps

1. Decide governance rule for updating fixed fee (for example: quarterly review, open report publication).
2. Add alert thresholds in code for auto-warning when real cost/order approaches fee.
3. Plug real provider invoices (maps, payment, SMS, ML) to tighten confidence intervals.
