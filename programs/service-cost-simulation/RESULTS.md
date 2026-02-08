# Service Cost Simulation — Выводы

Источник данных: `reports/service_cost_report.json` (генерация: 2026-02-08T03:06:52.097849+00:00).

## Сводка по сценариям

| Scenario | Cost/Order Mean | Best Model | Best Model Mean Profit | Fixed Break-even | Percent Break-even |
| --- | --- | --- | --- | --- | --- |
| budget | $2.68 | fixed_fee_high | $6663.19 | $2.68 | 10.69% |
| standard | $3.27 | fixed_fee_high | $-48365.89 | $3.27 | 13.04% |
| premium | $4.18 | fixed_fee_high | $-133377.94 | $4.18 | 16.67% |

## Основные драйверы затрат (mean)

**budget**
- payment_fees_usd: $86878.66
- map_cost_usd: $54895.31
- support_cost_usd: $15357.95

**standard**
- map_cost_usd: $100216.65
- payment_fees_usd: $96147.33
- support_cost_usd: $14850.06

**premium**
- map_cost_usd: $177544.30
- payment_fees_usd: $100735.08
- support_cost_usd: $16695.76

## Рекомендации по тарифам (безубыточность)

**budget**
- Fixed fee: $2.68
- Percent: 10.69%
- Hybrid (percent 3.50%): fixed $1.80
- Hybrid (fixed $1.40): percent 5.10%
- Subscription (per-order $0.60): monthly $332.57
- Subscription (monthly $280.00): per-order $0.93

**standard**
- Fixed fee: $3.27
- Percent: 13.04%
- Hybrid (percent 3.50%): fixed $2.39
- Hybrid (fixed $1.40): percent 7.45%
- Subscription (per-order $0.60): monthly $426.14
- Subscription (monthly $280.00): per-order $1.51

**premium**
- Fixed fee: $4.18
- Percent: 16.67%
- Hybrid (percent 3.50%): fixed $3.30
- Hybrid (fixed $1.40): percent 11.08%
- Subscription (per-order $0.60): monthly $570.86
- Subscription (monthly $280.00): per-order $2.42

## Что означает результат

- В текущей конфигурации затраты на сервис в среднем выше базовых тарифов, особенно в standard и premium сценариях.
- Наиболее затратные статьи во всех сценариях: карты и комиссии платежного провайдера.
- Если сохранять UX с высокой частотой трекинга, то процент/фикс должны быть заметно выше текущих (см. break-even).

## Следующие шаги

1. Подставить реальные цены провайдеров карт/SMS/ML и платежного процессинга.
2. Уточнить допустимую частоту трекинга и лимиты обновлений карты.
3. Зафиксировать целевую маржу (например, 10–20%) и пересчитать тарифы.