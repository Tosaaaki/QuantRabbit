# Post-Gate Capture Economics Decomposition

- Generated: `2026-07-05T17:30:13Z`
- Capture economics status: `NEGATIVE_EXPECTANCY`
- NEGATIVE_EXPECTANCY_ACTIVE should remain: `True`

| scope | trades | wins | losses | net | exp/trade | PF | avg win | avg loss | max loss | clears negative? | live permission | proof pack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `raw_realized_system_ledger` | 229 | 137 | 92 | -40616.8595 | -177.3662 | 0.585 | 417.9691 | 1063.8981 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_ledger` | 229 | 137 | 92 | -40616.8595 | -177.3662 | 0.585 | 417.9691 | 1063.8981 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_market_close_leak_family_blocked_excluded` | 222 | 137 | 85 | -25525.193 | -114.9783 | 0.6917 | 417.9691 | 973.9643 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_residual_family_blocked_excluded` | 203 | 137 | 66 | -16286.1321 | -80.2273 | 0.7786 | 417.9691 | 1114.3622 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_both_market_close_leak_and_residual_family_filters` | 196 | 137 | 59 | -1194.4656 | -6.0942 | 0.9796 | 417.9691 | 990.7837 | -11986.9276 | `False` | `False` | `False` |
| `attached_tp_only_harvest_segments` | 96 | 96 | 0 | 48804.8389 | 508.3837 | None | 508.3837 | None | None | `False` | `False` | `True` |
| `market_order_trade_close_segments` | 98 | 24 | 74 | -74151.7656 | -756.6507 | 0.0672 | 222.6069 | 1074.2477 | -11986.9276 | `False` | `False` | `False` |
| `unknown_unverified_gateway_close_segments` | 97 | 24 | 73 | -73982.6065 | -762.7073 | 0.0674 | 222.6069 | 1086.6462 | -11986.9276 | `False` | `False` | `False` |

## Loss Contributors

- `raw_realized_system_ledger`: 470854 (-11986.9276), 470730 (-8378.521), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422)
- `manual_excluded_ledger`: 470854 (-11986.9276), 470730 (-8378.521), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422)
- `manual_excluded_plus_market_close_leak_family_blocked_excluded`: 470854 (-11986.9276), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
- `manual_excluded_plus_residual_family_blocked_excluded`: 470854 (-11986.9276), 470730 (-8378.521), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422)
- `manual_excluded_plus_both_market_close_leak_and_residual_family_filters`: 470854 (-11986.9276), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
- `market_order_trade_close_segments`: 470854 (-11986.9276), 470730 (-8378.521), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
- `unknown_unverified_gateway_close_segments`: 470854 (-11986.9276), 470730 (-8378.521), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
