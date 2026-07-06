# Post-Gate Capture Economics Decomposition

- Generated: `2026-07-06T15:11:03Z`
- Capture economics status: `NEGATIVE_EXPECTANCY`
- NEGATIVE_EXPECTANCY_ACTIVE should remain: `True`

| scope | trades | wins | losses | net | exp/trade | PF | avg win | avg loss | max loss | clears negative? | live permission | proof pack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `raw_realized_system_ledger` | 234 | 137 | 97 | -41225.88 | -176.179 | 0.5814 | 417.9691 | 1015.3366 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_ledger` | 234 | 137 | 97 | -41225.88 | -176.179 | 0.5814 | 417.9691 | 1015.3366 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_market_close_leak_family_blocked_excluded` | 227 | 137 | 90 | -26134.2135 | -115.1287 | 0.6866 | 417.9691 | 926.622 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_residual_family_blocked_excluded` | 208 | 137 | 71 | -16895.1526 | -81.2267 | 0.7722 | 417.9691 | 1044.4637 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_both_market_close_leak_and_residual_family_filters` | 201 | 137 | 64 | -1803.4861 | -8.9726 | 0.9695 | 417.9691 | 922.8946 | -11986.9276 | `False` | `False` | `False` |
| `manual_excluded_plus_existing_filters_plus_new_family_containment` | 154 | 132 | 22 | 50899.9535 | 330.5192 | 10.3222 | 426.9699 | 248.185 | -981.7942 | `False` | `False` | `False` |
| `attached_tp_only_harvest_segments` | 96 | 96 | 0 | 48804.8389 | 508.3837 | None | 508.3837 | None | None | `False` | `False` | `True` |
| `market_order_trade_close_segments` | 98 | 24 | 74 | -74151.7656 | -756.6507 | 0.0672 | 222.6069 | 1074.2477 | -11986.9276 | `False` | `False` | `False` |
| `unknown_unverified_gateway_close_segments` | 35 | 1 | 34 | -53835.8033 | -1538.1658 | 0.0006 | 30.1535 | 1584.2928 | -11986.9276 | `False` | `False` | `False` |

## Loss Contributors

- `raw_realized_system_ledger`: 470854 (-11986.9276), 470730 (-8378.521), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422)
- `manual_excluded_ledger`: 470854 (-11986.9276), 470730 (-8378.521), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422)
- `manual_excluded_plus_market_close_leak_family_blocked_excluded`: 470854 (-11986.9276), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
- `manual_excluded_plus_residual_family_blocked_excluded`: 470854 (-11986.9276), 470730 (-8378.521), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422)
- `manual_excluded_plus_both_market_close_leak_and_residual_family_filters`: 470854 (-11986.9276), 471020 (-3947.4), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
- `manual_excluded_plus_existing_filters_plus_new_family_containment`: 472222 (-981.7942), 470945 (-672.0), 470992 (-546.0), 471000 (-454.9862), 470951 (-371.194)
- `market_order_trade_close_segments`: 470854 (-11986.9276), 470730 (-8378.521), 471008 (-3688.6643), 471232 (-3307.422), 470898 (-3171.0)
- `unknown_unverified_gateway_close_segments`: 470854 (-11986.9276), 470730 (-8378.521), 471232 (-3307.422), 470898 (-3171.0), 472071 (-2981.8961)
