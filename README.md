# Leonida Week-One Rarity Forecaster (Python)

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Apip21/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Apip21/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of **Grand Theft Auto VI** on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — one week after launch — as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

Built for the make.fun bounty **gta6-trophy-rarity-forecast-01** ($250, closes 2026-11-21 11:30 UTC). The program reads one case file (`/input/<id>.in`: every valid read the house made of the game's Exophase trophy page before the case's checkpoint) and writes one forecast (`/output/<id>.out`).

## How it forecasts

Per trophy, on the logit scale, the median blends two forces with a horizon-dependent weight `w = days / (days + 0.75)`:

- **Refresh dilution** (weight `1 - w`): Exophase refreshes its counts on its own schedule, and each refresh pulls in newly tracked players who have earned almost nothing yet, so rates fall. The expected player multiplier to the target combines steady growth (a daily rate read off the history when it grew, else a 45%/day launch-week prior) with one pending-refresh jump — the launch-night refresh multiplied tracked players 638 → 3,868 (×6.06) after ~4.7 h of staleness, so an overdue refresh scales the jump by how stale the last read is (measured to the checkpoint). Old holders are mixed with newcomers (who hold each trophy at ~0.1%, except a trophy already at ~100%, which newcomers hold at once) to give the diluted rate.
- **Progression** (weight `w`): over the horizon players earn trophies. Each trophy gets a daily logit velocity: a tier prior (platinum 0.02, gold 0.10, silver 0.20, bronze 0.35 logit/day, scaled up when the trophy is already common) blended with the trophy's own observed velocity across the history, shrunk by how far the trophy actually moved. Flat histories fall back to the prior alone. The rise saturates at 99.5%.
- **Spread**: a random-walk term (0.5 logit per √day) combined in quadrature with a refresh-uncertainty term (large while dilution dominates) and a progression-uncertainty term (grows with the horizon), floored at 0.55 and capped at 2.5 logit; the near-100% trophy gets a wider floor because it can only move down.

A 90-minute horizon is almost all dilution — the practice cases straddle Exophase's first refresh, so the forecast expects the rates to fall. A 6.5-day horizon is mostly progression — easy story trophies rise while rare ones stay low.

Python 3.13, standard library only, no network, deterministic: two runs on the same case write identical bytes.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/p00.out and out/p01.out for the practice cases
```

## Local validation (practice cases p00/p01)

- **Shape: PASS.** 55 rows each, ids once, five non-decreasing quantiles in 0–100, outputs 2,795–2,796 bytes (cap 65,536).
- **Determinism: PASS.** Two runs byte-identical; ~52–66 ms per run (cap 8,000 ms).
- **Starter reproduction: PASS.** The kit starter run on the same cases gives trophy 4 exactly `[0.8974, 1.012, 1.1, 1.1955, 1.3477]`, the row the rules document.
- **Pinball proxy (assumptions, not facts — the kit holds no answers):** against a crude refresh proxy (new rate = old rate × 638/3868, 100% trophy held at 100%), this entry scores **0.374 / 0.358** (p00/p01) vs the starter's **1.688 / 1.688**; against an aggregate-grounded proxy (non-100% trophies × 0.2206, the factor reproducing the published post-refresh totals of 3,868 players / 4,276 trophies earned exactly), **0.309 / 0.300** vs the starter's **1.402 / 1.402**. Typical miss 0.15/0.13 pp (proxy A) vs the starter's 0.73 pp; 90% coverage 0.98 vs the starter's 0.00. The scorer itself was verified against the rules' worked example (loss 0.403404, reproduced exactly).
- **True practice loss is not computable locally** — the .in files hold no answers; the published house-baseline practice loss is ~1.51.

## Caveats

- The dilution jump (×6.06) and the 45%/day growth prior come from a single observed refresh on launch night; later refreshes may be much smaller, and the final's real player growth is unknown.
- The tier velocity priors are judgement calls, not fitted values — the practice histories are flat, so nothing in them can fit a progression rate.
- The final case's history (two days of 10-minute reads) will show real movement; the model's observed-velocity term is what adapts to it, and it is shrunk toward the tier prior, so a trophy that surged for a day is not extrapolated at full speed.
- The 100% trophy is a special case: it is held near 100% with a wide downward spread, on the launch-night evidence that it did not fall in the refresh.
