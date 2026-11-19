"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its
Exophase earn rate at the case's target, from the house's recorded reads up
to the case's checkpoint. Python 3.13, standard library only, no network,
fully deterministic.

Model (per trophy, on the logit scale; see README.md for the reasoning):

  * Start from the trophy's last observed rate r (logit L).
  * Refresh dilution: Exophase refreshes its counts on its own schedule and
    each refresh pulls in newly tracked players who have earned almost
    nothing yet, so rates fall. The expected player multiplier M to the
    target combines steady growth (a daily rate read off the history when
    it grew, else a 45%/day launch-week prior) with one pending-refresh
    jump: the launch-night refresh multiplied tracked players 638 -> 3868
    (x6.06) after ~4.7 h of staleness, so a refresh that is overdue scales
    the jump by how stale the last read is. Mixing the old holders with
    the newcomers (who hold each trophy at ~0.1%, except a trophy already
    at ~100%, which newcomers hold at once) gives the diluted rate.
  * Progression: over the horizon players earn trophies. Each trophy gets
    a daily logit velocity: a tier prior (platinum slowest, bronze story
    trophies fastest, scaled up when the trophy is already common) blended
    with the trophy's own observed velocity across the history, shrunk by
    how far the trophy actually moved. Flat histories fall back to the
    prior alone.
  * Horizon blend: the median logit is L plus the dilution shift weighted
    by 1 - w and the progression shift weighted by w, where
    w = days / (days + 0.75). A 90-minute horizon is almost all dilution
    (the practice cases straddle a refresh); a 6.5-day horizon is mostly
    progression.
  * Spread: a random-walk term (0.5 logit per root-day) combined in
    quadrature with a refresh-uncertainty term (large while dilution
    dominates) and a progression-uncertainty term (grows with the
    horizon), floored and capped; the near-100% trophy gets a wider floor
    because it can only move down.

IO: reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes
/output/<id>.out = {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
per case. Two runs on the same case write identical bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch of GTA VI (start of the release day, UTC); log-time origin.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- model constants (priors, fixed before seeing the practice answers) ---
SIGMA_RW = 0.5          # random-walk logit sd per sqrt(day)
SIGMA_REFRESH = 0.9     # refresh-uncertainty scale (logit)
SIGMA_PROG = 0.30       # progression-uncertainty scale per sqrt(day)
SIGMA_FLOOR = 0.55
SIGMA_CAP = 2.5
SIGMA_FLOOR_TOP = 1.0   # near-100% trophy: it can only move down
HORIZON_BLEND = 0.75    # days; w_prog = d / (d + HORIZON_BLEND)
GROWTH_PRIOR = 0.45     # daily tracked-player growth when history is flat
FIRST_REFRESH_X = 6.06  # 3868 / 638, the launch-night refresh's jump
FIRST_REFRESH_STALE_H = 4.7  # hours of staleness before that refresh
NEWCOMER_RATE = 0.001   # fraction of newcomers holding a typical trophy
SHRINK_RANGE = 0.5      # logit range at which observed velocity gets w=0.5

# Daily logit velocity prior per tier (week one: story bronze move fastest).
TIER_VELOCITY = {
    "platinum": 0.02,
    "gold": 0.10,
    "silver": 0.20,
    "bronze": 0.35,
}
# Fallback last rate (percent) per tier when a case carries no history.
TIER_FALLBACK_RATE = {
    "platinum": 0.5,
    "gold": 0.8,
    "silver": 1.0,
    "bronze": 1.2,
}


def clamp_pct(x):
    """Percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    """Logit back to a percent, guarded against overflow."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg)) if lg < 700 else 100.0
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p):
    """Standard-normal quantile (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def parse_time(text):
    """ISO-8601 UTC timestamp (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round4_half_up(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    value = math.floor(scaled + 0.5) / 10000.0
    return int(value) if float(value).is_integer() else value


def last_valid_rate(history, index):
    """A trophy's rate in the newest read that actually lists it."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return rates[index]
    return None


def first_valid_rate(history, index):
    for read in history:
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return rates[index]
    return None


def analyse_history(history, checkpoint_at=None):
    """Player-base facts the dilution term needs, from the reads alone."""
    reads = sorted(history, key=lambda r: (r.get("slot", ""), r.get("at", "")))
    if not reads:
        return {"growth": GROWTH_PRIOR, "stale_h": 0.0, "span_days": 0.0}
    first_players = reads[0].get("players_tracked")
    last_players = reads[-1].get("players_tracked")
    t0 = parse_time(reads[0]["at"])
    t1 = parse_time(reads[-1]["at"])
    span_days = max((t1 - t0).total_seconds() / 86400.0, 0.0)
    growth = GROWTH_PRIOR
    if first_players and last_players and last_players > first_players and span_days > 0.05:
        daily = (last_players / first_players) ** (1.0 / span_days) - 1.0
        growth = min(max(daily, 0.0), 2.0)
    # Hours since the tracked-player count last changed, measured to the
    # checkpoint (the forecast is made then, and the page may have sat
    # unchanged since the last read too). Never changed -> since the
    # first read.
    stale_h = 0.0
    last_count = None
    last_change_at = t0
    for read in reads:
        count = read.get("players_tracked")
        if count is not None and count != last_count:
            last_count = count
            last_change_at = parse_time(read["at"])
    end = parse_time(checkpoint_at) if checkpoint_at else t1
    stale_h = (end - last_change_at).total_seconds() / 3600.0
    return {"growth": growth, "stale_h": stale_h, "span_days": span_days}


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case.get("history") or [],
                     key=lambda r: (r.get("slot", ""), r.get("at", "")))
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zvals = [norm_ppf(level) for level in levels]
    days = case["horizon_minutes"] / 1440.0
    info = analyse_history(history, case.get("at"))

    # Expected tracked-player multiplier by the target: steady growth,
    # times one pending-refresh jump scaled by staleness.
    jump = 1.0 + (FIRST_REFRESH_X - 1.0) * min(1.0, info["stale_h"] / FIRST_REFRESH_STALE_H)
    mult = min((1.0 + info["growth"]) ** days * jump, 200.0)

    w_prog = days / (days + HORIZON_BLEND) if days > 0 else 0.0
    w_dil = 1.0 - w_prog

    sigma_rw = SIGMA_RW * math.sqrt(days)
    sigma_prog = SIGMA_PROG * math.sqrt(days)
    sigma_refresh = SIGMA_REFRESH * w_dil * (1.0 + 0.5 * math.log10(max(mult, 1.0)))

    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        rate = last_valid_rate(history, i)
        if rate is None:
            rate = TIER_FALLBACK_RATE.get(tier, 1.0)
        base_logit = logit_of(rate)

        # --- dilution: mix old holders with newcomers at the target ---
        frac = rate / 100.0
        newcomer = frac if frac >= 0.99 else NEWCOMER_RATE
        diluted_frac = frac / mult + newcomer * (1.0 - 1.0 / mult)
        dil_shift = logit_of(diluted_frac * 100.0) - base_logit

        # --- progression: tier prior blended with the trophy's own move ---
        prior_v = TIER_VELOCITY.get(tier, 0.35)
        prior_v *= 0.6 + 0.4 * min(1.0, rate / 2.0)
        observed_v = 0.0
        shrink_w = 0.0
        first = first_valid_rate(history, i)
        if first is not None and info["span_days"] > 0.05 and first != rate:
            moved = abs(logit_of(rate) - logit_of(first))
            shrink_w = moved / (moved + SHRINK_RANGE)
            observed_v = (logit_of(rate) - logit_of(first)) / info["span_days"]
        velocity = (prior_v + shrink_w * observed_v) / (1.0 + shrink_w)
        prog_shift = velocity * days
        # A median cannot pass 99.5%: saturation caps the rise, but must
        # never drag down a trophy already above 99.5% (its cap is 0).
        prog_shift = min(prog_shift, max(0.0, logit_of(99.5) - base_logit))

        median_logit = base_logit + w_dil * dil_shift + w_prog * prog_shift

        sigma = math.sqrt(sigma_rw ** 2 + sigma_refresh ** 2 + sigma_prog ** 2)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        if frac >= 0.9999:
            sigma = max(sigma, SIGMA_FLOOR_TOP)

        quantiles = []
        prev = None
        for z in zvals:
            q = round4_half_up(min(max(inv_logit_pct(median_logit + z * sigma), 0.01), 99.99))
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
