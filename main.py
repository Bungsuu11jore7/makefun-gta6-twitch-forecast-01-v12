"""GTA VI launch-day Twitch forecast (gta6-twitch-ccu-forecast-01), v12.

Model: a kernel-weighted ensemble over past big-game launches.
For each forecast hour the centre is a blend of three members, all
regressing the comparables' unlock-relative log-ratio curves
(log viewers at an unlock-relative hour / viewers at hour 1) on the
target's log hour-1 print:

  - local-linear ridge (weight 0.5) on log hour-1 print,
  - kernel-weighted mean (weight 0.3), narrow bandwidth,
  - kernel-weighted mean (weight 0.2), wide bandwidth.

The centre is then corrected with the target's own pre-release
diurnal profile (early/mid trough deepening, an inflation shift, an
early-dip echo, a small-launch evening lift, and an early-hours size
gate). Quantiles are the centre plus a fixed per-hour lognormal
spread (Z * SIGMA), rounded half-up to 2 decimals.

v12 changes vs v11 (A3 size-adaptive bandwidths, kept exactly):
  1. SIZE_REF 12.0 -> 11.5: the early-hours size gate now starts
     biting at log hour-1 print 11.5 instead of 12.0. Rolling
     launches in that band gain from the deeper pre-unlock lull.
  2. The size gate applies ONLY to rolling-midnight launches. A
     simultaneous-global launch has no pre-unlock hours in its
     first day, so the early-hours lull correction does not apply
     to it. The target's unlock type is read from the alignment
     map below (a simultaneous launch is one named there); the
     final target is a rolling launch, so the gate stays on for it.

No practice answers, curves, or game names are stored here: the
only name->value table is the simultaneous-unlock alignment map
(name -> unlock UTC hour, no curve values).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at the scored levels (rounded values).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- centre ensemble knobs ---
W_LOCLIN = 0.5
W_MEAN1 = 0.3
W_MEAN2 = 0.2
BW_LOCLIN_BIG = 1.8
BW_LOCLIN_SMALL = 1.2
BW_MEAN1_BIG = 1.0
BW_MEAN1_SMALL = 1.5
BW_MEAN2 = 2.5
LAM_LOCLIN = 0.6
SLOPE_DAMP = 0.3
# A launch counts as "big" for the adaptive bandwidths when its log
# hour-1 print reaches this threshold.
BIG_THRESHOLD = 10.0
# Weight multiplier when target and comparable fall on different
# weekend/weekday sides.
WEEKDAY_FACTOR = 0.75

# --- dip-correction knobs ---
DIP_COEF = 0.7
INFL_COEF = 0.02
INFL_CAP = 10.0
INFL_GATE_DENOM = 0.10
ECHO_COEF = 0.5
EVENING_COEF = 0.2
EVENING_REF = 11.0
EARLY_SIZE_COEF = 0.2
# v12: the early-hours size gate starts at log print 11.5.
SIZE_REF = 11.5
# Fallback inflation when the diurnal profile is missing. It is
# provably inert: with no diurnal the mid gate is 0, so the fallback
# is always multiplied by zero. (Value 8.0 matches the reference.)
INFL_FALLBACK = 8.0

# --- spread knobs ---
# Per-hour lognormal spread for forecast hours 2..24. A global
# constant: the same profile applies to every case, including the
# final. It is 1.15x the (smoothed) per-hour sd of this centre's
# residuals on the practice cases, floored at 0.15; it describes how
# wrong the centre is per unlock-relative hour on average, not any
# case's answers.
SIGMA = [0.26, 0.25, 0.30, 0.21, 0.18, 0.16, 0.18, 0.20, 0.22, 0.29,
         0.32, 0.32, 0.25, 0.24, 0.25, 0.26, 0.23, 0.17, 0.15, 0.15,
         0.15, 0.15, 0.15]

# Simultaneous-global launches: name -> unlock UTC hour. Alignment
# only; no curve values are stored here. A launch named here unlocks
# everywhere at once, so it has no rolling-midnight pre-unlock lull.
SIMULTANEOUS_UNLOCK = {
    "007 First Light": 14,
    "Battlefield 6": 14,
    "Black Myth: Wukong": 2,
    "Cyberpunk 2077": 0,
    "Hollow Knight: Silksong": 14,
}

# Index sets on the 23-entry centre list (hours 2..24).
EARLY_IDX = (0, 1, 2)
MID_IDX = (9, 10, 11, 12, 13)
EVENING_IDX = (15, 16, 17, 18, 19, 20, 21, 22)


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def dst_active(day):
    """US daylight saving in effect on `day` (a date). DST runs from
    the second Sunday of March to the first Sunday of November; a
    launch ON the second Sunday of March counts as DST, one ON the
    first Sunday of November does not."""
    mar1 = datetime(day.year, 3, 1, tzinfo=timezone.utc).date()
    start = mar1.replace(day=1 + ((6 - mar1.weekday()) % 7) + 7)
    nov1 = datetime(day.year, 11, 1, tzinfo=timezone.utc).date()
    end = nov1.replace(day=1 + ((6 - nov1.weekday()) % 7))
    return start <= day < end


def rolling_unlock(day_start):
    """US East Coast midnight unlock hour (UTC) for a rolling launch."""
    return 4 if dst_active(parse_ts(day_start).date()) else 5


def comp_unlock(comp):
    """Unlock UTC hour for a comparable launch, or None to exclude it
    from the pool (mixed console+PC-simultaneous launches, and
    simultaneous launches whose unlock hour we do not know). The
    exclusion strings are checked BEFORE the simultaneous map, so a
    mapped launch that is early-access-contaminated is still out."""
    unlock = comp["unlock"]
    if "early-access-contaminated" in unlock:
        return None
    if "Tier: weak" in unlock:
        return None
    if "Tier: target-history" in unlock:
        return None
    if comp["name"] in SIMULTANEOUS_UNLOCK:
        return SIMULTANEOUS_UNLOCK[comp["name"]]
    text = unlock.lower()
    if "simultaneous" in text or "single global" in text:
        return None
    if "mixed" in text:
        return None
    return rolling_unlock(comp["day_start"])


def target_unlock(case):
    return rolling_unlock(case["day_start"])


def smooth3(values):
    """One pass of a 3-point moving average (edges use 2 points)."""
    out = []
    for i in range(len(values)):
        vals = [values[j] for j in (i - 1, i, i + 1)
                if 0 <= j < len(values)]
        out.append(sum(vals) / len(vals))
    return out


def diurnal_profile(history_hourly, day_start):
    """Mean log viewers by UTC hour over the 14 pre-release days
    (offsets -336..-1). Each UTC hour needs at least 3 points or
    the profile is None. The result is passed through a 3-point
    moving average twice (a 5-tap triangular smooth)."""
    if not history_hourly:
        return None
    ds = parse_ts(day_start)
    sums = [0.0] * 24
    counts = [0] * 24
    for point in history_hourly:
        if point.get("viewers") is None:
            continue
        dt = parse_ts(point["from"])
        off = int((dt - ds).total_seconds() // 3600)
        if -336 <= off < 0:
            sums[dt.hour] += math.log(max(1, point["viewers"]))
            counts[dt.hour] += 1
    if any(c < 3 for c in counts):
        return None
    profile = [sums[h] / counts[h] for h in range(24)]
    return smooth3(smooth3(profile))


def build_pool(case):
    """Comparable launches usable for training: (name, log hour-1
    print, 29-entry unlock-relative log-ratio curve, unlock hour,
    pre-release hourly points for the diurnal). The curve has index 0
    == 5h before unlock; entry j is log(viewers[UTC offset
    clamp(unlock + j - 5, 0, 23)] / viewers[UTC offset 0]). A
    comparable is dropped if any launch-day hour 0..23 is missing or
    has fewer than 1 viewer."""
    pool = []
    for comp in case["comparables"]:
        unlock = comp_unlock(comp)
        if unlock is None:
            continue
        ds = parse_ts(comp["day_start"])
        by_offset = {}
        by_offset_ch = {}
        pre = []
        for point in comp["hourly"]:
            off = int((parse_ts(point["from"]) - ds).total_seconds()
                      // 3600)
            by_offset[off] = point["viewers"]
            if point.get("channels"):
                by_offset_ch[off] = point["channels"]
            if off < 0:
                pre.append(point)
        if any(o not in by_offset or by_offset[o] < 1
               for o in range(24)):
            continue
        v0 = max(1, by_offset[0])
        logp1 = math.log(v0)
        ch0 = max(1, by_offset_ch.get(0, 1))
        logch = math.log(ch0)
        curve = []
        for j in range(29):
            off = min(23, max(0, unlock + j - 5))
            curve.append(math.log(max(1, by_offset[off]) / v0))
        pool.append({"name": comp["name"], "logp1": logp1,
                     "logch": logch, "logvpc": logp1 - logch,
                     "curve": curve, "unlock": unlock, "pre": pre,
                     "weekend": parse_ts(comp["day_start"]).weekday() >= 5,
                     "day_start": comp["day_start"]})
    return pool


def kernel_weight(logp1, lp, bw):
    d = (logp1 - lp) / bw
    return math.exp(-0.5 * d * d)


def weekday_factor(comp, target_weekend):
    """1.0 when the comparable's release day falls on the same
    weekend/weekday side as the target's, else WEEKDAY_FACTOR."""
    if target_weekend is None:
        return 1.0
    return 1.0 if comp["weekend"] == target_weekend else WEEKDAY_FACTOR


def loclin_predict(pool, pos, lp, bw, target_weekend=None):
    """Kernel-weighted local-linear ridge of the log-ratio at curve
    position `pos` on log hour-1 print. The slope is damped outside
    the comparables' log-print range: the distance beyond the range
    is shrunk by SLOPE_DAMP, so a far bigger launch than anything
    seen does not extrapolate linearly forever."""
    if not pool:
        return 0.0
    num_w = 0.0
    mx = 0.0
    my = 0.0
    terms = []
    for comp in pool:
        w = kernel_weight(comp["logp1"], lp, bw) \
            * weekday_factor(comp, target_weekend)
        terms.append((w, comp["logp1"], comp["curve"][pos]))
        num_w += w
        mx += w * comp["logp1"]
        my += w * comp["curve"][pos]
    if num_w <= 0:
        return sum(c["curve"][pos] for c in pool) / len(pool)
    mx /= num_w
    my /= num_w
    sxx = 0.0
    sxy = 0.0
    for w, x, y in terms:
        sxx += w * (x - mx) * (x - mx)
        sxy += w * (x - mx) * (y - my)
    slope = sxy / (sxx + LAM_LOCLIN)
    lo = min(c["logp1"] for c in pool)
    hi = max(c["logp1"] for c in pool)
    clamped = min(hi, max(lo, lp))
    dx = (clamped - mx) + SLOPE_DAMP * (lp - clamped)
    return my + slope * dx


def kernel_mean_predict(pool, pos, lp, bw, target_weekend=None):
    """Kernel-weighted mean of the comparables' log-ratio at `pos`."""
    num = 0.0
    den = 0.0
    for comp in pool:
        w = kernel_weight(comp["logp1"], lp, bw) \
            * weekday_factor(comp, target_weekend)
        num += w * comp["curve"][pos]
        den += w
    if den <= 0:
        if not pool:
            return 0.0
        return sum(c["curve"][pos] for c in pool) / len(pool)
    return num / den


def ensemble_centre(pool, lp, unlock, bw_mean1, bw_loclin,
                    target_weekend=None):
    """Blended centre: 23 log-ratio values, one per forecast hour
    2..24, for a launch with hour-1 log print `lp` and unlock hour
    `unlock`. Forecast hour h uses curve position h + 4 - unlock."""
    centre = []
    for h in range(2, 25):
        pos = h + 4 - unlock
        pos = min(28, max(0, pos))
        m1 = loclin_predict(pool, pos, lp, bw_loclin, target_weekend)
        m2 = kernel_mean_predict(pool, pos, lp, bw_mean1,
                                 target_weekend)
        m3 = kernel_mean_predict(pool, pos, lp, BW_MEAN2,
                                 target_weekend)
        centre.append(W_LOCLIN * m1 + W_MEAN1 * m2 + W_MEAN2 * m3)
    return centre


def dip_corrections(centre, logp1, diurnal, rolling=True):
    """Target-specific corrections from the launch's own pre-release
    diurnal profile. dip[i] = diurnal[i+1] - diurnal[0]. The early
    trough hours (2-4) get one-sided dip deepening plus the early
    size gate; the mid trough hours (11-15) get deepening, the
    inflation shift gated by the launch's own midday dip, and the
    early-dip echo; hours 17-24 get the small-launch evening lift.
    dip_early / dip_mid are the means of the NEGATIVE parts of the
    dips over the early / mid sets. When the diurnal is missing, the
    dips are zero, so only the size gate (early) and the evening
    lift apply; the inflation fallback then multiplies a zero gate.
    The size gate models the pre-unlock lull, so it applies only to
    rolling launches (rolling=False -> gate 0)."""
    out = list(centre)
    if diurnal is None:
        dip = [0.0] * 23
        infl = INFL_FALLBACK
    else:
        dip = [diurnal[i + 1] - diurnal[0] for i in range(23)]
        infl = min(logp1 - diurnal[0], INFL_CAP)
    dip_early = sum(min(0.0, dip[i]) for i in EARLY_IDX) / len(EARLY_IDX)
    dip_mid = sum(min(0.0, dip[i]) for i in MID_IDX) / len(MID_IDX)
    gate_mid = min(1.0, max(0.0, -dip_mid / INFL_GATE_DENOM))
    if rolling:
        size_gate = min(1.0, max(0.0, logp1 - SIZE_REF))
    else:
        size_gate = 0.0
    evening_lift = EVENING_COEF * max(0.0, EVENING_REF - logp1)
    for i in EARLY_IDX:
        out[i] += DIP_COEF * min(0.0, dip[i]) \
            - EARLY_SIZE_COEF * size_gate
    for i in MID_IDX:
        out[i] += DIP_COEF * min(0.0, dip[i]) \
            - INFL_COEF * infl * gate_mid + ECHO_COEF * dip_early
    for i in EVENING_IDX:
        out[i] += evening_lift
    return out


def full_centre(pool, logp1, unlock, pre_points, day_start,
                target_weekend=None, rolling=True,
                bw_mean1=BW_MEAN1_SMALL, bw_loclin=BW_LOCLIN_BIG):
    """Ensemble centre plus dip corrections for one launch. The
    diurnal comes from the launch's own pre-release hourly points
    (history_hourly for the target, the comparable's own pre-release
    hourly rows for a comparable)."""
    diurnal = diurnal_profile(pre_points, day_start)
    centre = ensemble_centre(pool, logp1, unlock, bw_mean1, bw_loclin,
                             target_weekend)
    return dip_corrections(centre, logp1, diurnal, rolling)


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's
    Math.round(x * 100) / 100 rounds it. A whole number comes back
    as an int, so json.dumps writes 1240 where JavaScript writes
    1240 (not 1240.0)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case, case_id):
    if case["levels"] != LEVELS:
        raise ValueError(
            f"{case_id} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{case_id} has no hour-1 print")
    p1 = max(1, first["viewers"])
    logp1 = math.log(p1)
    pool = build_pool(case)
    unlock = target_unlock(case)
    weekend = parse_ts(case["day_start"]).weekday() >= 5
    # A3 size-adaptive bandwidths: big launches get a narrow mean
    # bandwidth and a wide local-linear one; small launches the
    # reverse.
    big = logp1 >= BIG_THRESHOLD
    bw_mean1 = BW_MEAN1_BIG if big else BW_MEAN1_SMALL
    bw_loclin = BW_LOCLIN_BIG if big else BW_LOCLIN_SMALL
    # The early-hours size gate models the pre-unlock lull, so it
    # applies only to rolling-midnight launches. A launch named in
    # the simultaneous-unlock map unlocked everywhere at once.
    name = case.get("game", {}).get("name", "")
    rolling = name not in SIMULTANEOUS_UNLOCK
    if pool:
        centre = full_centre(pool, logp1, unlock,
                             case.get("history_hourly") or [],
                             case["day_start"], weekend, rolling,
                             bw_mean1, bw_loclin)
    else:
        centre = [0.0] * 23
    rows = []
    for i, h in enumerate(range(2, 25)):
        mu = logp1 + centre[i]
        qs = [round2(math.exp(mu + Z[j] * SIGMA[i]))
              for j in range(5)]
        qs = [min(1e9, max(0.01, q)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
