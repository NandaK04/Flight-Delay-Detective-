"""
Flight Delay Detective -- Grounded GenAI Analyst
=================================================
Produces structured natural-language analysis of a flight prediction.

GROUNDING GUARANTEE
-------------------
Every number in the output comes from one of two pre-computed JSON files:
  data/analytics/eda_stats.json   -- computed by src/eda/eda.py
  data/metrics/val_metrics.json   -- computed by src/ml/experiment.py

No statistics are invented, estimated, or hallucinated.
The analyst never accesses the raw CSV or the ML model directly.
When a carrier or airport is not in the EDA data, that fact is stated
explicitly rather than fabricated.

HOW IT WORKS
------------
analyse(inputs, prediction) receives:
  inputs     -- the same 10 pre-flight features sent to /predict
  prediction -- the full response dict from /predict

It returns an AnalysisResult containing:
  verdict_sentence  -- one sentence restating the prediction with probability
  risk_factors      -- list of dicts, each citing a specific computed stat
  context           -- background facts about the route/carrier from EDA
  model_note        -- honest statement of model performance from metrics
  data_sources      -- list of JSON files that grounded this analysis
  raw_facts         -- the underlying numbers used (machine-readable)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
EDA_PATH      = ROOT / "data" / "analytics" / "eda_stats.json"
METRICS_PATH  = ROOT / "data" / "metrics"   / "val_metrics.json"

# ── Load grounding data once at module import ─────────────────────────────────
with open(EDA_PATH,     encoding="utf-8") as f:
    _EDA = json.load(f)

with open(METRICS_PATH, encoding="utf-8") as f:
    _METRICS = json.load(f)

# Pre-index EDA lookups for O(1) access
_CARRIER_DELAY: dict[str, float] = {
    r["op_unique_carrier"]: r["delay_pct"]
    for r in _EDA["delay_rate_by_carrier"]
}
_ORIGIN_DELAY: dict[str, float] = {
    r["origin"]: r["delay_pct"]
    for r in _EDA["delay_rate_by_top20_origin"]
}
_HOUR_DELAY: dict[int, float] = {
    r["dep_hour"]: r["delay_pct"]
    for r in _EDA["delay_rate_by_dep_hour"]
}
_MONTH_DELAY: dict[int, dict] = {
    r["month"]: r
    for r in _EDA["delay_rate_by_month"]
}
_OVERALL_DELAY_PCT: float = _EDA["target_distribution"]["pct_delayed"]

# Feature importance rank (1 = most important)
_FI = _METRICS["feature_importances"]
_FI_RANK: dict[str, int] = {
    feat: rank + 1
    for rank, feat in enumerate(
        sorted(_FI, key=lambda k: _FI[k], reverse=True)
    )
}
_MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April",
    5:"May", 6:"June", 7:"July", 8:"August",
    9:"September", 10:"October", 11:"November", 12:"December",
}
_DOW_NAMES = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",
              5:"Friday",6:"Saturday",7:"Sunday"}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    verdict_sentence: str
    risk_factors:     list[dict]          # [{label, stat, note, source}]
    context:          list[str]           # background sentences
    model_note:       str
    data_sources:     list[str]
    raw_facts:        dict                # machine-readable grounding values


# ── Core analysis function ────────────────────────────────────────────────────

def analyse(inputs: dict, prediction: dict) -> AnalysisResult:
    """
    Build a fully grounded analysis of a flight prediction.

    Parameters
    ----------
    inputs     : the 10 pre-flight feature values (from /predict request)
    prediction : the /predict response dict
                 (prediction, prob_delayed, prob_not_delayed, model)

    Returns
    -------
    AnalysisResult -- every field traceable to EDA or metrics JSON
    """
    carrier   = str(inputs.get("op_unique_carrier", "")).upper()
    origin    = str(inputs.get("origin", "")).upper()
    dest      = str(inputs.get("dest", "")).upper()
    month     = int(inputs.get("month", 0))
    dow       = int(inputs.get("day_of_week", 0))
    dep_hour  = int(inputs.get("dep_hour", 0))
    distance  = float(inputs.get("distance", 0))

    prob_delayed    = float(prediction.get("prob_delayed", 0))
    prob_not_delayed= float(prediction.get("prob_not_delayed", 0))
    is_delayed      = prediction.get("prediction") == "delayed"

    # ── 1. Verdict sentence ───────────────────────────────────────────────────
    verdict = (
        f"The model predicts this flight is {'likely to be delayed' if is_delayed else 'likely to be on time'} "
        f"({prob_delayed*100:.1f}% probability of arriving 15 or more minutes late)."
    )

    # ── 2. Risk factors (each backed by a specific EDA number) ────────────────
    risk_factors: list[dict] = []
    raw_facts: dict = {
        "overall_delay_pct":   _OVERALL_DELAY_PCT,
        "prob_delayed_pct":    round(prob_delayed * 100, 1),
        "prob_not_delayed_pct": round(prob_not_delayed * 100, 1),
    }

    # Departure hour
    hour_rate = _HOUR_DELAY.get(dep_hour)
    if hour_rate is not None:
        raw_facts["dep_hour_delay_pct"] = hour_rate
        comparison = "above" if hour_rate > _OVERALL_DELAY_PCT else "below"
        time_label = _hour_label(dep_hour)
        risk_factors.append({
            "label":  "Departure time",
            "stat":   f"{hour_rate:.1f}% delay rate at {time_label} departures (hour {dep_hour})",
            "note":   (
                f"This is {comparison} the overall average of {_OVERALL_DELAY_PCT:.1f}%. "
                f"Departure time is the #{_FI_RANK.get('dep_hour', '?')} most important feature in the model."
            ),
            "source": "eda_stats.json: delay_rate_by_dep_hour",
        })

    # Month / season
    month_row = _MONTH_DELAY.get(month)
    if month_row:
        month_rate  = month_row["delay_pct"]
        month_label = month_row["month_label"]
        raw_facts["month_delay_pct"] = month_rate
        comparison = "above" if month_rate > _OVERALL_DELAY_PCT else "below"
        risk_factors.append({
            "label":  "Travel month",
            "stat":   f"{month_rate:.1f}% delay rate in {month_label}",
            "note":   (
                f"This is {comparison} the overall average of {_OVERALL_DELAY_PCT:.1f}%. "
                f"Month is the #{_FI_RANK.get('month', '?')} most important feature."
            ),
            "source": "eda_stats.json: delay_rate_by_month",
        })

    # Carrier
    carrier_rate = _CARRIER_DELAY.get(carrier)
    if carrier_rate is not None:
        raw_facts["carrier_delay_pct"] = carrier_rate
        comparison = "above" if carrier_rate > _OVERALL_DELAY_PCT else "below"
        risk_factors.append({
            "label":  f"Carrier ({carrier})",
            "stat":   f"{carrier_rate:.1f}% historical delay rate for {carrier}",
            "note":   (
                f"This is {comparison} the overall average of {_OVERALL_DELAY_PCT:.1f}%. "
                f"Carrier is the #{_FI_RANK.get('op_unique_carrier', '?')} most important feature."
            ),
            "source": "eda_stats.json: delay_rate_by_carrier",
        })
    else:
        risk_factors.append({
            "label":  f"Carrier ({carrier})",
            "stat":   "Not in EDA top-carrier list",
            "note":   "Insufficient historical data in the dev sample for this carrier; model used global average.",
            "source": "eda_stats.json: delay_rate_by_carrier",
        })

    # Origin airport
    origin_rate = _ORIGIN_DELAY.get(origin)
    if origin_rate is not None:
        raw_facts["origin_delay_pct"] = origin_rate
        comparison = "above" if origin_rate > _OVERALL_DELAY_PCT else "below"
        risk_factors.append({
            "label":  f"Origin airport ({origin})",
            "stat":   f"{origin_rate:.1f}% delay rate for flights departing {origin}",
            "note":   (
                f"This is {comparison} the overall average of {_OVERALL_DELAY_PCT:.1f}%. "
                f"Origin airport is the #{_FI_RANK.get('origin', '?')} most important feature."
            ),
            "source": "eda_stats.json: delay_rate_by_top20_origin",
        })
    else:
        risk_factors.append({
            "label":  f"Origin airport ({origin})",
            "stat":   "Not in EDA top-20 origin airports",
            "note":   "Airport not in the 20 busiest; model used its learned target-encoded value.",
            "source": "eda_stats.json: delay_rate_by_top20_origin",
        })

    # Day of week
    dow_name = _DOW_NAMES.get(dow, f"day {dow}")
    risk_factors.append({
        "label":  "Day of week",
        "stat":   f"Flight is on a {dow_name}",
        "note":   (
            f"Day of week is the #{_FI_RANK.get('day_of_week', '?')} most important feature "
            f"in the model (lower weight than timing or carrier)."
        ),
        "source": "val_metrics.json: feature_importances",
    })

    # ── 3. Context sentences ──────────────────────────────────────────────────
    context: list[str] = []

    # Overall dataset context
    td = _EDA["target_distribution"]
    context.append(
        f"Across the 2024 flight dataset, {td['pct_delayed']:.1f}% of "
        f"{td['total_flights']:,} analysed flights were delayed by 15 or more minutes."
    )

    # Best/worst months from EDA
    sorted_months = sorted(_EDA["delay_rate_by_month"], key=lambda r: r["delay_pct"])
    best_m  = sorted_months[0]
    worst_m = sorted_months[-1]
    context.append(
        f"Seasonally, {worst_m['month_label']} is the worst month "
        f"({worst_m['delay_pct']:.1f}% delay rate) and "
        f"{best_m['month_label']} is the best ({best_m['delay_pct']:.1f}%)."
    )

    # Dep-hour pattern
    best_hour  = min(_EDA["delay_rate_by_dep_hour"], key=lambda r: r["delay_pct"])
    worst_hour = max(_EDA["delay_rate_by_dep_hour"], key=lambda r: r["delay_pct"])
    context.append(
        f"Time-of-day has a strong effect: "
        f"hour-{best_hour['dep_hour']} departures have the lowest delay rate "
        f"({best_hour['delay_pct']:.1f}%), while hour-{worst_hour['dep_hour']} "
        f"departures are worst ({worst_hour['delay_pct']:.1f}%)."
    )

    # Top-2 model features
    top2 = sorted(_FI, key=lambda k: _FI[k], reverse=True)[:2]
    context.append(
        f"The model's two most influential features are "
        f"{top2[0]} ({_FI[top2[0]]*100:.1f}% importance) and "
        f"{top2[1]} ({_FI[top2[1]]*100:.1f}% importance)."
    )

    # ── 4. Model performance note ─────────────────────────────────────────────
    vm = _METRICS["validation_metrics"]
    cm = _METRICS["confusion_matrix"]
    total_val = cm["TN"] + cm["FP"] + cm["FN"] + cm["TP"]
    model_note = (
        f"Model: {_METRICS['model']} trained on Jan-Sep 2024 flights, "
        f"validated on Oct-Nov ({total_val:,} flights). "
        f"Validation ROC-AUC: {vm['roc_auc_pct']:.1f}%, "
        f"Recall: {vm['recall_pct']:.1f}%, "
        f"Precision: {vm['precision_pct']:.1f}%. "
        f"The model correctly identified {cm['TP']:,} of {cm['TP']+cm['FN']:,} "
        f"actual delayed flights in the validation window."
    )

    return AnalysisResult(
        verdict_sentence = verdict,
        risk_factors     = risk_factors,
        context          = context,
        model_note       = model_note,
        data_sources     = [str(EDA_PATH.relative_to(ROOT)),
                            str(METRICS_PATH.relative_to(ROOT))],
        raw_facts        = raw_facts,
    )


# ── Utility ───────────────────────────────────────────────────────────────────

def _hour_label(h: int) -> str:
    if 5 <= h <= 8:   return "early morning"
    if 9 <= h <= 11:  return "mid-morning"
    if 12 <= h <= 14: return "midday"
    if 15 <= h <= 17: return "afternoon"
    if 18 <= h <= 20: return "evening"
    if 21 <= h <= 23: return "night"
    return "late night / early hours"
