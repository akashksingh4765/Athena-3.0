from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
"""
Athena 3.0 - Phase 11A - MODEL INTERACTION DISCOVERY ENGINE V2.1

SCOPE STATEMENT (read this before interpreting any output)
----------------------------------------------------------
This is a SYSTEMATIC BOUNDED SEARCH over eight specified pattern families.
It is NOT an exhaustive search over every possible joint state of five
models across ten lags -- that space is combinatorially enormous, and
searching it is precisely how false discoveries are manufactured. The
families below were chosen in advance; the bound is a feature, not a
limitation.

What this engine can tell you:
  - whether specified interaction patterns between five models recur
  - whether they survive a chronological hold-out
  - whether they are stable month to month

What this engine CANNOT tell you:
  - whether any surviving pattern is profitable. Entry timing, the actual
    trade horizon, transaction costs, slippage, trade count and capital
    constraints are deliberately OUT OF SCOPE here (see PHASE BOUNDARY).


CORRECTIONS IN V2.1 (over V2)
-----------------------------
C1. MARKET TARGET LABEL OVERLAP -- the serious one.
    The class label at bar t is itself a forward-looking target spanning
    TARGET_HORIZON_BARS future bars. The label at t+1 therefore covers
    overlapping future bars. Consecutive MARKET observations are
    autocorrelated BY CONSTRUCTION.
    Consequences, now handled explicitly:
      - effective sample size < row count, so Wilson bounds on MARKET
        targets are optimistically tight;
      - an effective-N deflation is applied to MARKET Wilson bounds;
      - MARKET targets carry a STRICTER survival bar than MODEL targets;
      - every MARKET row in the output is flagged overlap_warning=True.
    MODEL targets (is model m correct at t+1) do not have this problem:
    correctness is a point evaluation, not a horizon evaluation.

C2. MULTIPLE-TESTING ACCOUNTING.
    The hold-out helps but does not eliminate multiple testing. Output now
    reports candidates tested PER TARGET (the axis the hold-out does not
    protect) and a crude expected-false-positive count, so the survivor
    count can be read against what chance alone would produce.

C3. HONEST SCOPE DOCUMENTATION. V2's docstring overclaimed "every possible
    variation". Corrected above.

C4. LAG-0 SEPARATED. Same-bar relationships are descriptive associations,
    not predictive conditions. They now carry their own family name
    ("same_bar_association") and are excluded from the robust set, so no
    one can later read a contemporaneous association as predictive.

C5. TRADE ECONOMICS OUT OF SCOPE. See PHASE BOUNDARY note at the bottom.


PHASE BOUNDARY
--------------
"Is there a relationship" and "is it worth trading" must be allowed to fail
independently. This script answers only the first. A pattern that clears
every bar here is a CANDIDATE for economic evaluation in a later phase, not
a signal. Nothing in this output should be wired into a strategy directly.


METHODOLOGY
-----------
  Discovery:    2023 Jan - Sep
  Verification: 2023 Oct - Dec
  Stability:    Oct / Nov / Dec scored separately
  Conditions evaluated at bars <= t. Targets evaluated at t+1.
  Cross-session boundaries excluded.
  Nothing is shuffled; strict chronological order throughout.
"""

import json
import math
import itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(str(_ATHENA_ROOT))
EXPERIMENTS = PROJECT_ROOT / "experiments"

MODEL_FILES = {
    "XGB":  EXPERIMENTS / "xgboost"     / "xgboost_v4_oos_2023_predictions.csv",
    "LGB":  EXPERIMENTS / "lightgbm"    / "lightgbm_v3_oos_2023_predictions.csv",
    "CAT":  EXPERIMENTS / "catboost"    / "catboost_v3_oos_2023_predictions.csv",
    "LSTM": EXPERIMENTS / "lstm"        / "lstm_v2_validation_predictions.csv",
    "TRF":  EXPERIMENTS / "transformer" / "transformer_v1_validation_predictions.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_comparison" / "model_interactions_v2_1_2023"


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["XGB", "LGB", "CAT", "LSTM", "TRF"]

CLASS_CODE = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}
CLASS_NAME = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
CLASSES = [0, 1, 2]

# --- C1: label horizon ---------------------------------------------------
# The number of future 5-min bars the class label looks ahead over.
# SET THIS TO MATCH THE ACTUAL TARGET DEFINITION used when the labels were
# built. If the label is "direction over the next 6 bars (30 min)", this is
# 6. Consecutive labels then share 5 of their 6 future bars.
# If you are unsure, DO NOT guess low -- guessing low makes the statistics
# look better than they are. Verify it against the target-design document.
TARGET_HORIZON_BARS = 6

VERIFY_START_MONTH = 10
STABILITY_MONTHS = [10, 11, 12]

LAGS = (1, 2, 3, 5, 10)          # causal lags
SAME_BAR_LAG = 0                 # C4: descriptive only, reported separately

# Support thresholds
MIN_SUPPORT_DISCOVERY = 30
MIN_SUPPORT_VERIFY = 15
MIN_SUPPORT_STABILITY = 8

# --- C1: separate survival bars by target type ---------------------------
# MODEL targets are point evaluations -> standard bar.
# MARKET targets are overlapping-horizon evaluations -> stricter bar,
# because their effective sample size is materially smaller than their
# row count and their Wilson bounds are correspondingly optimistic.
SURVIVAL_BARS = {
    "MODEL":  {"lift_disc": 0.05, "lift_ver": 0.02, "wilson_edge": 0.00},
    "MARKET": {"lift_disc": 0.10, "lift_ver": 0.05, "wilson_edge": 0.01},
}

CONF_EDGES = [0.0, 0.40, 0.50, 0.60, 0.70, 1.01]
CONF_LABELS = ["c<40", "c40-50", "c50-60", "c60-70", "c70+"]

MARGIN_EDGES = [0.0, 0.05, 0.15, 0.30, 1.01]
MARGIN_LABELS = ["m<05", "m05-15", "m15-30", "m30+"]

MAX_STREAK = 4

# C2: nominal false-positive rate assumed per test, for the crude
# expected-false-positive estimate reported at the end.
ASSUMED_ALPHA = 0.05


# ============================================================
# STATISTICS
# ============================================================

def wilson_lower_bound(successes, n, z=1.96):
    """Wilson score interval lower bound. Conservative for small n."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + (z * z) / n
    centre = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def effective_n(n, target_kind):
    """
    C1: deflate sample size for overlapping-horizon targets.

    With a horizon of H bars, consecutive labels share H-1 future bars.
    The independent-information content of n consecutive observations is
    closer to n/H than to n. This is a deliberately crude adjustment --
    it is not a substitute for a block bootstrap -- but it is far better
    than pretending the observations are independent, which is what
    reporting a raw Wilson bound on an overlapping target does.

    MODEL targets are point evaluations and are not deflated.
    """
    if target_kind == "MARKET" and TARGET_HORIZON_BARS > 1:
        return max(1.0, n / float(TARGET_HORIZON_BARS))
    return float(max(n, 0))


def lift(accuracy, baseline):
    if baseline <= 0:
        return 0.0
    return (accuracy - baseline) / baseline


def target_kind(target_name):
    """MARKET targets predict the next bar's class; MODEL targets predict
    whether a given model will be correct."""
    return "MARKET" if target_name.startswith("Next_") else "MODEL"


# ============================================================
# CATEGORICAL MAJORITY (correct; no arithmetic mean of class codes)
# ============================================================

def categorical_majority(votes):
    """
    Returns (winner_code, votes_for_winner, is_tie). Ties return -1 and are
    reported as their own state rather than resolved.

    The V1 bug this replaces: int(round(mean(codes))) turned a 2-2-1
    DOWN/UP/NEUTRAL split into a fictitious NEUTRAL "majority".
    """
    counts = Counter(votes)
    ranked = counts.most_common()
    best_n = ranked[0][1]
    winners = [c for c, n in ranked if n == best_n]
    if len(winners) > 1:
        return -1, best_n, True
    return winners[0], best_n, False


def majority_series(pred_matrix):
    out = np.empty(len(pred_matrix), dtype=int)
    nvotes = np.empty(len(pred_matrix), dtype=int)
    tie = np.zeros(len(pred_matrix), dtype=bool)
    for i, row in enumerate(pred_matrix):
        w, n, t = categorical_majority(row.tolist())
        out[i], nvotes[i], tie[i] = w, n, t
    return out, nvotes, tie


# ============================================================
# LOADING
# ============================================================

def load_and_align():
    print()
    print("Loading prediction files...")
    frames = {}
    for name, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {name}: {path}")
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
        frames[name] = df
        print(f"  {name:5} {len(df):>7,} rows   "
              f"{df['Timestamp'].iloc[0]} -> {df['Timestamp'].iloc[-1]}")

    common = set(frames[MODELS[0]]["Timestamp"])
    for name in MODELS[1:]:
        common &= set(frames[name]["Timestamp"])
    common = sorted(common)

    print()
    print(f"Common timestamps across all five models: {len(common):,}")
    if not common:
        raise ValueError("No common timestamps -- cannot compare models.")

    for name in MODELS:
        f = frames[name]
        frames[name] = (f[f["Timestamp"].isin(common)]
                        .sort_values("Timestamp")
                        .reset_index(drop=True))

    ref = frames[MODELS[0]]["Timestamp"].values
    for name in MODELS[1:]:
        if not np.array_equal(frames[name]["Timestamp"].values, ref):
            raise ValueError(f"Timestamp alignment failed for {name}")
    print("  Timestamp alignment: PASS")
    return frames


def to_code(series):
    if series.dtype == object:
        return series.map(CLASS_CODE).astype(int)
    return series.astype(int)


# ============================================================
# STATE CONSTRUCTION
# ============================================================

def _streak(values, target):
    out = np.zeros(len(values), dtype=int)
    run = 0
    for i, v in enumerate(values):
        run = run + 1 if v == target else 0
        out[i] = min(run, MAX_STREAK)
    return out


def _value_streak(values):
    out = np.zeros(len(values), dtype=int)
    run, prev = 0, None
    for i, v in enumerate(values):
        run = run + 1 if v == prev else 1
        out[i] = min(run, MAX_STREAK)
        prev = v
    return out


def build_state(frames):
    print()
    print("Building unified prediction state...")

    ref = frames[MODELS[0]]
    state = pd.DataFrame({"Timestamp": ref["Timestamp"].values})
    state["Actual"] = to_code(ref["Actual_Class"]).values
    state["Date"] = state["Timestamp"].dt.date
    state["Month"] = state["Timestamp"].dt.month
    state["MinuteOfDay"] = state["Timestamp"].dt.hour * 60 + state["Timestamp"].dt.minute

    for name in MODELS:
        actual_here = to_code(frames[name]["Actual_Class"]).values
        mismatches = int(np.sum(actual_here != state["Actual"].values))
        if mismatches:
            raise ValueError(f"{name} disagrees on {mismatches} actual targets")
    print("  Actual-target consistency across all five files: PASS")

    for name in MODELS:
        f = frames[name]
        pred = to_code(f["Predicted_Class"]).values
        state[f"{name}_pred"] = pred
        state[f"{name}_correct"] = (pred == state["Actual"].values).astype(int)

        probs = f[["Prob_DOWN", "Prob_NEUTRAL", "Prob_UP"]].values.astype(float)
        ordered = np.sort(probs, axis=1)
        conf = ordered[:, 2]
        margin = ordered[:, 2] - ordered[:, 1]

        state[f"{name}_conf"] = conf
        state[f"{name}_margin"] = margin
        safe = np.clip(probs, 1e-12, 1.0)
        state[f"{name}_entropy"] = -np.sum(safe * np.log(safe), axis=1)

        state[f"{name}_confb"] = pd.cut(conf, bins=CONF_EDGES,
                                        labels=CONF_LABELS, right=False).astype(str)
        state[f"{name}_marginb"] = pd.cut(margin, bins=MARGIN_EDGES,
                                          labels=MARGIN_LABELS, right=False).astype(str)

    pred_matrix = state[[f"{m}_pred" for m in MODELS]].values
    maj, nvotes, tie = majority_series(pred_matrix)
    state["Maj_pred"] = maj
    state["Maj_votes"] = nvotes
    state["Maj_tie"] = tie.astype(int)
    state["Maj_correct"] = np.where(tie, 0, (maj == state["Actual"].values).astype(int))
    state["N_correct"] = state[[f"{m}_correct" for m in MODELS]].sum(axis=1)
    state["All_agree"] = pd.Series([len(set(r)) == 1 for r in pred_matrix.tolist()]).astype(int)

    tree_m = state[["XGB_pred", "LGB_pred", "CAT_pred"]].values
    seq_m = state[["LSTM_pred", "TRF_pred"]].values
    state["Trees_agree"] = pd.Series([len(set(r)) == 1 for r in tree_m.tolist()]).astype(int)
    state["Seq_agree"] = pd.Series([len(set(r)) == 1 for r in seq_m.tolist()]).astype(int)
    state["Trees_vs_Seq"] = ((state["Trees_agree"] == 1) &
                             (state["Seq_agree"] == 1) &
                             (state["XGB_pred"] != state["LSTM_pred"])).astype(int)

    for name in MODELS:
        c = state[f"{name}_correct"].values
        state[f"{name}_cstreak"] = _streak(c, 1)
        state[f"{name}_wstreak"] = _streak(c, 0)
        state[f"{name}_dirstreak"] = _value_streak(state[f"{name}_pred"].values)

    print(f"  State table: {state.shape[0]:,} rows x {state.shape[1]} columns")
    return state


def attach_targets(state):
    print()
    print("Attaching next-bar targets (session-aware)...")

    same_session = (state["Date"].shift(-1) == state["Date"]).fillna(False).values
    state["Valid_next"] = same_session.astype(int)

    state["Next_actual"] = state["Actual"].shift(-1)
    state["Next_up"] = (state["Next_actual"] == 2).astype(float)
    state["Next_down"] = (state["Next_actual"] == 0).astype(float)
    for name in MODELS:
        state[f"{name}_next_correct"] = state[f"{name}_correct"].shift(-1)
    state["Maj_next_correct"] = state["Maj_correct"].shift(-1)

    print(f"  Cross-session boundaries excluded: {int((~same_session).sum()):,}")
    print(f"  Usable rows: {int(same_session.sum()):,}")
    print()
    print("  C1 NOTE: MARKET targets (Next_up / Next_down) are derived from a")
    print(f"  label with a {TARGET_HORIZON_BARS}-bar forward horizon. Consecutive")
    print(f"  MARKET observations share {TARGET_HORIZON_BARS - 1} future bars and are")
    print("  autocorrelated by construction. Effective N is deflated accordingly")
    print("  and MARKET patterns face a stricter survival bar than MODEL patterns.")
    return state


# ============================================================
# EVALUATOR
# ============================================================

class PatternEvaluator:
    """
    Scores a condition mask against a target in both periods, with monthly
    stability, target-type-aware survival bars, and overlap-aware Wilson
    bounds. Every family goes through this same path -- no family gets a
    softer rule.
    """

    def __init__(self, discovery, verification):
        self.disc = discovery
        self.ver = verification
        self._baselines = {}
        self.tested_per_target = Counter()   # C2

    def baseline(self, which, target):
        key = (which, target)
        if key not in self._baselines:
            frame = self.disc if which == "disc" else self.ver
            vals = frame.loc[frame["Valid_next"] == 1, target].dropna()
            self._baselines[key] = float(vals.mean()) if len(vals) else 0.0
        return self._baselines[key]

    def _score(self, frame, mask, target, kind):
        sub = frame.loc[mask & (frame["Valid_next"] == 1), target].dropna()
        n = len(sub)
        if n == 0:
            return 0, 0.0, 0.0, 0.0
        succ = float(sub.sum())
        acc = succ / n
        # C1: Wilson computed on EFFECTIVE n for overlapping targets
        n_eff = effective_n(n, kind)
        wlb = wilson_lower_bound(acc * n_eff, n_eff)
        return n, acc, wlb, n_eff

    def evaluate(self, family, description, disc_mask, ver_mask, target,
                 causal=True):
        kind = target_kind(target)
        bars = SURVIVAL_BARS[kind]

        d_n, d_acc, d_wlb, d_neff = self._score(self.disc, disc_mask, target, kind)
        if d_n < MIN_SUPPORT_DISCOVERY:
            return None

        self.tested_per_target[target] += 1   # C2: count only real tests

        v_n, v_acc, v_wlb, v_neff = self._score(self.ver, ver_mask, target, kind)

        d_base = self.baseline("disc", target)
        v_base = self.baseline("ver", target)
        d_lift = lift(d_acc, d_base)
        v_lift = lift(v_acc, v_base)

        monthly, stable_months = {}, 0
        for month in STABILITY_MONTHS:
            mmask = ver_mask & (self.ver["Month"] == month)
            m_n, m_acc, _, _ = self._score(self.ver, mmask, target, kind)
            monthly[f"m{month}_n"] = m_n
            monthly[f"m{month}_acc"] = round(m_acc, 4) if m_n else None
            if m_n >= MIN_SUPPORT_STABILITY and m_acc > v_base:
                stable_months += 1

        verified = (
            v_n >= MIN_SUPPORT_VERIFY
            and d_lift >= bars["lift_disc"]
            and v_lift >= bars["lift_ver"]
            and (v_wlb - v_base) > bars["wilson_edge"]
        )
        # C4: same-bar associations can never be "robust" -- they are not
        # predictive claims, and labelling them robust would invite misuse.
        robust = bool(verified and stable_months >= 2 and causal)

        row = {
            "family": family,
            "causal": causal,
            "pattern": description,
            "target": target,
            "target_kind": kind,
            "overlap_warning": kind == "MARKET",          # C1
            "disc_n": d_n,
            "disc_n_eff": round(d_neff, 1),
            "disc_acc": round(d_acc, 4),
            "disc_base": round(d_base, 4),
            "disc_lift": round(d_lift, 4),
            "disc_wilson_lb": round(d_wlb, 4),
            "ver_n": v_n,
            "ver_n_eff": round(v_neff, 1),
            "ver_acc": round(v_acc, 4) if v_n else None,
            "ver_base": round(v_base, 4),
            "ver_lift": round(v_lift, 4) if v_n else None,
            "ver_wilson_lb": round(v_wlb, 4) if v_n else None,
            "stable_months": stable_months,
            "verified": bool(verified),
            "robust": robust,
        }
        row.update(monthly)
        return row


# ============================================================
# PATTERN FAMILIES
# ============================================================

def family_pairwise_correctness(ev, targets):
    """Joint correctness state of two models at t-lag, swept over all lags."""
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        for lag in LAGS:
            for sa, sb in itertools.product([1, 0], repeat=2):
                desc = (f"{a}(t-{lag})={'C' if sa else 'W'} & "
                        f"{b}(t-{lag})={'C' if sb else 'W'}")
                dm = (ev.disc[f"{a}_correct"].shift(lag) == sa) & \
                     (ev.disc[f"{b}_correct"].shift(lag) == sb)
                vm = (ev.ver[f"{a}_correct"].shift(lag) == sa) & \
                     (ev.ver[f"{b}_correct"].shift(lag) == sb)
                for t in targets:
                    r = ev.evaluate("pairwise_correctness", desc, dm, vm, t,
                                    causal=True)
                    if r:
                        rows.append(r)
    return rows


def family_same_bar_association(ev, targets):
    """
    C4: lag-0 predicted-class relationships.

    These are CONTEMPORANEOUS associations, reported separately and never
    marked robust. They describe how the models relate to each other on the
    same bar; they are not predictive conditions.
    """
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        for ca, cb in itertools.product(CLASSES, repeat=2):
            desc = f"[same-bar] {a}={CLASS_NAME[ca]} & {b}={CLASS_NAME[cb]}"
            dm = (ev.disc[f"{a}_pred"] == ca) & (ev.disc[f"{b}_pred"] == cb)
            vm = (ev.ver[f"{a}_pred"] == ca) & (ev.ver[f"{b}_pred"] == cb)
            for t in targets:
                r = ev.evaluate("same_bar_association", desc, dm, vm, t,
                                causal=False)
                if r:
                    rows.append(r)
    return rows


def family_lagged_direction(ev, targets):
    """Predicted-class interaction at strictly positive lags (causal)."""
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        for lag in (1, 2):
            for ca, cb in itertools.product(CLASSES, repeat=2):
                desc = (f"{a}(t-{lag})={CLASS_NAME[ca]} & "
                        f"{b}(t-{lag})={CLASS_NAME[cb]}")
                dm = (ev.disc[f"{a}_pred"].shift(lag) == ca) & \
                     (ev.disc[f"{b}_pred"].shift(lag) == cb)
                vm = (ev.ver[f"{a}_pred"].shift(lag) == ca) & \
                     (ev.ver[f"{b}_pred"].shift(lag) == cb)
                for t in targets:
                    r = ev.evaluate("lagged_direction", desc, dm, vm, t,
                                    causal=True)
                    if r:
                        rows.append(r)
    return rows


def family_confidence(ev, targets):
    """Bucketed confidence and margin. Bucketed, not threshold-swept --
    keeping the best cut from a continuous sweep manufactures discoveries."""
    rows = []
    for m in MODELS:
        for bucket in CONF_LABELS:
            desc = f"{m} confidence {bucket}"
            dm = ev.disc[f"{m}_confb"] == bucket
            vm = ev.ver[f"{m}_confb"] == bucket
            for t in targets:
                r = ev.evaluate("confidence", desc, dm, vm, t, causal=True)
                if r:
                    rows.append(r)
        for bucket in MARGIN_LABELS:
            desc = f"{m} margin {bucket}"
            dm = ev.disc[f"{m}_marginb"] == bucket
            vm = ev.ver[f"{m}_marginb"] == bucket
            for t in targets:
                r = ev.evaluate("margin", desc, dm, vm, t, causal=True)
                if r:
                    rows.append(r)
    return rows


def family_confidence_cross(ev, targets):
    """Does A's certainty say anything about B's next correctness?"""
    rows = []
    for a, b in itertools.permutations(MODELS, 2):
        for bucket in ("c60-70", "c70+"):
            for sa in (1, 0):
                desc = (f"{a} conf {bucket} & {a}(t)="
                        f"{'C' if sa else 'W'} -> {b}")
                dm = (ev.disc[f"{a}_confb"] == bucket) & (ev.disc[f"{a}_correct"] == sa)
                vm = (ev.ver[f"{a}_confb"] == bucket) & (ev.ver[f"{a}_correct"] == sa)
                t = f"{b}_next_correct"
                if t in targets:
                    r = ev.evaluate("confidence_cross", desc, dm, vm, t, causal=True)
                    if r:
                        rows.append(r)
    return rows


def family_streaks(ev, targets):
    rows = []
    for m in MODELS:
        for k in range(2, MAX_STREAK + 1):
            for kind, col in (("correct", "cstreak"), ("wrong", "wstreak")):
                desc = f"{m} {kind} streak >= {k}"
                dm = ev.disc[f"{m}_{col}"] >= k
                vm = ev.ver[f"{m}_{col}"] >= k
                for t in targets:
                    r = ev.evaluate("streak", desc, dm, vm, t, causal=True)
                    if r:
                        rows.append(r)
            desc = f"{m} same-direction streak >= {k}"
            dm = ev.disc[f"{m}_dirstreak"] >= k
            vm = ev.ver[f"{m}_dirstreak"] >= k
            for t in targets:
                r = ev.evaluate("direction_streak", desc, dm, vm, t, causal=True)
                if r:
                    rows.append(r)
    return rows


def family_triples(ev, targets):
    """All C(5,3)=10 three-model combinations."""
    rows = []
    for combo in itertools.combinations(MODELS, 3):
        label = "+".join(combo)
        nc_d = sum(ev.disc[f"{m}_correct"] for m in combo)
        nc_v = sum(ev.ver[f"{m}_correct"] for m in combo)

        for k in range(4):
            desc = f"{label}: exactly {k}/3 correct"
            for t in targets:
                r = ev.evaluate("triple_correctness", desc, nc_d == k, nc_v == k,
                                t, causal=True)
                if r:
                    rows.append(r)

        for c in CLASSES:
            desc = f"{label}: all three predict {CLASS_NAME[c]}"
            dm = pd.Series(
                np.logical_and.reduce([(ev.disc[f"{m}_pred"] == c).values for m in combo]),
                index=ev.disc.index)
            vm = pd.Series(
                np.logical_and.reduce([(ev.ver[f"{m}_pred"] == c).values for m in combo]),
                index=ev.ver.index)
            for t in targets:
                r = ev.evaluate("triple_agreement", desc, dm, vm, t, causal=True)
                if r:
                    rows.append(r)
    return rows


def family_ensemble_states(ev, targets):
    """Five-model states using correct categorical voting, ties included."""
    rows = []

    for k in range(6):
        desc = f"exactly {k}/5 models correct"
        for t in targets:
            r = ev.evaluate("ensemble_correctness", desc,
                            ev.disc["N_correct"] == k, ev.ver["N_correct"] == k,
                            t, causal=True)
            if r:
                rows.append(r)

    for votes in (3, 4, 5):
        for c in CLASSES:
            desc = f"majority {CLASS_NAME[c]} with {votes}/5 votes"
            dm = ((ev.disc["Maj_pred"] == c) & (ev.disc["Maj_votes"] == votes)
                  & (ev.disc["Maj_tie"] == 0))
            vm = ((ev.ver["Maj_pred"] == c) & (ev.ver["Maj_votes"] == votes)
                  & (ev.ver["Maj_tie"] == 0))
            for t in targets:
                r = ev.evaluate("ensemble_majority", desc, dm, vm, t, causal=True)
                if r:
                    rows.append(r)

    for desc, col in (
        ("all five agree", "All_agree"),
        ("ensemble split (no majority)", "Maj_tie"),
        ("three trees agree", "Trees_agree"),
        ("both sequence models agree", "Seq_agree"),
        ("trees and sequence models disagree", "Trees_vs_Seq"),
    ):
        for t in targets:
            r = ev.evaluate("ensemble_structure", desc,
                            ev.disc[col] == 1, ev.ver[col] == 1, t, causal=True)
            if r:
                rows.append(r)
    return rows


def family_time_of_day(ev, targets):
    rows = []
    phases = [("early session", 0, 690), ("late morning", 690, 750),
              ("midday", 750, 810), ("afternoon", 810, 870), ("close", 870, 1000)]
    for desc, lo, hi in phases:
        dm = (ev.disc["MinuteOfDay"] >= lo) & (ev.disc["MinuteOfDay"] < hi)
        vm = (ev.ver["MinuteOfDay"] >= lo) & (ev.ver["MinuteOfDay"] < hi)
        for t in targets:
            r = ev.evaluate("time_of_day", desc, dm, vm, t, causal=True)
            if r:
                rows.append(r)
    return rows


# ============================================================
# DESCRIPTIVE ENSEMBLE TABLE
# ============================================================

def build_ensemble_state_table(state):
    records = defaultdict(lambda: {"n": 0, "maj_correct": 0, "actual": Counter()})
    for _, row in state.iterrows():
        key = tuple(int(row[f"{m}_pred"]) for m in MODELS)
        rec = records[key]
        rec["n"] += 1
        rec["maj_correct"] += int(row["Maj_correct"])
        rec["actual"][int(row["Actual"])] += 1

    out = []
    for key, rec in sorted(records.items(), key=lambda x: -x[1]["n"]):
        winner, votes, tie = categorical_majority(list(key))
        out.append({
            "state": " | ".join(f"{m}={CLASS_NAME[c]}" for m, c in zip(MODELS, key)),
            "support": rec["n"],
            "majority": "TIE" if tie else CLASS_NAME[winner],
            "majority_votes": votes,
            "is_tie": tie,
            "majority_accuracy": round(rec["maj_correct"] / rec["n"], 4),
            "actual_DOWN": rec["actual"][0],
            "actual_NEUTRAL": rec["actual"][1],
            "actual_UP": rec["actual"][2],
        })
    return pd.DataFrame(out)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 84)
    print("ATHENA 3.0 - PHASE 11A - MODEL INTERACTION DISCOVERY ENGINE V2.1")
    print("=" * 84)
    print()
    print("SCOPE: systematic bounded search over eight pattern families.")
    print("       NOT exhaustive over all joint states -- see module docstring.")
    print("       Trade economics are OUT OF SCOPE in this phase.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_and_align()
    state = build_state(frames)
    state = attach_targets(state)

    disc = state[state["Month"] < VERIFY_START_MONTH].reset_index(drop=True)
    ver = state[state["Month"] >= VERIFY_START_MONTH].reset_index(drop=True)

    print()
    print("Chronological split:")
    print(f"  Discovery    (Jan-Sep): {len(disc):,} rows  "
          f"{disc['Timestamp'].iloc[0]} -> {disc['Timestamp'].iloc[-1]}")
    print(f"  Verification (Oct-Dec): {len(ver):,} rows  "
          f"{ver['Timestamp'].iloc[0]} -> {ver['Timestamp'].iloc[-1]}")
    if disc["Timestamp"].max() >= ver["Timestamp"].min():
        raise ValueError("Discovery/verification overlap -- leakage.")
    print("  No overlap: PASS")

    model_targets = [f"{m}_next_correct" for m in MODELS] + ["Maj_next_correct"]
    market_targets = ["Next_up", "Next_down"]
    all_targets = model_targets + market_targets

    print()
    print("Targets (kept strictly separate):")
    print(f"  MODEL  ({len(model_targets)}): will a given model be correct next bar")
    print(f"  MARKET ({len(market_targets)}): next bar direction "
          f"[overlapping {TARGET_HORIZON_BARS}-bar horizon, stricter bar applied]")

    print()
    print("Unconditional baselines:")
    for t in all_targets:
        d = disc.loc[disc["Valid_next"] == 1, t].dropna()
        v = ver.loc[ver["Valid_next"] == 1, t].dropna()
        print(f"  {t:22} [{target_kind(t):6}] discovery={d.mean():.4f}  "
              f"verification={v.mean():.4f}")

    ev = PatternEvaluator(disc, ver)

    print()
    print("Searching pattern families...")
    families = [
        ("pairwise correctness",  family_pairwise_correctness, all_targets),
        ("lagged direction",      family_lagged_direction,     all_targets),
        ("same-bar association",  family_same_bar_association, all_targets),
        ("confidence / margin",   family_confidence,           all_targets),
        ("confidence cross",      family_confidence_cross,     model_targets),
        ("streaks",               family_streaks,              all_targets),
        ("triple interactions",   family_triples,              all_targets),
        ("ensemble states",       family_ensemble_states,      all_targets),
        ("time of day",           family_time_of_day,          all_targets),
    ]

    rows = []
    for label, fn, targets in families:
        found = fn(ev, targets)
        rows.extend(found)
        print(f"  {label:24} {len(found):>6,} cleared discovery support")

    results = pd.DataFrame(rows)
    if results.empty:
        print()
        print(f"No pattern cleared discovery support (n >= {MIN_SUPPORT_DISCOVERY}).")
        return

    results = results.sort_values(["robust", "verified", "ver_lift"],
                                  ascending=[False, False, False]).reset_index(drop=True)

    verified = results[results["verified"]].copy()
    robust = results[results["robust"]].copy()

    results.to_csv(OUTPUT_DIR / "all_candidates.csv", index=False)
    verified.to_csv(OUTPUT_DIR / "verified_patterns.csv", index=False)
    robust.to_csv(OUTPUT_DIR / "robust_patterns.csv", index=False)
    build_ensemble_state_table(state).to_csv(
        OUTPUT_DIR / "ensemble_state_table.csv", index=False)

    by_family = (results.groupby("family")
                 .agg(candidates=("pattern", "size"),
                      verified=("verified", "sum"),
                      robust=("robust", "sum"),
                      best_ver_lift=("ver_lift", "max"))
                 .reset_index().sort_values("robust", ascending=False))
    by_family.to_csv(OUTPUT_DIR / "family_summary.csv", index=False)

    # ---- C2: multiple-testing accounting ----
    testing = []
    for t in all_targets:
        n_tested = ev.tested_per_target[t]
        n_verified = int((verified["target"] == t).sum()) if len(verified) else 0
        n_robust = int((robust["target"] == t).sum()) if len(robust) else 0
        testing.append({
            "target": t,
            "target_kind": target_kind(t),
            "candidates_tested": n_tested,
            "expected_false_positives": round(n_tested * ASSUMED_ALPHA, 1),
            "verified": n_verified,
            "robust": n_robust,
        })
    testing_df = pd.DataFrame(testing)
    testing_df.to_csv(OUTPUT_DIR / "multiple_testing_accounting.csv", index=False)

    summary = {
        "generated": str(datetime.now()),
        "engine": "Model Interaction Discovery Engine V2.1",
        "scope": "systematic bounded search over 8 families; NOT exhaustive",
        "trade_economics": "out of scope in this phase",
        "target_horizon_bars": TARGET_HORIZON_BARS,
        "market_target_overlap_note": (
            f"consecutive MARKET labels share {TARGET_HORIZON_BARS - 1} future "
            "bars; effective N deflated, stricter survival bar applied"),
        "common_rows": int(len(state)),
        "discovery_rows": int(len(disc)),
        "verification_rows": int(len(ver)),
        "lags_examined": list(LAGS),
        "survival_bars": SURVIVAL_BARS,
        "thresholds": {
            "min_support_discovery": MIN_SUPPORT_DISCOVERY,
            "min_support_verify": MIN_SUPPORT_VERIFY,
            "min_support_stability": MIN_SUPPORT_STABILITY,
            "assumed_alpha": ASSUMED_ALPHA,
        },
        "candidates": int(len(results)),
        "verified": int(len(verified)),
        "robust": int(len(robust)),
        "multiple_testing": testing,
        "output_directory": str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "discovery_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---------------- console ----------------
    print()
    print("=" * 84)
    print("RESULTS")
    print("=" * 84)
    print(f"  Candidates tested : {len(results):,}")
    print(f"  Verified          : {len(verified):,}")
    print(f"  Robust            : {len(robust):,}  (verified + causal + >=2 stable months)")

    print()
    print("Multiple-testing accounting (C2):")
    print(f"  {'target':22} {'kind':7} {'tested':>7} {'exp.FP':>7} {'verif':>6} {'robust':>7}")
    for r in testing:
        print(f"  {r['target']:22} {r['target_kind']:7} {r['candidates_tested']:>7,} "
              f"{r['expected_false_positives']:>7} {r['verified']:>6} {r['robust']:>7}")
    print()
    print("  Read the survivor count against expected false positives. If they")
    print("  are comparable, the survivors are indistinguishable from chance.")

    print()
    print("By family:")
    print(f"  {'family':26} {'cand':>7} {'verif':>7} {'robust':>7} {'best_lift':>10}")
    for _, r in by_family.iterrows():
        bl = r["best_ver_lift"]
        print(f"  {r['family']:26} {r['candidates']:>7,} {int(r['verified']):>7,} "
              f"{int(r['robust']):>7,} {(f'{bl:+.3f}' if pd.notna(bl) else 'n/a'):>10}")

    if len(robust):
        print()
        print("Top robust patterns:")
        print()
        for _, r in robust.head(15).iterrows():
            flag = "  [OVERLAPPING TARGET]" if r["overlap_warning"] else ""
            print(f"  [{r['target_kind']}] {r['pattern']} -> {r['target']}{flag}")
            print(f"     discovery : n={r['disc_n']:<5} (eff {r['disc_n_eff']}) "
                  f"acc={r['disc_acc']:.4f} base={r['disc_base']:.4f} "
                  f"lift={r['disc_lift']:+.3f}")
            print(f"     verify    : n={r['ver_n']:<5} (eff {r['ver_n_eff']}) "
                  f"acc={r['ver_acc']:.4f} base={r['ver_base']:.4f} "
                  f"lift={r['ver_lift']:+.3f} wilson_lb={r['ver_wilson_lb']:.4f}")
            print(f"     stability : {r['stable_months']}/3 months above baseline")
            print()
    else:
        print()
        print("  No pattern met the robustness bar.")
        print("  This is a result, not a failure. It says the five models' errors")
        print("  contain no repeatable, out-of-sample-stable relationship at these")
        print("  support levels in 2023. See all_candidates.csv for near misses.")

    if len(robust):
        n_market = int((robust["target_kind"] == "MARKET").sum())
        print("Target split among robust patterns:")
        print(f"  MODEL  (ensemble diagnostics)      : {len(robust) - n_market}")
        print(f"  MARKET (candidates for evaluation) : {n_market}")
        print()
        print("  MARKET patterns are CANDIDATES only. None is a trading signal")
        print("  until it has been through entry timing, horizon, costs,")
        print("  slippage and trade-count analysis in a later phase.")

    print()
    print("Saved to:", OUTPUT_DIR)
    for f in ["all_candidates.csv", "verified_patterns.csv", "robust_patterns.csv",
              "ensemble_state_table.csv", "family_summary.csv",
              "multiple_testing_accounting.csv", "discovery_summary.json"]:
        print(f"  {f}")
    print()
    print("=" * 84)


if __name__ == "__main__":
    main()