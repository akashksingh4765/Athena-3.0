from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
"""
Athena 3.0 - Phase 12A - MISTAKE LAYER V2 (leak-safe rebuild)

PURPOSE
-------
Predict P(base ensemble decision is wrong | information available NOW),
and use it to filter out low-quality decisions (selective classification).

WHY V2 EXISTS -- THE V1 LEAK
----------------------------
V1 built history features with shift(1): "was the previous row correct".
But every class label is a FORWARD label spanning HORIZON_BARS future bars.
Whether the 11:35 prediction was right is not known until ~12:05. At 11:40,
V1 already "knew" it. Because overlapping labels are strongly correlated,
"was the previous row wrong" leaks much of "will this row be wrong".
V1's ROC-AUC of 0.80 is therefore not trustworthy.

V2 FIXES
--------
F1. History features are built ONLY from labels whose resolution time
    (timestamp + HORIZON_BARS bars, session-aware) is <= the current row's
    timestamp. Asserted, not assumed.
F2. Base decision ties are detected and broken by mean probability,
    not by argmax's silent lowest-class-code rule. Tie flag is a feature.
F3. Purged chronological splits: Train (Jan-Aug) / Calibration (Sep) /
    Verification (Oct-Dec). Rows whose labels resolve after a split
    boundary are purged from the earlier set.
F4. Operating threshold chosen on CALIBRATION, applied unchanged to
    verification. No threshold is picked by looking at verification.
F5. Free baselines. The layer must beat simple rules that need no training
    (base confidence, probability margin, vote count). If it doesn't,
    it adds complexity without information.
F6. Day-block bootstrap confidence intervals. 590 verification rows is
    small; point estimates alone are misleading.
F7. Results by base class. NEUTRAL decisions are no-trade decisions;
    filtering matters most for UP/DOWN.
F8. Leak audit: a V1-style model is trained side by side so the size of
    the leak is measured, not guessed. Its numbers must NOT be used.
F9. No hyperparameter search. With ~1,600 training rows, tuning to raise
    verification accuracy is overfitting to verification.

OUT OF SCOPE: trade economics (costs, slippage, P&L). Filtering accuracy
is not profitability.

LOCKED TARGET DEFINITION
------------------------
Athena intraday target v1 uses 6 future 5-minute bars = 30 minutes.
This horizon is now locked and matches the target definition used by the
base models.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(str(_ATHENA_ROOT))

FILES = {
    "XGB": ROOT / "experiments/xgboost/xgboost_v4_oos_2023_predictions.csv",
    "LGB": ROOT / "experiments/lightgbm/lightgbm_v3_oos_2023_predictions.csv",
    "CAT": ROOT / "experiments/catboost/catboost_v3_oos_2023_predictions.csv",
    "LSTM": ROOT / "experiments/lstm/lstm_v2_validation_predictions.csv",
    "TRF": ROOT / "experiments/transformer/transformer_v1_validation_predictions.csv",
}

OUT = ROOT / "reports/model_comparison/mistake_layer_v2"

MODELS = list(FILES)

TREES = [
    "XGB",
    "LGB",
    "CAT",
]

SEQS = [
    "LSTM",
    "TRF",
]

CLASSES = [
    0,
    1,
    2,
]

LABELS = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}

# Locked Athena target definition:
# 6 future 5-minute bars = 30 minutes.
HORIZON_BARS = 6
BAR_MINUTES = 5

EMBARGO_EXTRA_BARS = 0

SESSION_OPEN = 9 * 60 + 15
SESSION_CLOSE = 15 * 60 + 30

TRAIN_END = pd.Timestamp("2023-09-01")
CALIB_END = pd.Timestamp("2023-10-01")

HISTORY_WINDOWS = (
    5,
    10,
    20,
)

COVERAGES = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

OPERATING_COVERAGE = 0.60

N_BOOT = 1000

SEED = 42


# ============================================================
# LOADING
# ============================================================

def code(s):
    n = pd.to_numeric(
        s,
        errors="coerce",
    )

    o = pd.Series(
        np.nan,
        index=s.index,
        dtype=float,
    )

    m = n.notna()

    o.loc[m] = n.loc[m]

    o.loc[~m] = (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map(
            {
                "DOWN": 0,
                "NEUTRAL": 1,
                "UP": 2,
            }
        )
        .loc[~m]
    )

    if o.isna().any():
        raise ValueError(
            "Unrecognized class values"
        )

    o = o.astype(int)

    if not o.isin(CLASSES).all():
        raise ValueError(
            "Bad class code"
        )

    return o


def col(
    df,
    names,
    what,
):
    for x in names:
        if x in df:
            return x

    raise KeyError(
        f"{what}: tried {names}; "
        f"available={list(df.columns)}"
    )


def load(path):

    d = pd.read_csv(path)

    o = pd.DataFrame()

    o["Timestamp"] = pd.to_datetime(
        d["Timestamp"]
    )

    o["Actual_Class"] = code(
        d[
            col(
                d,
                [
                    "Actual_Class",
                    "Target_Code",
                    "Actual_Code",
                    "Target_Name",
                ],
                "actual",
            )
        ]
    )

    o["Predicted_Class"] = code(
        d[
            col(
                d,
                [
                    "Predicted_Class",
                    "Predicted_Code",
                    "Prediction_Code",
                    "Predicted_Name",
                ],
                "prediction",
            )
        ]
    )

    for c in [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]:

        o[f"Prob_{c}"] = pd.to_numeric(
            d[
                col(
                    d,
                    [
                        f"Prob_{c}",
                        f"Probability_{c}",
                    ],
                    f"{c} probability",
                )
            ],
            errors="raise",
        )

    p = o[
        [
            "Prob_DOWN",
            "Prob_NEUTRAL",
            "Prob_UP",
        ]
    ].to_numpy(float)

    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(
            p.sum(1),
            1,
            atol=1e-5,
        )
    ):
        raise ValueError(
            f"Invalid probabilities in {path.name}"
        )

    return (
        o
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )


def load_aligned():

    fs = {
        m: load(p)
        for m, p in FILES.items()
    }

    common = sorted(
        set(
            fs[MODELS[0]].Timestamp
        ).intersection(
            *[
                set(fs[m].Timestamp)
                for m in MODELS[1:]
            ]
        )
    )

    if not common:
        raise ValueError(
            "No common timestamps"
        )

    fs = {
        m:
            fs[m][
                fs[m].Timestamp.isin(common)
            ]
            .sort_values("Timestamp")
            .reset_index(drop=True)
        for m in MODELS
    }

    if (
        fs[MODELS[0]]
        .Timestamp
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate timestamps -- "
            "multi-instrument data needs a symbol key"
        )

    actual = (
        fs[MODELS[0]]
        .Actual_Class
        .to_numpy()
    )

    for m in MODELS:

        if not np.array_equal(
            actual,
            fs[m]
            .Actual_Class
            .to_numpy(),
        ):
            raise ValueError(
                f"Actual target mismatch in {m}"
            )

    return fs, actual


# ============================================================
# TIMING
# ============================================================

def label_resolution_times(ts):
    """
    When each row's label becomes KNOWN:
    timestamp + locked 6-bar horizon, in trading time.
    """

    need = (
        HORIZON_BARS
        + EMBARGO_EXTRA_BARS
    ) * BAR_MINUTES

    days = sorted(
        ts.dt.normalize().unique()
    )

    nxt = {
        d:
            (
                days[i + 1]
                if i + 1 < len(days)
                else d + pd.Timedelta(days=1)
            )
        for i, d in enumerate(days)
    }

    out = []

    for t in ts:

        m = (
            t.hour * 60
            + t.minute
        )

        if (
            m + need
            <= SESSION_CLOSE
        ):

            out.append(
                t
                + pd.Timedelta(
                    minutes=need
                )
            )

        else:

            overflow = (
                m
                + need
                - SESSION_CLOSE
            )

            out.append(
                nxt[t.normalize()]
                + pd.Timedelta(
                    minutes=(
                        SESSION_OPEN
                        + overflow
                    )
                )
            )

    return pd.Series(
        out,
        index=ts.index,
    )


def timing_diagnostics(
    ts,
    res,
):

    same_day = (
        ts.dt.date
        .eq(ts.dt.date.shift(1))
    )

    gaps = (
        ts
        - ts.shift(1)
    ).dt.total_seconds() / 60

    prev_unresolved = (
        res.shift(1) > ts
    ) & same_day

    return {
        "rows": int(
            len(ts)
        ),
        "median_gap_minutes_same_session":
            float(
                gaps[same_day].median()
            ),
        "share_gaps_under_horizon":
            float(
                (
                    gaps[same_day]
                    < HORIZON_BARS * BAR_MINUTES
                ).mean()
            ),
        "share_rows_where_prev_label_unresolved":
            float(
                prev_unresolved.mean()
            ),
    }


# ============================================================
# BASE DECISION
# ============================================================

def base_decision(
    pm,
    mean_probs,
):

    counts = np.stack(
        [
            (pm == c).sum(1)
            for c in CLASSES
        ],
        1,
    )

    top = counts.max(1)

    tied = (
        counts
        == top[:, None]
    ).sum(1) > 1

    masked = np.where(
        counts
        == top[:, None],
        mean_probs,
        -np.inf,
    )

    base = np.where(
        tied,
        masked.argmax(1),
        counts.argmax(1),
    )

    return (
        base,
        top,
        tied,
    )


# ============================================================
# FEATURES
# ============================================================

def contemporaneous_features(
    ts,
    pm,
    P,
    base,
    votes,
    tied,
):
    """
    Everything here is known the moment
    the five predictions exist.
    """

    n = len(ts)

    rows = np.arange(n)

    f = {}

    mean = P.mean(1)

    for i, m in enumerate(MODELS):

        p = P[:, i, :]

        sp = np.sort(
            p,
            1,
        )

        for c in CLASSES:

            f[
                f"{m}_Prob_{LABELS[c]}"
            ] = p[:, c]

        f[
            f"{m}_Confidence"
        ] = sp[:, -1]

        f[
            f"{m}_Margin"
        ] = (
            sp[:, -1]
            - sp[:, -2]
        )

        f[
            f"{m}_Entropy"
        ] = -(
            np.clip(
                p,
                1e-12,
                1,
            )
            * np.log(
                np.clip(
                    p,
                    1e-12,
                    1,
                )
            )
        ).sum(1)

        f[
            f"{m}_ProbOfBase"
        ] = p[
            rows,
            base,
        ]

        f[
            f"{m}_Agrees_Base"
        ] = (
            pm[:, i]
            == base
        ).astype(int)

    sm = np.sort(
        mean,
        1,
    )

    for c in CLASSES:

        f[
            f"Mean_Prob_{LABELS[c]}"
        ] = mean[:, c]

        f[
            f"Std_Prob_{LABELS[c]}"
        ] = P[
            :,
            :,
            c,
        ].std(1)

        f[
            f"Base_Is_{LABELS[c]}"
        ] = (
            base == c
        ).astype(int)

    f["Mean_Margin"] = (
        sm[:, -1]
        - sm[:, -2]
    )

    f["Mean_ProbOfBase"] = mean[
        rows,
        base,
    ]

    f["Base_Votes"] = votes

    f["Base_Tie"] = tied.astype(int)

    f["N_Distinct_Preds"] = np.array(
        [
            len(set(r))
            for r in pm.tolist()
        ]
    )

    f["Pairwise_Disagreements"] = sum(
        (
            pm[:, a]
            != pm[:, b]
        ).astype(int)
        for a in range(5)
        for b in range(a + 1, 5)
    )

    ti = [
        MODELS.index(m)
        for m in TREES
    ]

    si = [
        MODELS.index(m)
        for m in SEQS
    ]

    tree_agree = (
        pm[:, ti]
        == pm[:, [ti[0]]]
    ).all(1)

    seq_agree = (
        pm[:, si]
        == pm[:, [si[0]]]
    ).all(1)

    f["Tree_Agree"] = (
        tree_agree
    ).astype(int)

    f["Seq_Agree"] = (
        seq_agree
    ).astype(int)

    f["Tree_Seq_Disagree"] = (
        tree_agree
        & seq_agree
        & (
            pm[:, ti[0]]
            != pm[:, si[0]]
        )
    ).astype(int)

    f["Tree_ProbOfBase"] = (
        P[:, ti, :].mean(1)
    )[
        rows,
        base,
    ]

    f["Seq_ProbOfBase"] = (
        P[:, si, :].mean(1)
    )[
        rows,
        base,
    ]

    f["Tree_minus_Seq_ProbOfBase"] = (
        f["Tree_ProbOfBase"]
        - f["Seq_ProbOfBase"]
    )

    mfo = (
        ts.dt.hour * 60
        + ts.dt.minute
        - SESSION_OPEN
    )

    f["Minutes_From_Open"] = (
        mfo.to_numpy()
    )

    f["Time_Sin"] = np.sin(
        2
        * np.pi
        * mfo
        / 375
    ).to_numpy()

    f["Time_Cos"] = np.cos(
        2
        * np.pi
        * mfo
        / 375
    ).to_numpy()

    return pd.DataFrame(
        f,
        index=ts.index,
    )


def lagged_prediction_features(
    ts,
    pm,
    P,
    base,
):
    """
    SAFE lags: previous row's PREDICTIONS,
    which were known at that time.

    Never uses previous correctness here.
    """

    same = (
        ts.dt.date
        .eq(ts.dt.date.shift(1))
        .to_numpy()
    )

    f = {}

    for i, m in enumerate(MODELS):

        prev = (
            pd.Series(
                pm[:, i]
            )
            .shift(1)
            .to_numpy()
        )

        f[
            f"{m}_Flipped"
        ] = np.where(
            same,
            (
                pm[:, i]
                != prev
            ).astype(float),
            np.nan,
        )

    prev_base = (
        pd.Series(base)
        .shift(1)
        .to_numpy()
    )

    f["Base_Flipped"] = np.where(
        same,
        (
            base
            != prev_base
        ).astype(float),
        np.nan,
    )

    mean = P.mean(1)

    sm = np.sort(
        mean,
        1,
    )

    margin = (
        sm[:, -1]
        - sm[:, -2]
    )

    f["Delta_Mean_Margin"] = np.where(
        same,
        (
            margin
            - pd.Series(
                margin
            ).shift(1).to_numpy()
        ),
        np.nan,
    )

    gap = (
        ts
        - ts.shift(1)
    ).dt.total_seconds().to_numpy() / 60

    f["Gap_From_Prev_Row_Min"] = np.where(
        same,
        gap,
        np.nan,
    )

    return pd.DataFrame(
        f,
        index=ts.index,
    )


def resolved_history(
    ts,
    res,
    ok,
    prefix,
):
    """
    History built ONLY from labels resolved by
    the current timestamp.

    Ordered by resolution time.
    """

    order = np.argsort(
        res.to_numpy(),
        kind="stable",
    )

    res_sorted = (
        res.to_numpy()[order]
    )

    ok_sorted = (
        ok[order]
        .astype(float)
    )

    ts_sorted = (
        ts.to_numpy()[order]
    )

    cum = np.concatenate(
        [
            [0.0],
            np.cumsum(ok_sorted),
        ]
    )

    streak = np.zeros(
        len(ok_sorted)
    )

    run = 0

    for i, v in enumerate(
        ok_sorted
    ):

        run = (
            run + 1
            if v == 0
            else 0
        )

        streak[i] = run

    n_known = np.searchsorted(
        res_sorted,
        ts.to_numpy(),
        side="right",
    )

    # F1 assertion:
    # no unresolved label may enter history.

    if len(ts_sorted):

        run_max_ts = np.maximum.accumulate(
            ts_sorted
        )

        has = n_known > 0

        idx = np.maximum(
            n_known - 1,
            0,
        )

        if (
            run_max_ts[idx][has]
            >= ts.to_numpy()[has]
        ).any():

            raise AssertionError(
                f"{prefix}: history includes "
                "a not-earlier row -- leak"
            )

        if (
            res_sorted[idx][has]
            > ts.to_numpy()[has]
        ).any():

            raise AssertionError(
                f"{prefix}: history includes "
                "an unresolved label -- leak"
            )

    has = (
        n_known > 0
    )

    last = np.maximum(
        n_known - 1,
        0,
    )

    f = {
        f"{prefix}_LastKnown_Correct":
            np.where(
                has,
                ok_sorted[last],
                np.nan,
            ),

        f"{prefix}_Known_ErrorStreak":
            np.where(
                has,
                streak[last],
                np.nan,
            ),
    }

    for w in HISTORY_WINDOWS:

        lo = np.maximum(
            n_known - w,
            0,
        )

        cnt = (
            n_known
            - lo
        )

        f[
            f"{prefix}_Known_Acc_{w}"
        ] = np.where(
            cnt >= w,
            (
                cum[n_known]
                - cum[lo]
            )
            / np.maximum(
                cnt,
                1,
            ),
            np.nan,
        )

    age = (
        ts.to_numpy()
        - ts_sorted[last]
    ) / np.timedelta64(
        1,
        "m",
    )

    return (
        pd.DataFrame(
            f,
            index=ts.index,
        ),
        np.where(
            has,
            age,
            np.nan,
        ),
    )


def v1_style_leaky_history(
    ok_by_prefix,
):
    """
    Reproduces V1's shift(1) history.

    Used ONLY for leak audit.
    """

    f = {}

    for prefix, ok in ok_by_prefix.items():

        s = pd.Series(
            ok.astype(float)
        )

        prev = s.shift(1)

        f[
            f"{prefix}_Prev1_Correct_LEAKY"
        ] = prev

        err = 1 - prev

        f[
            f"{prefix}_ErrorStreak_LEAKY"
        ] = (
            err
            .groupby(
                err.eq(0).cumsum()
            )
            .cumsum()
        )

        for w in HISTORY_WINDOWS:

            f[
                f"{prefix}_Acc_{w}_LEAKY"
            ] = (
                prev
                .rolling(
                    w,
                    min_periods=1,
                )
                .mean()
            )

    return pd.DataFrame(f)


# ============================================================
# MODELS
# ============================================================

def make_hgb():

    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=8,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
    )


def make_logit():

    return make_pipeline(
        SimpleImputer(
            strategy="median"
        ),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=5000,
        ),
    )


def platt(
    raw_cal,
    y_cal,
    raw_apply,
):
    """
    Sigmoid calibration fit on the
    calibration month only.
    """

    eps = 1e-6

    z = lambda p: np.log(
        np.clip(
            p,
            eps,
            1 - eps,
        )
        /
        (
            1
            - np.clip(
                p,
                eps,
                1 - eps,
            )
        )
    )

    lr = LogisticRegression(
        C=1e6,
        max_iter=5000,
    )

    lr.fit(
        z(raw_cal).reshape(-1, 1),
        y_cal,
    )

    return lr.predict_proba(
        z(raw_apply).reshape(-1, 1)
    )[:, 1]


# ============================================================
# EVALUATION
# ============================================================

def safe_auc(
    y,
    s,
):

    return (
        roc_auc_score(
            y,
            s,
        )
        if len(np.unique(y)) == 2
        else np.nan
    )


def selective_curve(
    score,
    correct,
    base_cls,
):
    """
    Accept the lowest-risk fraction
    of rows at each coverage.
    """

    order = np.argsort(
        score,
        kind="stable",
    )

    rows = []

    for c in COVERAGES:

        k = max(
            1,
            int(
                round(
                    c * len(score)
                )
            ),
        )

        acc_idx = order[:k]

        directional = acc_idx[
            base_cls[acc_idx] != 1
        ]

        rows.append(
            {
                "coverage": c,
                "accepted": k,
                "accepted_accuracy":
                    float(
                        correct[
                            acc_idx
                        ].mean()
                    ),
                "directional_accepted":
                    int(
                        len(directional)
                    ),
                "directional_accuracy":
                    (
                        float(
                            correct[
                                directional
                            ].mean()
                        )
                        if len(directional)
                        else np.nan
                    ),
            }
        )

    return rows


def day_bootstrap(
    days,
    stat,
    rng,
):

    uniq = np.unique(
        days
    )

    by_day = {
        d:
            np.where(
                days == d
            )[0]
        for d in uniq
    }

    vals = []

    for _ in range(
        N_BOOT
    ):

        idx = np.concatenate(
            [
                by_day[d]
                for d in rng.choice(
                    uniq,
                    len(uniq),
                    replace=True,
                )
            ]
        )

        v = stat(idx)

        if (
            v is not None
            and np.isfinite(v)
        ):
            vals.append(v)

    if len(vals) < (
        N_BOOT * 0.5
    ):

        return [
            np.nan,
            np.nan,
        ]

    return [
        float(
            np.percentile(
                vals,
                2.5,
            )
        ),
        float(
            np.percentile(
                vals,
                97.5,
            )
        ),
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        SEED
    )

    print(
        "=" * 84
    )

    print(
        "ATHENA 3.0 - PHASE 12A - "
        "MISTAKE LAYER V2 (leak-safe)"
    )

    print(
        "=" * 84
    )

    # --------------------------------------------------------
    # LOAD AND ALIGN
    # --------------------------------------------------------

    fs, actual = load_aligned()

    ts = (
        fs[MODELS[0]]
        .Timestamp
        .reset_index(drop=True)
    )

    n = len(ts)

    pm = np.column_stack(
        [
            fs[m]
            .Predicted_Class
            .to_numpy()
            for m in MODELS
        ]
    )

    P = np.stack(
        [
            fs[m][
                [
                    "Prob_DOWN",
                    "Prob_NEUTRAL",
                    "Prob_UP",
                ]
            ].to_numpy(float)
            for m in MODELS
        ],
        axis=1,
    )

    base, votes, tied = base_decision(
        pm,
        P.mean(1),
    )

    base_correct = (
        base == actual
    ).astype(int)

    y_wrong = 1 - base_correct

    # --------------------------------------------------------
    # LABEL RESOLUTION
    # --------------------------------------------------------

    res = label_resolution_times(
        ts
    )

    diag = timing_diagnostics(
        ts,
        res,
    )

    print(
        f"\nRows: {n:,}   "
        f"Horizon: {HORIZON_BARS} bars "
        f"({HORIZON_BARS * BAR_MINUTES} min) "
        f"+ embargo {EMBARGO_EXTRA_BARS}"
    )

    if tied.any():

        print(
            "Base ties "
            f"(resolved by mean probability): "
            f"{int(tied.sum()):,} "
            f"({tied.mean():.1%}); "
            f"base accuracy on ties "
            f"{base_correct[tied].mean():.3f}"
        )

    else:

        print(
            "Base ties: 0"
        )

    print(
        "\nTiming diagnostics:"
    )

    print(
        "  median gap between rows "
        "(same session): "
        f"{diag['median_gap_minutes_same_session']:.0f} min"
    )

    print(
        "  gaps shorter than label horizon       : "
        f"{diag['share_gaps_under_horizon']:.1%}"
    )

    print(
        "  rows where V1 used an UNRESOLVED label : "
        f"{diag['share_rows_where_prev_label_unresolved']:.1%}"
        "   <-- size of V1's leak exposure"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = contemporaneous_features(
        ts,
        pm,
        P,
        base,
        votes,
        tied,
    )

    X = X.join(
        lagged_prediction_features(
            ts,
            pm,
            P,
            base,
        )
    )

    ok_by_prefix = {
        m:
            (
                pm[:, i]
                == actual
            ).astype(int)
        for i, m in enumerate(
            MODELS
        )
    }

    ok_by_prefix["Base"] = (
        base_correct
    )

    age = None

    for prefix, ok in (
        ok_by_prefix.items()
    ):

        h, age = resolved_history(
            ts,
            res,
            ok,
            prefix,
        )

        X = X.join(h)

    X["Known_History_AgeMin"] = age

    print(
        "\nLeak-safety assertions "
        "on history features: PASS"
    )

    # --------------------------------------------------------
    # LEAKY VERSION FOR AUDIT ONLY
    # --------------------------------------------------------

    X_leaky = (
        contemporaneous_features(
            ts,
            pm,
            P,
            base,
            votes,
            tied,
        )
        .join(
            lagged_prediction_features(
                ts,
                pm,
                P,
                base,
            )
        )
        .join(
            v1_style_leaky_history(
                ok_by_prefix
            )
        )
    )

    features = list(
        X.columns
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLITS
    # --------------------------------------------------------

    train = (
        (
            ts
            < TRAIN_END
        )
        &
        (
            res
            <= TRAIN_END
        )
    ).to_numpy()

    calib = (
        (
            ts
            >= TRAIN_END
        )
        &
        (
            ts
            < CALIB_END
        )
        &
        (
            res
            <= CALIB_END
        )
    ).to_numpy()

    ver = (
        ts
        >= CALIB_END
    ).to_numpy()

    purged = int(
        (
            ts
            < CALIB_END
        ).sum()
        - train.sum()
        - calib.sum()
    )

    print(
        f"\nSplits: "
        f"train {train.sum():,} | "
        f"calibration {calib.sum():,} | "
        f"verification {ver.sum():,} | "
        f"purged at boundaries {purged}"
    )

    if (
        calib.sum() < 50
        or ver.sum() < 100
    ):

        raise ValueError(
            "Split too small to evaluate honestly"
        )

    yv = y_wrong[ver]

    correct_v = (
        base_correct[ver]
    )

    base_v = base[ver]

    days_v = (
        ts[ver]
        .dt.date
        .to_numpy()
    )

    months_v = (
        ts[ver]
        .dt.month
        .to_numpy()
    )

    # --------------------------------------------------------
    # FIT MODELS
    # --------------------------------------------------------

    scorers = {}

    fitted = {}

    for name, factory in (
        (
            "HGB",
            make_hgb,
        ),
        (
            "LOGIT",
            make_logit,
        ),
    ):

        mdl = factory()

        mdl.fit(
            X[train],
            y_wrong[train],
        )

        raw_cal = (
            mdl
            .predict_proba(
                X[calib]
            )[:, 1]
        )

        raw_ver = (
            mdl
            .predict_proba(
                X[ver]
            )[:, 1]
        )

        scorers[
            f"Layer_{name}"
        ] = {
            "cal": raw_cal,
            "ver": raw_ver,
            "ver_calibrated":
                platt(
                    raw_cal,
                    y_wrong[calib],
                    raw_ver,
                ),
        }

        fitted[name] = mdl

    # --------------------------------------------------------
    # LEAK AUDIT MODEL
    # --------------------------------------------------------

    leaky = make_hgb()

    leaky.fit(
        X_leaky[train],
        y_wrong[train],
    )

    leaky_auc = safe_auc(
        yv,
        leaky
        .predict_proba(
            X_leaky[ver]
        )[:, 1],
    )

    # --------------------------------------------------------
    # FREE BASELINES
    # --------------------------------------------------------

    free = {
        "Free_LowBaseConfidence":
            -X[
                "Mean_ProbOfBase"
            ].to_numpy(),

        "Free_LowMargin":
            -X[
                "Mean_Margin"
            ].to_numpy(),

        "Free_FewVotes":
            -(
                X[
                    "Base_Votes"
                ].to_numpy()
                +
                0.5
                * (
                    1
                    - X[
                        "Base_Tie"
                    ].to_numpy()
                )
            ),
    }

    for k, s in free.items():

        scorers[k] = {
            "cal": s[calib],
            "ver": s[ver],
        }

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    summary = []

    curves = []

    for name, s in (
        scorers.items()
    ):

        auc = safe_auc(
            yv,
            s["ver"],
        )

        ci = day_bootstrap(
            days_v,
            lambda idx,
            sv=s["ver"]:
                safe_auc(
                    yv[idx],
                    sv[idx],
                ),
            rng,
        )

        monthly = {
            int(mo):
                safe_auc(
                    yv[
                        months_v == mo
                    ],
                    s["ver"][
                        months_v == mo
                    ],
                )
            for mo in np.unique(
                months_v
            )
        }

        # ----------------------------------------------------
        # IMPORTANT:
        # THRESHOLD IS CHOSEN ONLY ON CALIBRATION.
        # IT IS NEVER OPTIMIZED ON VERIFICATION.
        # ----------------------------------------------------

        thr = float(
            np.quantile(
                s["cal"],
                OPERATING_COVERAGE,
            )
        )

        acc_mask = (
            s["ver"]
            <= thr
        )

        row = {
            "scorer": name,

            "ver_auc": auc,

            "ver_auc_ci_low":
                ci[0],

            "ver_auc_ci_high":
                ci[1],

            **{
                f"auc_month_{mo}": v
                for mo, v in monthly.items()
            },

            "op_target_coverage":
                OPERATING_COVERAGE,

            # NEW:
            # Save the exact calibration threshold.
            "op_threshold":
                thr,

            "op_realized_coverage":
                float(
                    acc_mask.mean()
                ),

            "op_accepted_accuracy":
                (
                    float(
                        correct_v[
                            acc_mask
                        ].mean()
                    )
                    if acc_mask.any()
                    else np.nan
                ),
        }

        if (
            "ver_calibrated"
            in s
        ):

            pc = s[
                "ver_calibrated"
            ]

            row.update(
                {
                    "ver_log_loss":
                        log_loss(
                            yv,
                            np.column_stack(
                                [
                                    1 - pc,
                                    pc,
                                ]
                            ),
                            labels=[
                                0,
                                1,
                            ],
                        ),

                    "ver_brier":
                        brier_score_loss(
                            yv,
                            pc,
                        ),

                    "ver_balanced_acc_at_0.5":
                        balanced_accuracy_score(
                            yv,
                            (
                                pc >= 0.5
                            ).astype(int),
                        ),
                }
            )

        summary.append(
            row
        )

        for r in selective_curve(
            s["ver"],
            correct_v,
            base_v,
        ):

            curves.append(
                {
                    "scorer": name,
                    **r,
                }
            )

    summary_df = pd.DataFrame(
        summary
    )

    curves_df = pd.DataFrame(
        curves
    )

    best_free = (
        summary_df[
            summary_df.scorer
            .str.startswith("Free")
        ]
        .sort_values(
            "ver_auc"
        )
        .iloc[-1]
    )

    # --------------------------------------------------------
    # PAIRED BOOTSTRAP
    # --------------------------------------------------------

    comparisons = {}

    for lname in (
        "Layer_HGB",
        "Layer_LOGIT",
    ):

        ls = scorers[
            lname
        ]["ver"]

        fsv = scorers[
            best_free.scorer
        ]["ver"]

        comparisons[lname] = (
            day_bootstrap(
                days_v,
                lambda idx,
                a=ls,
                b=fsv:
                    safe_auc(
                        yv[idx],
                        a[idx],
                    )
                    -
                    safe_auc(
                        yv[idx],
                        b[idx],
                    ),
                rng,
            )
        )

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    pi = permutation_importance(
        fitted["HGB"],
        X[ver],
        yv,
        scoring="roc_auc",
        n_repeats=10,
        random_state=SEED,
    )

    imp = (
        pd.DataFrame(
            {
                "feature":
                    features,

                "auc_drop_mean":
                    pi.importances_mean,

                "auc_drop_std":
                    pi.importances_std,
            }
        )
        .sort_values(
            "auc_drop_mean",
            ascending=False,
        )
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    ds = X.copy()

    ds.insert(
        0,
        "Timestamp",
        ts,
    )

    ds["Label_Resolves_At"] = res

    ds["Base_Class"] = base

    ds["Actual_Class"] = actual

    ds["Base_Wrong"] = y_wrong

    ds["Split"] = np.select(
        [
            train,
            calib,
            ver,
        ],
        [
            "train",
            "calibration",
            "verification",
        ],
        "purged",
    )

    ds.to_csv(
        OUT
        / "mistake_layer_v2_dataset.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE VERIFICATION PREDICTIONS
    # --------------------------------------------------------

    vp = ds.loc[
        ver,
        [
            "Timestamp",
            "Base_Class",
            "Actual_Class",
            "Base_Wrong",
        ],
    ].copy()

    for k, s in scorers.items():

        vp[
            f"Score_{k}"
        ] = s["ver"]

    vp[
        "P_Base_Wrong_HGB_calibrated"
    ] = scorers[
        "Layer_HGB"
    ][
        "ver_calibrated"
    ]

    vp.to_csv(
        OUT
        / "mistake_layer_v2_verification_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_df.to_csv(
        OUT
        / "mistake_layer_v2_scorer_summary.csv",
        index=False,
    )

    curves_df.to_csv(
        OUT
        / "mistake_layer_v2_selective_curves.csv",
        index=False,
    )

    imp.to_csv(
        OUT
        / "mistake_layer_v2_permutation_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    base_acc_v = float(
        correct_v.mean()
    )

    # Extract the exact HGB threshold
    # that was selected from September
    # calibration data.

    hgb_summary_row = (
        summary_df[
            summary_df["scorer"]
            == "Layer_HGB"
        ]
        .iloc[0]
    )

    hgb_operating_threshold = float(
        hgb_summary_row[
            "op_threshold"
        ]
    )

    report = {
        "phase":
            "12A",

        "version":
            "V2 leak-safe",

        "horizon_bars_assumed":
            HORIZON_BARS,

        "horizon_minutes":
            HORIZON_BARS * BAR_MINUTES,

        "embargo_extra_bars":
            EMBARGO_EXTRA_BARS,

        "timing_diagnostics":
            diag,

        "splits":
            {
                "train":
                    int(
                        train.sum()
                    ),

                "calibration":
                    int(
                        calib.sum()
                    ),

                "verification":
                    int(
                        ver.sum()
                    ),

                "purged":
                    purged,
            },

        "base_ties":
            int(
                tied.sum()
            ),

        "verification_base_accuracy":
            base_acc_v,

        "verification_base_mistake_rate":
            float(
                yv.mean()
            ),

        "feature_count":
            len(features),

        "operating_coverage":
            OPERATING_COVERAGE,

        # NEW:
        # Exact numeric threshold used by HGB.
        "layer_hgb_operating_threshold":
            hgb_operating_threshold,

        "scorers":
            summary,

        "best_free_baseline":
            best_free.scorer,

        "layer_minus_best_free_auc_ci":
            comparisons,

        "leak_audit_v1_style_auc":
            leaky_auc,

        "out_of_scope":
            "trade economics; filtering accuracy is not profitability",
    }

    with open(
        OUT
        / "mistake_layer_v2_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # CONSOLE
    # --------------------------------------------------------

    print(
        f"\nVerification: "
        f"base accuracy "
        f"{base_acc_v:.4f}, "
        f"mistake rate "
        f"{yv.mean():.4f}"
    )

    print(
        "\nScorer comparison "
        "(verification; AUC for "
        "predicting base mistakes):"
    )

    print(
        f"  {'scorer':24} "
        f"{'AUC':>6}  "
        f"{'95% CI (day blocks)':>20}  "
        f"{'op cov':>7} "
        f"{'op acc':>7} "
        f"{'threshold':>12}"
    )

    for r in summary:

        print(
            f"  {r['scorer']:24} "
            f"{r['ver_auc']:6.3f}  "
            f"[{r['ver_auc_ci_low']:.3f}, "
            f"{r['ver_auc_ci_high']:.3f}]      "
            f"{r['op_realized_coverage']:7.1%} "
            f"{r['op_accepted_accuracy']:7.4f} "
            f"{r['op_threshold']:12.6f}"
        )

    print(
        f"\n  op = threshold fixed on "
        f"September for "
        f"{OPERATING_COVERAGE:.0%} coverage, "
        "applied unchanged to Oct-Dec"
    )

    print(
        "\nLayer vs best free baseline "
        f"({best_free.scorer}), "
        "AUC difference 95% CI:"
    )

    for k, ci in comparisons.items():

        verdict = (
            "beats baseline"
            if ci[0] > 0
            else
            (
                "NOT distinguishable "
                "from free baseline"
                if ci[1] > 0
                else
                "worse than baseline"
            )
        )

        print(
            f"  {k:12} "
            f"[{ci[0]:+.3f}, "
            f"{ci[1]:+.3f}] "
            f"-> {verdict}"
        )

    print(
        "\nLeak audit:"
    )

    hgb_auc = (
        summary_df
        .set_index("scorer")
        .loc[
            "Layer_HGB",
            "ver_auc",
        ]
    )

    print(
        "  V1-style "
        "(shift-1 history) AUC : "
        f"{leaky_auc:.3f}"
        "   <-- contaminated, "
        "do not use"
    )

    print(
        "  V2 leak-safe HGB AUC           : "
        f"{hgb_auc:.3f}"
    )

    print(
        "  Portion attributable to leak   : "
        f"{leaky_auc - hgb_auc:+.3f}"
    )

    print(
        "\nFrozen HGB operating threshold:"
    )

    print(
        f"  September calibration "
        f"threshold: "
        f"{hgb_operating_threshold:.10f}"
    )

    print(
        f"  Target coverage: "
        f"{OPERATING_COVERAGE:.1%}"
    )

    print(
        "\nSelective curve, "
        "Layer_HGB "
        "(accept lowest-risk fraction):"
    )

    c = curves_df[
        curves_df.scorer
        == "Layer_HGB"
    ]

    print(
        c[
            [
                "coverage",
                "accepted",
                "accepted_accuracy",
                "directional_accepted",
                "directional_accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTop 10 features "
        "(permutation importance, "
        "verification, descriptive):"
    )

    for _, r in (
        imp.head(10)
        .iterrows()
    ):

        print(
            f"  {r.feature:34} "
            f"{r.auc_drop_mean:+.4f} "
            f"+/- "
            f"{r.auc_drop_std:.4f}"
        )

    print(
        f"\nSaved to: {OUT}"
    )

    print(
        "=" * 84
    )


if __name__ == "__main__":
    main()