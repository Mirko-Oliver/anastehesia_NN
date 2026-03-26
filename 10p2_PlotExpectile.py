"""
Load a saved GAM bundle and plot a dashboard on the Testing split.

Supports:
1) three-curve Expectile bundle:
   bundle["models"] = {0.1: ..., 0.5: ..., 0.9: ...}

2) single-model bundle:
   bundle["model"] = ...

Outputs:
    gam_dashboard_test.png
"""

import os
import csv
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple


# -----------------------------
# Paths / constants
# -----------------------------
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

BUNDLE_PATH = "bis_causal_expectilegam.pkl"
OUTPUT_PNG = "gam_dashboard_test2.png"

STATIC_FEATURES = ["age", "sex", "lbm"]
TARGET_COL = "BIS/BIS"

BIS_EDGES = np.array(
    [16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101],
    dtype=float,
)


# -----------------------------
# Data loading helpers
# -----------------------------
def collect_case_ids(dataset: str) -> List[str]:
    case_ids = []
    file_path = os.path.join(DATA_DIR, CLINICAL_INFO)
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("Dataset") == dataset:
                case_ids.append(row.get("caseid"))
    return case_ids


def create_mask(df: pd.DataFrame) -> pd.Series:
    return (df["label"] == "regular") & (df["Time"] > 1500)


def load_clinical_info_map(
    static_cols: List[str] = STATIC_FEATURES,
) -> Dict[str, Dict[str, str]]:
    fpath = os.path.join(DATA_DIR, CLINICAL_INFO)
    info: Dict[str, Dict[str, str]] = {}

    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("caseid")
            if cid is None:
                continue
            info[cid] = {c: row.get(c) for c in static_cols}

    return info


def add_static_features(
    df: pd.DataFrame,
    caseid: str,
    clinical_map: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    statics = clinical_map[caseid]
    for k, v in statics.items():
        df[k] = v
    return df


def coerce_static_types(df: pd.DataFrame) -> pd.DataFrame:
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({"m": 1, "f": 0})
    if "lbm" in df.columns:
        df["lbm"] = pd.to_numeric(df["lbm"], errors="coerce")
    return df


def load_split_arrays(
    case_ids: List[str],
    features: List[str],
    clinical_map: Dict[str, Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []

    for cid in case_ids:
        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_mask(df)].copy()
        if len(df) == 0:
            continue

        df = add_static_features(df, cid, clinical_map)
        df = coerce_static_types(df)

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Case {cid} missing feature columns: {missing}")

        X_list.append(df[features].to_numpy(dtype=np.float32))
        y_list.append(df[TARGET_COL].to_numpy(dtype=np.float32))

    if not X_list:
        raise RuntimeError("No testing data loaded. Check paths, mask, and features.")

    return np.vstack(X_list), np.concatenate(y_list)


def filter_finite(X: np.ndarray, y: np.ndarray):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


# -----------------------------
# Metrics / binning
# -----------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def make_bis_bins(y: np.ndarray) -> np.ndarray:
    y_num = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    b = np.digitize(y_num, BIS_EDGES, right=False) - 1
    b[np.isnan(y_num)] = -1
    b[(y_num < BIS_EDGES[0]) | (y_num >= BIS_EDGES[-1])] = -1
    return b.astype(int)


# -----------------------------
# Bundle / prediction helpers
# -----------------------------
def is_three_curve_bundle(bundle: dict) -> bool:
    return "models" in bundle and isinstance(bundle["models"], dict)


def predict_ordered_expectiles(models: Dict[float, object], X: np.ndarray) -> Dict[float, np.ndarray]:
    taus = sorted(models.keys())
    pred_stack = np.vstack([models[tau].predict(X) for tau in taus])
    pred_stack = np.maximum.accumulate(pred_stack, axis=0)
    return {tau: pred_stack[i] for i, tau in enumerate(taus)}


def get_term_label(features_used: List[str], gam, term_idx: int) -> str:
    term = gam.terms[term_idx]

    if getattr(term, "istensor", False):
        idxs = list(term.feature)
        names = [features_used[i] for i in idxs]
        return " × ".join(names)

    if getattr(term, "isintercept", False):
        return "Intercept"

    feat = getattr(term, "feature", None)
    if isinstance(feat, (int, np.integer)):
        return features_used[int(feat)]

    return f"term_{term_idx}"


def get_feature_quantiles(x: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.nanquantile(x, lo)), float(np.nanquantile(x, hi))


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_pred_vs_obs(ax, y_true, y_pred):
    n = len(y_true)
    rng = np.random.default_rng(13)
    idx = rng.choice(n, size=min(n, 15000), replace=False) if n > 15000 else np.arange(n)

    ax.scatter(y_true[idx], y_pred[idx], s=6, alpha=0.25)
    lo = float(min(np.nanmin(y_true[idx]), np.nanmin(y_pred[idx])))
    hi = float(max(np.nanmax(y_true[idx]), np.nanmax(y_pred[idx])))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title("Observed vs Predicted")
    ax.set_xlabel("Observed BIS")
    ax.set_ylabel("Predicted BIS")


def plot_residuals(ax, y_true, y_pred):
    resid = y_true - y_pred
    n = len(y_true)
    rng = np.random.default_rng(17)
    idx = rng.choice(n, size=min(n, 15000), replace=False) if n > 15000 else np.arange(n)

    ax.scatter(y_pred[idx], resid[idx], s=6, alpha=0.25)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted BIS")
    ax.set_ylabel("Residual")


def plot_rmse_by_bin(ax, y_true, y_pred):
    bins = make_bis_bins(y_true)
    labels = []
    rmses = []

    for b in range(len(BIS_EDGES) - 1):
        m = bins == b
        if not np.any(m):
            continue
        lo, hi = BIS_EDGES[b], BIS_EDGES[b + 1]
        labels.append(f"{int(lo)}-{int(hi)}")
        rmses.append(rmse(y_true[m], y_pred[m]))

    ax.bar(np.arange(len(rmses)), rmses)
    ax.set_xticks(np.arange(len(rmses)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_title("RMSE by BIS Bin")
    ax.set_ylabel("RMSE")


def plot_band_coverage(ax, y_true, p10, p50, p90):
    covered = (y_true >= p10) & (y_true <= p90)
    width = p90 - p10

    ax.scatter(p50, width, s=6, alpha=0.2)
    ax.set_title(f"10-90 Band Width | coverage={covered.mean():.3f}")
    ax.set_xlabel("Median prediction")
    ax.set_ylabel("Band width")


def plot_sorted_band(ax, y_true, p10, p50, p90):
    order = np.argsort(p50)
    xs = np.arange(len(order))
    ax.fill_between(xs, p10[order], p90[order], alpha=0.3, label="10-90 band")
    ax.plot(xs, y_true[order], linewidth=1, label="Observed")
    ax.plot(xs, p50[order], linewidth=1, label="Median")
    ax.set_title("Sorted Predictions with Expectile Band")
    ax.set_xlabel("Sorted test rows")
    ax.set_ylabel("BIS")
    ax.legend()


def plot_1d_terms(fig, outer_spec, gam, X_test, features_used, max_terms: int = 12):
    one_d_terms = []
    for term_idx, term in enumerate(gam.terms):
        if getattr(term, "isintercept", False):
            continue
        if getattr(term, "istensor", False):
            continue
        feat = getattr(term, "feature", None)
        if isinstance(feat, (int, np.integer)):
            one_d_terms.append(term_idx)

    one_d_terms = one_d_terms[:max_terms]
    n = len(one_d_terms)
    if n == 0:
        return

    ncols = 3
    nrows = math.ceil(n / ncols)
    subgs = outer_spec.subgridspec(nrows, ncols)

    for k, term_idx in enumerate(one_d_terms):
        ax = fig.add_subplot(subgs[k // ncols, k % ncols])

        feat_idx = int(gam.terms[term_idx].feature)
        feat_name = features_used[feat_idx]

        xx = gam.generate_X_grid(term=term_idx, n=200)
        pdp = gam.partial_dependence(term=term_idx, X=xx)

        ax.plot(xx[:, feat_idx], pdp)
        ax.axhline(0.0, linestyle="--")
        ax.set_title(feat_name)
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Effect")


def plot_2d_terms(fig, outer_spec, gam, features_used, max_terms: int = 6):
    tensor_terms = []
    for term_idx, term in enumerate(gam.terms):
        if getattr(term, "istensor", False):
            tensor_terms.append(term_idx)

    tensor_terms = tensor_terms[:max_terms]
    n = len(tensor_terms)
    if n == 0:
        return

    ncols = 2
    nrows = math.ceil(n / ncols)
    subgs = outer_spec.subgridspec(nrows, ncols)

    for k, term_idx in enumerate(tensor_terms):
        ax = fig.add_subplot(subgs[k // ncols, k % ncols])

        try:
            XX = gam.generate_X_grid(term=term_idx, n=40, meshgrid=True)
            Z = gam.partial_dependence(term=term_idx, X=XX, meshgrid=True)

            idxs = list(gam.terms[term_idx].feature)
            xname = features_used[idxs[0]]
            yname = features_used[idxs[1]]

            m = ax.contourf(XX[0], XX[1], Z, levels=20)
            fig.colorbar(m, ax=ax)
            ax.set_title(f"{xname} × {yname}")
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Could not plot term {term_idx}\n{exc}", ha="center", va="center")
            ax.set_axis_off()


# -----------------------------
# Main dashboard
# -----------------------------
def main():
    bundle = joblib.load(BUNDLE_PATH)
    features_used = bundle["features_used"]

    clinical_map = load_clinical_info_map(STATIC_FEATURES)
    test_case_ids = collect_case_ids("Testing")
    X_test, y_test = load_split_arrays(test_case_ids, features_used, clinical_map)
    X_test, y_test = filter_finite(X_test, y_test)

    if is_three_curve_bundle(bundle):
        models = bundle["models"]
        pred_map = predict_ordered_expectiles(models, X_test)
        p10 = pred_map[min(pred_map.keys())]
        p50 = pred_map[0.5] if 0.5 in pred_map else pred_map[sorted(pred_map.keys())[len(pred_map) // 2]]
        p90 = pred_map[max(pred_map.keys())]
        gam_for_terms = models[0.5] if 0.5 in models else models[sorted(models.keys())[len(models) // 2]]
        title_suffix = "Three-Curve Expectile GAM"
    else:
        gam = bundle["model"]
        p50 = gam.predict(X_test)
        p10 = None
        p90 = None
        gam_for_terms = gam
        title_suffix = "Single GAM"

    overall_rmse = rmse(y_test, p50)

    fig = plt.figure(figsize=(22, 18))
    outer = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.4, 1.8])

    # Top-left: summary diagnostics
    diag_gs = outer[0, 0].subgridspec(2, 2)

    ax1 = fig.add_subplot(diag_gs[0, 0])
    plot_pred_vs_obs(ax1, y_test, p50)

    ax2 = fig.add_subplot(diag_gs[0, 1])
    plot_residuals(ax2, y_test, p50)

    ax3 = fig.add_subplot(diag_gs[1, 0])
    plot_rmse_by_bin(ax3, y_test, p50)

    ax4 = fig.add_subplot(diag_gs[1, 1])
    ax4.axis("off")
    text = [
        f"Model type: {title_suffix}",
        f"Testing rows: {len(y_test):,}",
        f"Features used: {len(features_used)}",
        f"Overall RMSE: {overall_rmse:.4f}",
    ]
    if is_three_curve_bundle(bundle):
        coverage = np.mean((y_test >= p10) & (y_test <= p90))
        text.extend([
            f"10-90 coverage: {coverage:.4f}",
            f"Mean band width: {np.mean(p90 - p10):.4f}",
        ])
    ax4.text(0.01, 0.98, "\n".join(text), va="top", ha="left", family="monospace", fontsize=11)

    # Top-right: expectile band diagnostics or residual hist
    if is_three_curve_bundle(bundle):
        band_gs = outer[0, 1].subgridspec(1, 2)

        ax5 = fig.add_subplot(band_gs[0, 0])
        plot_band_coverage(ax5, y_test, p10, p50, p90)

        ax6 = fig.add_subplot(band_gs[0, 1])
        plot_sorted_band(ax6, y_test, p10, p50, p90)
    else:
        ax5 = fig.add_subplot(outer[0, 1])
        resid = y_test - p50
        ax5.hist(resid, bins=50)
        ax5.set_title("Residual Histogram")
        ax5.set_xlabel("Residual")
        ax5.set_ylabel("Count")

    # Middle row: 1D term effects
    plot_1d_terms(fig, outer[1, :], gam_for_terms, X_test, features_used, max_terms=12)

    # Bottom row: 2D interaction surfaces
    plot_2d_terms(fig, outer[2, :], gam_for_terms, features_used, max_terms=6)

    fig.suptitle(f"GAM Dashboard on Testing Data | {title_suffix}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    print(f"[INFO] Saved dashboard to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
