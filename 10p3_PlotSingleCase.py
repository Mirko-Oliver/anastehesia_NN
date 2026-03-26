"""
Plot a single-case BIS comparison for:
- bis_causal_expectilegam.pkl   (uses tau = 0.5, with 0.1-0.9 band if available)
- bis_causal_gam.pkl            (single GAM)

Allows:
- different feature sets between bundles
- different feature order between bundles
- different term sets / term counts / term ordering between models
"""

import os
import csv
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional


DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"
TARGET_COL = "BIS/BIS"
STATIC_FEATURES = ["age", "sex", "lbm"]


# -----------------------------
# Data helpers
# -----------------------------
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
    statics = clinical_map.get(caseid, {})
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


def load_case_dataframe(caseid: str, clinical_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, str(caseid) + SUFFIX)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Case parquet not found: {path}")

    df = pd.read_parquet(path)
    df = df.loc[create_mask(df)].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows left after mask for case {caseid}")

    df = add_static_features(df, str(caseid), clinical_map)
    df = coerce_static_types(df)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Case {caseid} missing target column {TARGET_COL}")

    return df


def load_case_arrays_from_df(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Case missing feature columns: {missing_cols}")

    X = df[features].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    t = df["Time"].to_numpy(dtype=float) if "Time" in df.columns else np.arange(len(df), dtype=float)

    finite_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    y = y[finite_mask]
    t = t[finite_mask]

    if len(y) == 0:
        raise RuntimeError("No finite rows left for requested feature set")

    return X, y, t


# -----------------------------
# Bundle/model helpers
# -----------------------------
def load_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected bundle dict in {path}, got {type(bundle)}")
    return bundle


def get_features_used(bundle: dict) -> List[str]:
    if "features_used" not in bundle:
        raise KeyError("Bundle missing 'features_used'")
    return bundle["features_used"]


def get_expectile_models(bundle: dict):
    if "models" in bundle and isinstance(bundle["models"], dict):
        models = bundle["models"]
        m10 = models.get(0.1, models.get("0.1"))
        m50 = models.get(0.5, models.get("0.5"))
        m90 = models.get(0.9, models.get("0.9"))
        if m50 is None:
            raise KeyError("Expectile bundle has models dict but no tau=0.5 model")
        return m10, m50, m90

    if "model" in bundle:
        return None, bundle["model"], None

    raise KeyError("Expectile bundle missing both 'models' and 'model'")


def get_linear_model(bundle: dict):
    if "model" in bundle:
        return bundle["model"]

    if "models" in bundle and isinstance(bundle["models"], dict):
        models = bundle["models"]
        if 0.5 in models:
            return models[0.5]
        if "0.5" in models:
            return models["0.5"]
        first_key = sorted(models.keys())[0]
        return models[first_key]

    raise KeyError("Linear bundle missing 'model'")


def predict_ordered_expectiles(
    m10,
    m50,
    m90,
    X: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    p50 = m50.predict(X)
    p10 = m10.predict(X) if m10 is not None else None
    p90 = m90.predict(X) if m90 is not None else None

    if p10 is not None and p90 is not None:
        stack = np.vstack([p10, p50, p90])
        stack = np.maximum.accumulate(stack, axis=0)
        p10, p50, p90 = stack[0], stack[1], stack[2]

    return p10, p50, p90


# -----------------------------
# Alignment helper
# -----------------------------
def align_case_views(
    y_exp: np.ndarray,
    t_exp: np.ndarray,
    y_lin: np.ndarray,
    t_lin: np.ndarray,
):
    """
    Align two model-specific case views by Time and target value.
    This lets models use different feature sets and therefore different finite-row masks.
    """
    df_exp = pd.DataFrame({
        "Time": t_exp,
        "y_exp": y_exp,
        "_idx_exp": np.arange(len(t_exp)),
    })
    df_lin = pd.DataFrame({
        "Time": t_lin,
        "y_lin": y_lin,
        "_idx_lin": np.arange(len(t_lin)),
    })

    merged = pd.merge(df_exp, df_lin, on="Time", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping time points between the two model-specific case views")

    # If duplicate times exist, keep only rows where y is effectively the same.
    same_y = np.isclose(merged["y_exp"].to_numpy(), merged["y_lin"].to_numpy(), equal_nan=False)
    merged = merged.loc[same_y].copy()

    if len(merged) == 0:
        raise ValueError(
            "Overlapping times exist, but no rows match on target values. "
            "This suggests duplicated Time stamps or inconsistent row alignment."
        )

    idx_exp = merged["_idx_exp"].to_numpy(dtype=int)
    idx_lin = merged["_idx_lin"].to_numpy(dtype=int)

    return idx_exp, idx_lin, merged["Time"].to_numpy(dtype=float), merged["y_exp"].to_numpy(dtype=float)


# -----------------------------
# Term contribution helpers
# -----------------------------
def get_term_label(gam, features_used: List[str], term_idx: int) -> str:
    term = gam.terms[term_idx]

    if getattr(term, "isintercept", False):
        return "Intercept"

    if getattr(term, "istensor", False):
        feat = getattr(term, "feature", None)
        if feat is None:
            return f"te(term {term_idx})"
        names = [features_used[int(i)] for i in list(feat)]
        return " × ".join(names)

    feat = getattr(term, "feature", None)
    if isinstance(feat, (int, np.integer)):
        return features_used[int(feat)]

    return f"term_{term_idx}"


def get_centered_term_contribution_matrix(
    gam,
    X: np.ndarray,
    features_used: List[str],
) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    rows: List[np.ndarray] = []

    for term_idx, term in enumerate(gam.terms):
        if getattr(term, "isintercept", False):
            continue
        try:
            vals = gam.partial_dependence(term=term_idx, X=X)
            vals = np.asarray(vals).reshape(-1)
            vals = vals - np.nanmean(vals)
            labels.append(get_term_label(gam, features_used, term_idx))
            rows.append(vals)
        except Exception:
            continue

    if not rows:
        return [], np.zeros((0, len(X)), dtype=float)

    return labels, np.vstack(rows)


def shorten_label(text: str, max_len: int = 40) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# -----------------------------
# Plot helpers
# -----------------------------
def plot_prediction_panel(
    ax,
    t: np.ndarray,
    y_true: np.ndarray,
    p_lin: np.ndarray,
    p_exp50: np.ndarray,
    p_exp10: Optional[np.ndarray],
    p_exp90: Optional[np.ndarray],
    caseid: str,
):
    ax.plot(t, y_true, linewidth=1.7, label="True BIS")
    ax.plot(t, p_lin, linewidth=1.2, label="LinearGAM prediction")
    ax.plot(t, p_exp50, linewidth=1.4, label="ExpectileGAM tau=0.5")

    if p_exp10 is not None and p_exp90 is not None:
        ax.fill_between(t, p_exp10, p_exp90, alpha=0.2, label="Expectile band 0.1-0.9")

    ax.set_title(f"Case {caseid}: True vs Predicted BIS")
    ax.set_xlabel("Time")
    ax.set_ylabel("BIS")
    ax.legend(loc="upper right", ncol=2)


def plot_contribution_heatmap(
    fig,
    ax,
    t: np.ndarray,
    labels: List[str],
    mat: np.ndarray,
    title: str,
    max_terms: int = 18,
    shared_vmax: Optional[float] = None,
):
    if mat.shape[0] == 0:
        ax.text(0.5, 0.5, "No term contributions available", ha="center", va="center")
        ax.set_axis_off()
        return

    importance = np.mean(np.abs(mat), axis=1)
    order = np.argsort(-importance)[:max_terms]

    mat = mat[order]
    labels = [labels[i] for i in order]

    vmax = shared_vmax
    if vmax is None:
        vmax = float(np.nanquantile(np.abs(mat), 0.99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[t[0], t[-1], mat.shape[0] - 0.5, -0.5],
    )

    short_labels = [shorten_label(x, max_len=42) for x in labels]
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Term")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Centered signed contribution")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caseid", required=True, help="Case ID to plot")
    parser.add_argument("--expectile_bundle", default="bis_causal_expectilegam.pkl")
    parser.add_argument("--linear_bundle", default="bis_causal_gam.pkl")
    parser.add_argument("--out", default="single_case_bis_dashboard.png")
    args = parser.parse_args()

    exp_bundle = load_bundle(args.expectile_bundle)
    lin_bundle = load_bundle(args.linear_bundle)

    exp_features = get_features_used(exp_bundle)
    lin_features = get_features_used(lin_bundle)

    clinical_map = load_clinical_info_map(STATIC_FEATURES)
    df_case = load_case_dataframe(args.caseid, clinical_map)

    X_case_exp, y_case_exp, t_case_exp = load_case_arrays_from_df(df_case, exp_features)
    X_case_lin, y_case_lin, t_case_lin = load_case_arrays_from_df(df_case, lin_features)

    idx_exp, idx_lin, t_case, y_case = align_case_views(
        y_case_exp, t_case_exp,
        y_case_lin, t_case_lin,
    )

    X_case_exp = X_case_exp[idx_exp]
    X_case_lin = X_case_lin[idx_lin]

    m10, m50, m90 = get_expectile_models(exp_bundle)
    linear_model = get_linear_model(lin_bundle)

    p10, p50, p90 = predict_ordered_expectiles(m10, m50, m90, X_case_exp)
    p_lin = linear_model.predict(X_case_lin)

    if p10 is not None:
        p10 = p10
    if p90 is not None:
        p90 = p90

    lin_labels, lin_mat = get_centered_term_contribution_matrix(linear_model, X_case_lin, lin_features)
    exp_labels, exp_mat = get_centered_term_contribution_matrix(m50, X_case_exp, exp_features)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(18, 14),
        sharex=False,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )

    plot_prediction_panel(
        axes[0],
        t_case,
        y_case,
        p_lin,
        p50,
        p10,
        p90,
        args.caseid,
    )

    plot_contribution_heatmap(
        fig,
        axes[1],
        t_case,
        lin_labels,
        lin_mat,
        "LinearGAM: centered signed term contributions",
        max_terms=18,
        shared_vmax=None,
    )

    plot_contribution_heatmap(
        fig,
        axes[2],
        t_case,
        exp_labels,
        exp_mat,
        "ExpectileGAM tau=0.5: centered signed term contributions",
        max_terms=18,
        shared_vmax=None,
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
