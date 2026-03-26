import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygam.terms import TensorTerm, FactorTerm, SplineTerm


MODEL_PATH = "bis_causal_gam.pkl"
DATA_DIR = Path("data")
CLINICAL_INFO = "__clinical_info.csv"
SUFFIX = "_rawdata.parquet"
TARGET_COL = "BIS/BIS"


# -----------------------------
# Data loading
# -----------------------------
def create_mask(df: pd.DataFrame) -> pd.Series:
    return (df["label"] == "regular") & (df["Time"] > 1500)


def load_model_bundle(path: str):
    bundle = joblib.load(path)
    gam = bundle["model"]
    features = bundle["features_used"]
    restrictors = bundle.get("restrictors", {})
    X_support = bundle.get("X_support_sample", None)
    w_support = bundle.get("w_support_sample", None)
    return gam, features, restrictors, X_support, w_support


def load_clinical_info_map(static_cols=("age", "sex", "lbm")):
    fpath = DATA_DIR / CLINICAL_INFO
    info = {}
    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("caseid")
            if cid is None:
                continue
            info[cid] = {c: row.get(c) for c in static_cols}
    return info


def add_static_features(df: pd.DataFrame, caseid: str, clinical_map):
    if caseid not in clinical_map:
        return df
    statics = clinical_map[caseid]
    for k, v in statics.items():
        if k not in df.columns:
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


def collect_case_ids(dataset: str):
    case_ids = []
    file_path = DATA_DIR / CLINICAL_INFO
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("Dataset") == dataset:
                case_ids.append(row.get("caseid"))
    return case_ids


def load_split_arrays(case_ids, features, clinical_map):
    X_list, y_list = [], []

    for cid in case_ids:
        path = DATA_DIR / f"{cid}{SUFFIX}"
        if not path.exists():
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
        raise RuntimeError("No data loaded. Check mask, paths, and columns.")

    return np.vstack(X_list), np.concatenate(y_list)


def filter_finite(X: np.ndarray, y: np.ndarray):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


# -----------------------------
# Term helpers
# -----------------------------
def _term_features(term):
    if hasattr(term, "feature"):
        feat = term.feature
        if isinstance(feat, (list, tuple, np.ndarray)):
            return tuple(int(x) for x in feat)
        try:
            return (int(feat),)
        except Exception:
            return tuple()
    return tuple()


def choose_baseline_vector(X_ref: np.ndarray) -> np.ndarray:
    return np.nanmedian(X_ref, axis=0)


def pretty_feature_name(name: str) -> str:
    mapping = {
        "BIS/SR": "Suppression Ratio",
        "BIS/SEF": "SEF",
        "BIS/EMG": "EMG",
        "Solar8000/HR": "Heart Rate",
        "Solar8000/ART_MBP": "MAP",
        "Solar8000/ETCO2": "ETCO2",
        "Solar8000/BT": "Body Temp",
        "EEG1_alpha_rel": "EEG Alpha (rel)",
        "EEG1_beta_rel": "EEG Beta (rel)",
        "EEG1_theta_rel": "EEG Theta (rel)",
        "EEG1_delta_rel": "EEG Delta (rel)",
        "EEG1_gamma_rel": "EEG Gamma (rel)",
        "age": "Age",
        "sex": "Sex",
        "lbm": "LBM",
    }

    if name in mapping:
        return mapping[name]

    suffix_map = {
        "_slope30": " (30s slope)",
        "_mean30": " (30s mean)",
        "_mean60": " (60s mean)",
        "_mean300": " (300s mean)",
    }

    for suf, repl in suffix_map.items():
        if name.endswith(suf):
            base = name[: -len(suf)]
            return f"{mapping.get(base, base)}{repl}"

    return mapping.get(name, name)


def get_term_label(term, features):
    feats = _term_features(term)
    if len(feats) == 1:
        return pretty_feature_name(features[feats[0]])
    if len(feats) == 2:
        return f"{pretty_feature_name(features[feats[0]])} × {pretty_feature_name(features[feats[1]])}"
    return str(term)


def infer_mechanism(term, features, restrictors):
    feats = _term_features(term)
    if not feats:
        return "other"

    feat_names = [features[i] for i in feats]
    mechs = restrictors.get("mechanisms", {})

    def matches_node(fname, node):
        return fname == node or fname.startswith(node + "_")

    # 1D terms: only match main nodes
    if len(feat_names) == 1:
        fname = feat_names[0]
        for mech_name, spec in mechs.items():
            for node in spec.get("main", set()):
                if matches_node(fname, node):
                    return mech_name
        return "other"

    # 2D terms: one feature from main, the other from mods
    if len(feat_names) == 2:
        f1, f2 = feat_names
        for mech_name, spec in mechs.items():
            mains = spec.get("main", set())
            mods = spec.get("mods", set())

            f1_main = any(matches_node(f1, n) for n in mains)
            f2_main = any(matches_node(f2, n) for n in mains)
            f1_mod = any(matches_node(f1, n) for n in mods)
            f2_mod = any(matches_node(f2, n) for n in mods)

            if (f1_main and f2_mod) or (f2_main and f1_mod):
                return mech_name

        return "other"

    return "other"


def compute_term_edof(gam):
    edof_per_coef = np.asarray(gam.statistics_.get("edof_per_coef", []), dtype=float)
    if edof_per_coef.size == 0:
        return [np.nan] * len(gam.terms)

    out = []
    start = 0
    for term in gam.terms:
        n_coefs = getattr(term, "n_coefs", None)
        if n_coefs is None:
            out.append(np.nan)
            continue

        end = start + int(n_coefs)
        out.append(float(np.nansum(edof_per_coef[start:end])))
        start = end

    return out


def get_term_lambda(term):
    lam = getattr(term, "lam", None)
    if lam is None:
        return np.nan

    arr = np.asarray(lam, dtype=float).ravel()
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return float(arr[0])
    return arr.tolist()


def make_term_summary(gam, X_ref, features, restrictors, max_rows_eval=4000):
    rng = np.random.default_rng(42)
    if len(X_ref) > max_rows_eval:
        idx = rng.choice(len(X_ref), size=max_rows_eval, replace=False)
        X_eval = X_ref[idx]
    else:
        X_eval = X_ref

    term_edof = compute_term_edof(gam)
    rows = []

    for i, term in enumerate(gam.terms):
        feats = _term_features(term)
        if len(feats) == 0:
            continue

        try:
            contrib = np.asarray(gam.partial_dependence(term=i, X=X_eval), dtype=float).ravel()
            importance = float(np.nanstd(contrib))
            effect_range = float(np.nanpercentile(contrib, 95) - np.nanpercentile(contrib, 5))
        except Exception:
            importance = np.nan
            effect_range = np.nan

        if isinstance(term, TensorTerm):
            term_type = "te"
        elif isinstance(term, SplineTerm):
            term_type = "s"
        elif isinstance(term, FactorTerm):
            term_type = "f"
        else:
            term_type = type(term).__name__

        rows.append(
            {
                "term_i": i,
                "label": get_term_label(term, features),
                "term_type": term_type,
                "n_features": len(feats),
                "feature_idx": feats,
                "lambda": get_term_lambda(term),
                "edof": term_edof[i] if i < len(term_edof) else np.nan,
                "importance": importance,
                "effect_range_p95_p5": effect_range,
                "mechanism": infer_mechanism(term, features, restrictors),
                "rank": getattr(term, "n_coefs", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["importance", "effect_range_p95_p5"], ascending=[False, False]).reset_index(drop=True)

    imp_q = df["importance"].quantile(0.35) if len(df) > 0 else np.nan
    df["low_impact"] = (
        (df["importance"] <= imp_q) |
        (df["edof"] < 0.5) |
        ((df["term_type"] == "f") & (df["edof"] < 0.2))
    )
    df["plot_priority"] = np.where(df["low_impact"], 0, 1)
    return df


# -----------------------------
# Metrics / general plots
# -----------------------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def r2_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    c = np.corrcoef(y_true, y_pred)[0, 1]
    return float(c ** 2)


def plot_true_vs_pred(ax, y_true, y_pred, max_points=5000):
    rng = np.random.default_rng(42)
    n = min(len(y_true), max_points)
    idx = rng.choice(len(y_true), size=n, replace=False) if len(y_true) > n else np.arange(len(y_true))

    ax.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.25)
    lo = min(np.nanmin(y_true[idx]), np.nanmin(y_pred[idx]))
    hi = max(np.nanmax(y_true[idx]), np.nanmax(y_pred[idx]))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.5)
    ax.set_title("True vs Predicted BIS")
    ax.set_xlabel("True BIS")
    ax.set_ylabel("Predicted BIS")


def plot_residual_hist(ax, residuals):
    ax.hist(residuals, bins=50, alpha=0.8)
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.set_title("Residual Histogram")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")


def plot_residual_vs_pred(ax, y_pred, residuals, max_points=5000):
    rng = np.random.default_rng(42)
    n = min(len(y_pred), max_points)
    idx = rng.choice(len(y_pred), size=n, replace=False) if len(y_pred) > n else np.arange(len(y_pred))

    ax.scatter(y_pred[idx], residuals[idx], s=8, alpha=0.25)
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set_title("Residual vs Predicted")
    ax.set_xlabel("Predicted BIS")
    ax.set_ylabel("Residual")


def plot_binned_calibration(ax, y_true, y_pred, n_bins=12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_true, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    bins = np.digitize(y_true, edges[1:-1], right=False)
    x_means, y_means = [], []

    for b in range(n_bins):
        mask = bins == b
        if mask.sum() < 5:
            continue
        x_means.append(np.nanmean(y_true[mask]))
        y_means.append(np.nanmean(y_pred[mask]))

    ax.plot(x_means, y_means, marker="o", linewidth=2)
    lo = min(x_means + y_means) if x_means else 0
    hi = max(x_means + y_means) if x_means else 1
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.5)
    ax.set_title("Binned Calibration")
    ax.set_xlabel("Mean True BIS")
    ax.set_ylabel("Mean Predicted BIS")


# -----------------------------
# 1D term utilities
# -----------------------------
def find_1d_term_index_for_feature(gam, features, feature_name):
    for i, term in enumerate(gam.terms):
        feats = _term_features(term)
        if len(feats) != 1:
            continue
        fi = feats[0]
        if features[fi] == feature_name:
            return i
    return None


def get_1d_curve_data(gam, term_i, feature_name, features, X_support, w_support=None):
    fi = features.index(feature_name)
    XX = gam.generate_X_grid(term=term_i)

    x = XX[:, fi]
    y = np.asarray(gam.partial_dependence(term=term_i, X=XX), dtype=float).ravel()

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_obs = np.asarray(X_support[:, fi], dtype=float)

    if w_support is None:
        w_obs = np.ones_like(x_obs, dtype=float)
    else:
        w_obs = np.asarray(w_support, dtype=float)

    return x, y, x_obs, w_obs


def draw_ticks_on_curve(ax, qx, x_curve, y_curve, color, tick_frac=0.02, alpha=0.9, lw=1.1):
    if len(qx) == 0 or len(x_curve) == 0:
        return

    y0, y1 = ax.get_ylim()
    tick_half = tick_frac * (y1 - y0)

    for q in qx:
        idx = np.argmin(np.abs(x_curve - q))
        yc = y_curve[idx]
        ax.plot([q, q], [yc - tick_half, yc + tick_half], color=color, alpha=alpha, linewidth=lw)


def plot_1d_term_on_ax(ax, gam, term_i, features, X_ref):
    term = gam.terms[term_i]
    feats = _term_features(term)
    if len(feats) != 1:
        return False

    fi = feats[0]
    name = pretty_feature_name(features[fi])

    if isinstance(term, FactorTerm):
        vals = np.unique(X_ref[:, fi])
        vals = vals[np.isfinite(vals)]
        vals = np.sort(vals)

        if len(vals) == 0:
            return False

        base = choose_baseline_vector(X_ref)
        Xp = np.tile(base, (len(vals), 1))
        Xp[:, fi] = vals
        pdep = np.asarray(gam.partial_dependence(term=term_i, X=Xp), dtype=float).ravel()

        labels = [str(int(v)) if float(v).is_integer() else str(v) for v in vals]
        if name.lower() == "sex":
            labels = ["F" if v == 0 else "M" if v == 1 else str(v) for v in vals]

        ax.bar(labels, pdep, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Term effect")
        return True

    XX = gam.generate_X_grid(term=term_i)

    try:
        pdep, confi = gam.partial_dependence(term=term_i, X=XX, width=0.95)
    except TypeError:
        pdep = gam.partial_dependence(term=term_i, X=XX)
        confi = None

    x = XX[:, fi]
    order = np.argsort(x)
    x = x[order]
    pdep = np.asarray(pdep)[order]

    line, = ax.plot(x, pdep, linewidth=2)

    if confi is not None:
        confi = np.asarray(confi)
        lower, upper = None, None
        if confi.ndim == 2 and confi.shape[0] == 2:
            lower, upper = confi[0][order], confi[1][order]
        elif confi.ndim == 2 and confi.shape[1] == 2:
            lower, upper = confi[:, 0][order], confi[:, 1][order]
        if lower is not None and len(lower) == len(x):
            ax.fill_between(x, lower, upper, alpha=0.2)

    try:
        qx = np.quantile(X_ref[:, fi][np.isfinite(X_ref[:, fi])], [0.05, 0.25, 0.5, 0.75, 0.95])
        draw_ticks_on_curve(ax, qx, x, pdep, color=line.get_color(), tick_frac=0.02, alpha=0.9)
    except Exception:
        pass

    ax.set_title(name, fontsize=10)
    ax.set_xlabel(name, fontsize=9)
    ax.set_ylabel("Term effect", fontsize=9)
    return True

def smooth_1d(arr, kernel_size=9):
    """
    Simple symmetric moving-average smoothing.
    kernel_size should be odd.
    """
    arr = np.asarray(arr, dtype=float)
    if kernel_size <= 1 or len(arr) == 0:
        return arr.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(kernel_size, dtype=float)
    kernel /= kernel.sum()

    padded = np.pad(arr, (kernel_size // 2,), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def estimate_weighted_density_1d(
    x_obs,
    w_obs=None,
    n_bins=120,
    smooth_kernel=9,
    x_min=None,
    x_max=None,
):
    """
    Estimate a weighted 1D density using a histogram + smoothing.
    Returns bin centers and normalized density in [0, 1].
    """
    x_obs = np.asarray(x_obs, dtype=float)

    if w_obs is None:
        w_obs = np.ones_like(x_obs, dtype=float)
    else:
        w_obs = np.asarray(w_obs, dtype=float)

    mask = np.isfinite(x_obs) & np.isfinite(w_obs) & (w_obs > 0)
    x_obs = x_obs[mask]
    w_obs = w_obs[mask]

    if len(x_obs) == 0:
        return np.array([]), np.array([])

    if x_min is None:
        x_min = float(np.nanmin(x_obs))
    if x_max is None:
        x_max = float(np.nanmax(x_obs))

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return np.array([]), np.array([])

    hist, edges = np.histogram(
        x_obs,
        bins=n_bins,
        range=(x_min, x_max),
        weights=w_obs,
        density=False,
    )

    hist = hist.astype(float)
    hist = smooth_1d(hist, kernel_size=smooth_kernel)

    centers = 0.5 * (edges[:-1] + edges[1:])

    if np.nanmax(hist) > 0:
        hist = hist / np.nanmax(hist)

    return centers, hist


def interpolate_density_to_curve(x_curve, density_x, density_y):
    """
    Interpolate estimated density onto the GAM curve x-grid.
    """
    x_curve = np.asarray(x_curve, dtype=float)
    density_x = np.asarray(density_x, dtype=float)
    density_y = np.asarray(density_y, dtype=float)

    if len(x_curve) == 0 or len(density_x) == 0:
        return np.zeros_like(x_curve, dtype=float)

    return np.interp(
        x_curve,
        density_x,
        density_y,
        left=0.0,
        right=0.0,
    )


def find_supported_segments(mask):
    """
    Convert a boolean mask into contiguous index segments.
    Returns list of (start_idx, end_idx_exclusive).
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or not np.any(mask):
        return []

    idx = np.flatnonzero(mask)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(g[0], g[-1] + 1) for g in groups if len(g) > 0]

# -----------------------------
# 2D term utilities
# -----------------------------
def build_2d_surface(gam, term_i, X_ref, n_grid=40):
    term = gam.terms[term_i]
    feats = _term_features(term)
    if len(feats) != 2:
        return None

    f1, f2 = feats
    base = choose_baseline_vector(X_ref)

    x1_vals = np.linspace(np.nanpercentile(X_ref[:, f1], 1), np.nanpercentile(X_ref[:, f1], 99), n_grid)
    x2_vals = np.linspace(np.nanpercentile(X_ref[:, f2], 1), np.nanpercentile(X_ref[:, f2], 99), n_grid)

    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Xflat = np.tile(base, (X1.size, 1))
    Xflat[:, f1] = X1.ravel()
    Xflat[:, f2] = X2.ravel()

    Z = gam.partial_dependence(term=term_i, X=Xflat).ravel().reshape(X1.shape)
    return X1, X2, Z


def plot_2d_term_on_ax(ax, gam, term_i, features, X_ref, vmin=None, vmax=None):
    surf = build_2d_surface(gam, term_i, X_ref, n_grid=40)
    if surf is None:
        return None

    term = gam.terms[term_i]
    f1, f2 = _term_features(term)
    name1 = pretty_feature_name(features[f1])
    name2 = pretty_feature_name(features[f2])

    X1, X2, Z = surf
    cf = ax.contourf(X1, X2, Z, levels=12, vmin=vmin, vmax=vmax)
    ax.set_title(f"{name1} × {name2}", fontsize=10)
    ax.set_xlabel(name1, fontsize=9)
    ax.set_ylabel(name2, fontsize=9)
    return cf


# -----------------------------
# Grouped physiology overlays
# -----------------------------

def plot_curve_with_density_fade(
    ax,
    x,
    y,
    x_obs,
    w_obs,
    label,
    color=None,
    lw=2.2,
    alpha_supported=0.95,
    alpha_unsupported=0.12,
    support_threshold=0.10,
    n_density_bins=120,
    smooth_kernel=9,
):
    """
    Plot one curve with strong alpha where weighted local support density
    exceeds a threshold, and faded alpha elsewhere.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_obs = np.asarray(x_obs, dtype=float)
    w_obs = np.asarray(w_obs, dtype=float)

    if len(x) == 0:
        return None

    if color is None:
        line, = ax.plot([], [])
        color = line.get_color()
        line.remove()

    density_x, density_y = estimate_weighted_density_1d(
        x_obs,
        w_obs=w_obs,
        n_bins=n_density_bins,
        smooth_kernel=smooth_kernel,
        x_min=float(np.nanmin(x)),
        x_max=float(np.nanmax(x)),
    )

    if len(density_x) == 0:
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha_supported, label=label)
        return color

    density_on_curve = interpolate_density_to_curve(x, density_x, density_y)
    support_mask = density_on_curve >= support_threshold

    supported_segments = find_supported_segments(support_mask)
    unsupported_segments = find_supported_segments(~support_mask)

    # draw unsupported first so supported sits on top
    for s0, s1 in unsupported_segments:
        ax.plot(
            x[s0:s1],
            y[s0:s1],
            color=color,
            linewidth=lw,
            alpha=alpha_unsupported,
        )

    label_drawn = False
    for s0, s1 in supported_segments:
        ax.plot(
            x[s0:s1],
            y[s0:s1],
            color=color,
            linewidth=lw,
            alpha=alpha_supported,
            label=label if not label_drawn else None,
        )
        label_drawn = True

    # if no supported segment survived threshold, draw whole curve faint with label
    if not label_drawn:
        ax.plot(
            x,
            y,
            color=color,
            linewidth=lw,
            alpha=alpha_unsupported,
            label=label,
        )

    return color

def plot_grouped_overlay_terms(
    ax,
    gam,
    features,
    X_support,
    raw_feature_names,
    title="",
    xlim=None,
    alpha_supported=0.95,
    alpha_unsupported=0.12,
    support_threshold=0.10,
    show_legend=True,
    w_support=None,
):
    plotted_any = False

    for feat in raw_feature_names:
        term_i = find_1d_term_index_for_feature(gam, features, feat)
        if term_i is None:
            continue

        x, y, x_obs, w_obs = get_1d_curve_data(
            gam,
            term_i,
            feat,
            features,
            X_support,
            w_support=w_support,
        )

        plot_curve_with_density_fade(
            ax,
            x,
            y,
            x_obs,
            w_obs,
            label=pretty_feature_name(feat),
            color=None,
            lw=2.2,
            alpha_supported=alpha_supported,
            alpha_unsupported=alpha_unsupported,
            support_threshold=support_threshold,
            n_density_bins=120,
            smooth_kernel=9,
        )
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No matching 1D terms found", ha="center", va="center")
        ax.set_title(title)
        return

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_title(title)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Term effect")

    if show_legend:
        ax.legend(fontsize=8, frameon=False)


def save_grouped_physiology_figure(gam, features, X_support, out_path, w_support=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    eeg_raw = [
        "EEG1_delta_rel",
        "EEG1_theta_rel",
        "EEG1_alpha_rel",
        "EEG1_beta_rel",
        "EEG1_gamma_rel",
    ]

    plot_grouped_overlay_terms(
        axes[0],
        gam,
        features,
        X_support,
        raw_feature_names=eeg_raw,
        title="EEG Bandpowers",
        xlim=(0, 1),
        alpha_supported=0.95,
        alpha_unsupported=0.10,
        support_threshold=0.10,
        show_legend=True,
        w_support=w_support,
    )

    plot_grouped_overlay_terms(
        axes[1],
        gam,
        features,
        X_support,
        raw_feature_names=["BIS/SEF"],
        title="SEF",
        alpha_supported=0.95,
        alpha_unsupported=0.10,
        support_threshold=0.10,
        show_legend=True,
        w_support=w_support,
    )

    plot_grouped_overlay_terms(
        axes[2],
        gam,
        features,
        X_support,
        raw_feature_names=["BIS/SR"],
        title="Suppression Ratio",
        alpha_supported=0.95,
        alpha_unsupported=0.10,
        support_threshold=0.10,
        show_legend=True,
        w_support=w_support,
    )

    plot_grouped_overlay_terms(
        axes[3],
        gam,
        features,
        X_support,
        raw_feature_names=["BIS/EMG"],
        title="EMG",
        alpha_supported=0.95,
        alpha_unsupported=0.10,
        support_threshold=0.10,
        show_legend=True,
        w_support=w_support,
    )

    fig.suptitle("Grouped 1D Physiology Terms (density-weighted support)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] grouped physiology -> {Path(out_path).resolve()}")


# -----------------------------
# Sorting
# -----------------------------
def mechanism_sort_key(mech):
    order = {
        "sr_mechanism": 0,
        "spectral_mechanism": 1,
        "sef_mechanism": 2,
        "EMG_mechanism": 3,
        "arousal_mechanism": 4,
        "statics": 5,
        "other": 6,
    }
    return order.get(mech, 99)


# -----------------------------
# Output builders
# -----------------------------
def save_performance_figure(y_true, y_pred, out_path):
    residuals = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    plot_true_vs_pred(axes[0], y_true, y_pred)
    plot_residual_hist(axes[1], residuals)
    plot_residual_vs_pred(axes[2], y_pred, residuals)
    plot_binned_calibration(axes[3], y_true, y_pred)

    fig.suptitle(
        f"GAM Performance Summary\nRMSE={rmse(y_true, y_pred):.3f} | corr²={r2_corr(y_true, y_pred):.3f} | N={len(y_true):,}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] performance -> {Path(out_path).resolve()}")


def save_top_1d_figure(gam, features, X_ref, term_df, out_path, max_terms=8):
    one_d = term_df[(term_df["n_features"] == 1) & (term_df["plot_priority"] == 1)].copy()
    one_d = one_d.sort_values(
        ["mechanism", "importance"],
        ascending=[True, False],
        key=lambda s: s.map(mechanism_sort_key) if s.name == "mechanism" else s,
    )

    top = one_d.head(max_terms).copy()
    if len(top) == 0:
        return

    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, top.iterrows()):
        plot_1d_term_on_ax(ax, gam, int(row["term_i"]), features, X_ref)
        ax.set_title(
            f"{row['label']}\n{row['mechanism']} | imp={row['importance']:.3f} | edof={row['edof']:.2f}",
            fontsize=9,
        )

    for j in range(len(top), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top 1D GAM Effects", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] top 1D -> {Path(out_path).resolve()}")


def save_top_2d_figure(gam, features, X_ref, term_df, out_path, max_terms=6):
    two_d = term_df[
        (term_df["n_features"] == 2) &
        (term_df["term_type"] == "te") &
        (term_df["plot_priority"] == 1)
    ].copy()

    two_d = two_d.sort_values(
        ["mechanism", "importance"],
        ascending=[True, False],
        key=lambda s: s.map(mechanism_sort_key) if s.name == "mechanism" else s,
    )

    top = two_d.head(max_terms).copy()
    if len(top) == 0:
        return

    zmins, zmaxs = [], []

    for _, row in top.iterrows():
        term_i = int(row["term_i"])
        surf = build_2d_surface(gam, term_i, X_ref, n_grid=40)
        if surf is None:
            continue
        _, _, Z = surf
        zmins.append(np.nanpercentile(Z, 2))
        zmaxs.append(np.nanpercentile(Z, 98))

    if not zmins or not zmaxs:
        return

    vmin = float(np.min(zmins))
    vmax = float(np.max(zmaxs))

    ncols = 2
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    last_cf = None
    for ax, (_, row) in zip(axes, top.iterrows()):
        term_i = int(row["term_i"])
        last_cf = plot_2d_term_on_ax(ax, gam, term_i, features, X_ref, vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{row['label']}\n{row['mechanism']} | imp={row['importance']:.3f} | edof={row['edof']:.2f}",
            fontsize=9,
        )

    for j in range(len(top), len(axes)):
        axes[j].axis("off")

    if last_cf is not None:
        fig.colorbar(last_cf, ax=axes[:len(top)], shrink=0.9)

    fig.suptitle("Top 2D Interaction Surfaces", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] top 2D -> {Path(out_path).resolve()}")


def save_term_summary_figure(term_df, out_path, max_lines=28):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    strong = term_df[term_df["plot_priority"] == 1].copy()
    weak = term_df[term_df["plot_priority"] == 0].copy()

    strong_lines = [
        "Plotted / higher-priority terms",
        "-" * 90,
    ]
    for _, row in strong.head(max_lines // 2).iterrows():
        strong_lines.append(
            f"{int(row['term_i']):>2d} | {row['term_type']:<2s} | {row['mechanism']:<20s} | "
            f"imp={row['importance']:.3f} | edof={row['edof']:.2f} | {row['label']}"
        )

    weak_lines = [
        "",
        "Low-impact / not plotted individually",
        "-" * 90,
    ]
    for _, row in weak.head(max_lines).iterrows():
        weak_lines.append(
            f"{int(row['term_i']):>2d} | {row['term_type']:<2s} | {row['mechanism']:<20s} | "
            f"imp={row['importance']:.3f} | edof={row['edof']:.2f} | {row['label']}"
        )

    txt = "\n".join(strong_lines + weak_lines)
    ax.text(0.01, 0.99, txt, va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("GAM Term Summary", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] term summary figure -> {Path(out_path).resolve()}")


def save_term_summary_csv(term_df, out_path):
    term_df.copy().to_csv(out_path, index=False)
    print(f"[SAVE] term summary csv -> {Path(out_path).resolve()}")


def build_dashboard_outputs(
    gam,
    features,
    restrictors,
    X_test,
    y_test,
    out_prefix="gam_dashboard",
    X_support=None,
    w_support=None,
):
    y_pred = gam.predict(X_test)

    term_df = make_term_summary(gam, X_test, features, restrictors, max_rows_eval=4000)

    # fallback if no weighted support sample is stored in the bundle
    if X_support is None:
        X_support = X_test
    if w_support is not None and len(w_support) != len(X_support):
        w_support = None

    save_term_summary_csv(term_df, f"{out_prefix}_term_summary.csv")
    save_performance_figure(y_test, y_pred, f"{out_prefix}_performance.png")
    save_grouped_physiology_figure(
        gam,
        features,
        X_support,
        f"{out_prefix}_grouped_physiology.png",
        w_support=w_support,
    )
    save_top_1d_figure(gam, features, X_test, term_df, f"{out_prefix}_top_1d.png", max_terms=8)
    save_top_2d_figure(gam, features, X_test, term_df, f"{out_prefix}_top_2d.png", max_terms=6)
    save_term_summary_figure(term_df, f"{out_prefix}_term_summary.png", max_lines=24)

    print("\n[INFO] Top terms by importance:")
    cols = ["term_i", "term_type", "mechanism", "importance", "edof", "label"]
    print(term_df[cols].head(15).to_string(index=False))

    print("\n[INFO] Low-impact terms not plotted individually:")
    weak = term_df[term_df["plot_priority"] == 0][cols]
    if len(weak):
        print(weak.head(20).to_string(index=False))
    else:
        print("None")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    gam, features, restrictors, X_support, w_support = load_model_bundle(MODEL_PATH)

    print("\n[DEBUG] FEATURES from loaded bundle:")
    for f in features:
        if "Solar8000/BT" in f:
            print(" ", f)

    print("\n[DEBUG] Restrictors from loaded bundle:")
    print(restrictors)

    clinical_map = load_clinical_info_map(("age", "sex", "lbm"))

    test_case_ids = collect_case_ids("Testing")
    X_test, y_test = load_split_arrays(test_case_ids, features, clinical_map)
    X_test, y_test = filter_finite(X_test, y_test)

    print(f"[INFO] Loaded test rows: {len(y_test):,}")
    build_dashboard_outputs(
        gam,
        features,
        restrictors,
        X_test,
        y_test,
        out_prefix="gam_dashboard",
        X_support=X_support,
        w_support=w_support,
    )
