"""
Build a Causalindex, using a three-curve Expectile GAM system
"""

import os
import csv
import joblib
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Set

from pygam import ExpectileGAM, s, f, te
from pygam.utils import flatten


DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

MAX_ROWS_PER_BIS_BUCKET = 50_000
RNG_SEED = 13

STATIC_FEATURES = ["age", "sex", "lbm"]
EXPECTILES = [0.1, 0.5, 0.9]

#Temporal Picks decides which timedata is used for each Track
TEMPORAL_PICK = {
    # EEG bands
    "EEG1_alpha_rel": ["raw", "slope30"],
    "EEG1_beta_rel":  ["raw", "slope30"],
    "EEG1_theta_rel": ["raw", "slope30"],
    "EEG1_delta_rel": ["raw", "slope30"],
    "EEG1_gamma_rel": ["raw", "slope30"],

    # BIS derived
    "BIS/SEF": ["raw", "slope30"],
    "BIS/EMG": ["raw", "slope30"],
    "BIS/SR": ["raw"],

    # vitals
    "Solar8000/HR": ["raw"],
    "Solar8000/ART_MBP": ["raw"],
	"Solar8000/ETCO2": ["raw"],
    "Solar8000/BT": ["raw"],
}

#Restrictors builds the mains and tensors that are allowed to have influence on the Target

RESTRICTORS = {
    "forbid_main": {"Solar8000/BT", "lbm"},
    "mechanisms": {
        "sr_mechanism": {
            "main": {"BIS/SR"},
            "mods": {"age"},
        },
        "spectral_mechanism": {
            "main": {
                "EEG1_alpha_rel",
                "EEG1_delta_rel",
                "EEG1_theta_rel",
                "EEG1_beta_rel",
                "EEG1_gamma_rel",
            },
            "mods": {"Solar8000/BT", "age", "sex"},
        },
        "sef_mechanism": {
            "main": {"BIS/SEF"},
            "mods": {"age", "Solar8000/BT"},
        },
        "EMG_mechanism": {
            "main": {"BIS/EMG"},
            "mods": {"Solar8000/HR", "Solar8000/ART_MBP", "Solar8000/ETCO2"},
        },
    },
}

#Set how many Splines are allowed for a Track (Less Splines if relation is expected: linear)
NODE_N_SPLINES_MAIN = {
    "BIS/SR": 10,
    "EEG1_alpha_rel": 10,
    "EEG1_beta_rel": 10,
    "EEG1_theta_rel": 10,
    "EEG1_delta_rel": 10,
    "EEG1_gamma_rel": 10,

    # keep simpler / closer to linear
    "BIS/SEF": 5,
    "BIS/EMG": 5,
    "Solar8000/HR": 5,
    "Solar8000/ART_MBP": 5,
    "age": 5,
}

# custom complexity for modifiers
NODE_N_SPLINES_MOD = {
    "age": 4,
    "sex": 4,
    "Solar8000/HR": 4,
    "Solar8000/ART_MBP": 4,
}
#Default Spline Counts
DEFAULT_N_SPLINES_MAIN = 6
DEFAULT_N_SPLINES_MOD = 4

BIS_EDGES = np.array(
    [16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101],
    dtype=float,
)
N_BINS = len(BIS_EDGES) - 1

TARGET_COL = "BIS/BIS"


# -----------------------------
# Data split + masking
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


# -----------------------------
# Static features
# -----------------------------
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

def make_bis_bins(y):
	"""
	Create BIS Buckets
	"""
	y_num = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
	b = np.digitize(y_num, BIS_EDGES, right=False) - 1
	b[np.isnan(y_num)] = -1
	b[(y_num < BIS_EDGES[0]) | (y_num >= BIS_EDGES[-1])] = -1
	return b.astype(int)



def scan_case_bucket_counts(case_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    For each case, count how many masked rows fall into each BIS bucket.
    """
    out: Dict[str, np.ndarray] = {}

    for cid in case_ids:
        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_mask(df)]
        if len(df) == 0:
            continue

        bins = make_bis_bins(df[TARGET_COL].to_numpy())
        valid = bins >= 0
        if not np.any(valid):
            continue

        counts = np.bincount(bins[valid], minlength=N_BINS).astype(int)
        out[cid] = counts

    return out


def allocate_global_bucket_quotas(
    case_bucket_counts: Dict[str, np.ndarray],
    *,
    max_rows_per_bucket: int = MAX_ROWS_PER_BIS_BUCKET,
) -> Dict[str, np.ndarray]:
    """
    Short: Downsample BIS Buckets across all cases evenly

    For each bucket:
      - if total rows <= cap, every case keeps all its rows
      - otherwise allocate quotas proportional to each case's contribution
        using largest-remainder rounding
    """
    case_ids = sorted(case_bucket_counts.keys())
    if not case_ids:
        return {}

    total_per_bucket = np.zeros(N_BINS, dtype=int)
    for cid in case_ids:
        total_per_bucket += case_bucket_counts[cid]

    quota_map: Dict[str, np.ndarray] = {
        cid: np.zeros(N_BINS, dtype=int) for cid in case_ids
    }

    for b in range(N_BINS):
        total_b = int(total_per_bucket[b])
        if total_b == 0:
            continue

        counts_b = {cid: int(case_bucket_counts[cid][b]) for cid in case_ids}
        active = [cid for cid in case_ids if counts_b[cid] > 0]

        if total_b <= max_rows_per_bucket:
            for cid in active:
                quota_map[cid][b] = counts_b[cid]
            continue

        # proportional quota
        raw = {
            cid: max_rows_per_bucket * (counts_b[cid] / total_b)
            for cid in active
        }

        base = {
            cid: min(counts_b[cid], int(np.floor(raw[cid])))
            for cid in active
        }

        assigned = sum(base.values())
        remainder = max_rows_per_bucket - assigned

        # largest remainder method, while never exceeding original count
        order = sorted(
            active,
            key=lambda cid: (raw[cid] - np.floor(raw[cid]), counts_b[cid]),
            reverse=True,
        )

        i = 0
        while remainder > 0 and order:
            cid = order[i % len(order)]
            if base[cid] < counts_b[cid]:
                base[cid] += 1
                remainder -= 1
            i += 1

            # safety stop if nobody can receive more
            if i > 10 * len(order) and all(base[x] >= counts_b[x] for x in order):
                break

        for cid in active:
            quota_map[cid][b] = base[cid]

    return quota_map


def print_global_bucket_cap_summary(
    case_bucket_counts: Dict[str, np.ndarray],
    case_quota_map: Dict[str, np.ndarray],
):
    total_before = np.zeros(N_BINS, dtype=int)
    total_after = np.zeros(N_BINS, dtype=int)

    for cid, counts in case_bucket_counts.items():
        total_before += counts
        total_after += case_quota_map.get(cid, np.zeros(N_BINS, dtype=int))

    print("\n[INFO] Global BIS bucket cap summary:")
    for b in range(N_BINS):
        lo, hi = BIS_EDGES[b], BIS_EDGES[b + 1]
        print(
            f"  [{lo:>5.1f}, {hi:>5.1f}): "
            f"before={int(total_before[b]):>8d}  after={int(total_after[b]):>8d}"
        )


def sample_case_by_bucket_quota(
    df: pd.DataFrame,
    case_bucket_quota: np.ndarray,
    *,
    target_col: str = TARGET_COL,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample rows from one case so that for each BIS bucket b,
    exactly up to case_bucket_quota[b] rows are retained.
    """
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    bins = make_bis_bins(y)

    selected_idx = []

    for b in range(N_BINS):
        q = int(case_bucket_quota[b])
        if q <= 0:
            continue

        idx_b = np.flatnonzero(bins == b)
        if len(idx_b) == 0:
            continue

        if len(idx_b) <= q:
            chosen = idx_b
        else:
            chosen = rng.choice(idx_b, size=q, replace=False)

        selected_idx.append(np.sort(chosen))

    if not selected_idx:
        return df.iloc[0:0].copy()

    selected_idx = np.concatenate(selected_idx)
    selected_idx.sort()
    return df.iloc[selected_idx].copy()


# Feature list builder

def build_feature_list(
    temporal_pick: Dict[str, List[str]],
    static_features: List[str],
    *,
    include_target: bool = False,
    target_col: str = TARGET_COL,
) -> List[str]:
    features: List[str] = []

    for node, suffixes in temporal_pick.items():
        if suffixes is None:
            continue

        for suf in suffixes:
            if suf is None:
                continue
            suf = str(suf).strip()

            if suf == "" or suf.lower() == "raw":
                col = node
            else:
                col = f"{node}_{suf}"

            if col not in features:
                features.append(col)

    for sname in static_features:
        if sname not in features:
            features.append(sname)

    if include_target and target_col not in features:
        features.append(target_col)

    return features


def scan_training_stats(
    case_ids: List[str],
    case_quota_map: Dict[str, np.ndarray],
    *,
    rng_seed: int = RNG_SEED,
) -> Tuple[Dict[str, int], np.ndarray]:
    case_len_map: Dict[str, int] = {}
    bin_counts = np.zeros(N_BINS, dtype=int)
    rng = np.random.default_rng(rng_seed)

    for cid in case_ids:
        quota = case_quota_map.get(cid)
        if quota is None:
            continue

        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_mask(df)].copy()
        if len(df) == 0:
            continue

        df = sample_case_by_bucket_quota(df, quota, rng=rng)
        n = len(df)
        if n == 0:
            continue

        case_len_map[cid] = n

        bins = make_bis_bins(df[TARGET_COL].to_numpy())
        valid = bins >= 0
        counts = np.bincount(bins[valid], minlength=N_BINS)
        bin_counts += counts

    return case_len_map, bin_counts


def calculate_bin_weight(bin_counts, min_count=200, alpha=0.75):
    freq = bin_counts.astype(float)
    freq[freq < min_count] = min_count
    return 1.0 / np.power(freq, alpha)


def print_weight_mass_by_bin(y, w, edges):
    bins = np.digitize(y, edges, right=False) - 1
    valid = (bins >= 0) & (bins < len(edges) - 1)

    print("\n[INFO] Weight mass by BIS bin:")
    for b in range(len(edges) - 1):
        m = valid & (bins == b)
        count = int(m.sum())
        mass = float(w[m].sum()) if count > 0 else 0.0
        lo, hi = edges[b], edges[b + 1]
        print(f"  [{lo:>5.1f}, {hi:>5.1f}): count={count:>8d}  weight_sum={mass:>10.4f}")


# -----------------------------
# Array loaders
# -----------------------------
def load_training_arrays(
    case_ids: List[str],
    features: List[str],
    case_len_map: Dict[str, int],
    bin_w_lookup: np.ndarray,
    clinical_map: Dict[str, Dict[str, str]],
    case_quota_map: Dict[str, np.ndarray],
    *,
    rng_seed: int = RNG_SEED,
):
    X_list, y_list, w_list = [], [], []
    rng = np.random.default_rng(rng_seed)

    for cid in case_ids:
        if cid not in case_len_map:
            continue

        quota = case_quota_map.get(cid)
        if quota is None:
            continue

        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_mask(df)].copy()
        if len(df) == 0:
            continue

        df = sample_case_by_bucket_quota(df, quota, rng=rng)
        if len(df) == 0:
            continue

        df = add_static_features(df, cid, clinical_map)
        df = coerce_static_types(df)

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Case {cid} missing feature columns: {missing}")

        X = df[features].to_numpy(dtype=np.float32)
        y = df[TARGET_COL].to_numpy(dtype=np.float32)

        w_case = np.full(len(df), 1.0 / max(case_len_map[cid], 1), dtype=np.float32)

        bins = make_bis_bins(y)
        valid = bins >= 0

        if not np.all(valid):
            X = X[valid]
            y = y[valid]
            w_case = w_case[valid]
            bins = bins[valid]

        w_bin = bin_w_lookup[bins].astype(np.float32)
        w = w_case * w_bin

        X_list.append(X)
        y_list.append(y)
        w_list.append(w)

    if not X_list:
        raise RuntimeError("No training data loaded. Check mask, paths, and columns.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    w_all = np.concatenate(w_list)

    w_all = w_all * (len(w_all) / w_all.sum())
    return X_all, y_all, w_all


def load_split_arrays(
    case_ids: List[str],
    features: List[str],
    clinical_map: Dict[str, Dict[str, str]],
):
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
        raise RuntimeError("No data loaded. Check mask, paths, and columns.")

    return np.vstack(X_list), np.concatenate(y_list)


def filter_finite(X: np.ndarray, y: np.ndarray, w: np.ndarray = None):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if w is None:
        return X[mask], y[mask]
    return X[mask], y[mask], w[mask]


# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def print_metrics(name, y_true, y_pred):
    print(f"[METRIC] {name} RMSE: {rmse(y_true, y_pred):.4f}")


def print_rmse_by_bis_bin(name, y_true, y_pred, edges):
    print(f"\n[INFO] {name} RMSE by BIS bin:")
    bins = np.digitize(y_true, edges, right=False) - 1
    for b in range(len(edges) - 1):
        m = bins == b
        if m.sum() == 0:
            continue
        lo, hi = edges[b], edges[b + 1]
        print(f"  [{lo:>5.1f}, {hi:>5.1f}): n={m.sum():>7d}  rmse={rmse(y_true[m], y_pred[m]):.4f}")


# -----------------------------
# Restrictors -> GAM terms
# -----------------------------
def build_used_columns_and_terms(
    features_all: List[str],
    temporal_pick: Dict[str, List[str]],
    restrictors: Dict,
    categorical_nodes: Set[str] = None,
    node_n_splines_main: Dict[str, int] = None,
    node_n_splines_mod: Dict[str, int] = None,
    default_n_splines_main: int = DEFAULT_N_SPLINES_MAIN,
    default_n_splines_mod: int = DEFAULT_N_SPLINES_MOD,
):
    if categorical_nodes is None:
        categorical_nodes = {"sex"}

    if node_n_splines_main is None:
        node_n_splines_main = {}

    if node_n_splines_mod is None:
        node_n_splines_mod = {}

    forbid_main = set(restrictors.get("forbid_main", set()))
    mechs = restrictors.get("mechanisms", {})

    node_to_cols = {}

    referenced_nodes: Set[str] = set()
    for spec in mechs.values():
        referenced_nodes |= set(spec.get("main", set()))
        referenced_nodes |= set(spec.get("mods", set()))

    referenced_nodes |= {"age", "sex", "lbm"}

    for node in referenced_nodes:
        if node in {"age", "sex", "lbm"}:
            node_to_cols[node] = [node]
            continue

        suffixes = temporal_pick.get(node, [])
        cols = []

        for suf in suffixes:
            if suf == "raw" or suf == "":
                col = node
            else:
                col = f"{node}_{suf}"
            cols.append(col)

        if not cols:
            raise KeyError(f"No TEMPORAL_PICK entry for node '{node}'")

        node_to_cols[node] = cols

    used_cols = []
    for cols in node_to_cols.values():
        for c in cols:
            if c not in used_cols:
                used_cols.append(c)

    missing = [c for c in used_cols if c not in features_all]
    if missing:
        raise KeyError(f"Columns needed by RESTRICTORS missing in FEATURES: {missing}")

    features_used = used_cols
    col_to_idx = {c: i for i, c in enumerate(features_used)}

    terms = None

    for mech_name, spec in mechs.items():
        mains = sorted(spec.get("main", set()))
        mods = sorted(spec.get("mods", set()))

        for main_node in mains:
            main_cols = node_to_cols[main_node]

            for main_col in main_cols:
                main_i = col_to_idx[main_col]

                main_n_splines = node_n_splines_main.get(main_node, default_n_splines_main)

                if main_node in categorical_nodes:
                    t = f(main_i)
                else:
                    if main_node in forbid_main:
                        t = None
                    else:
                        t = s(main_i, n_splines=main_n_splines)

                if t is not None:
                    terms = t if terms is None else (terms + t)

                for mod_node in mods:
                    mod_cols = node_to_cols[mod_node]

                    for mod_col in mod_cols:
                        mod_i = col_to_idx[mod_col]
                        mod_n_splines = node_n_splines_mod.get(mod_node, default_n_splines_mod)

                        t2 = te(
                            main_i,
                            mod_i,
                            n_splines=[main_n_splines, mod_n_splines],
                        )
                        terms = t2 if terms is None else (terms + t2)

    if terms is None:
        raise RuntimeError("No GAM terms created")

    return features_used, terms, node_to_cols


def print_mechanism_term_plan_multi(
    features_used: List[str],
    node_to_cols: Dict[str, List[str]],
    restrictors: Dict,
):
    col_to_idx = {c: i for i, c in enumerate(features_used)}
    forbid_main = set(restrictors.get("forbid_main", set()))
    mechs = restrictors.get("mechanisms", {})

    print("\n[INFO] MECHANISM TERM PLAN:")
    for mech_name, spec in mechs.items():
        mains = sorted(spec.get("main", set()))
        mods = sorted(spec.get("mods", set()))
        print(f"\n  [{mech_name}]")

        for m in mains:
            mcols = node_to_cols[m]
            for mcol in mcols:
                midx = col_to_idx[mcol]
                if m in forbid_main:
                    print(f"    (NO MAIN) {mcol} (forbidden via node {m})")
                else:
                    print(f"    MAIN      s({midx}) on {mcol}")

                for z in mods:
                    zcols = node_to_cols[z]
                    for zcol in zcols:
                        zidx = col_to_idx[zcol]
                        print(f"    INTERACT  te({midx},{zidx}) on ({mcol}, {zcol})")


# -----------------------------
# Stratified subset for lambda tuning
# -----------------------------
def stratified_bis_subset(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    subset_size: int,
    rng: np.random.Generator,
):
    bins = make_bis_bins(y)
    valid = bins >= 0

    X = X[valid]
    y = y[valid]
    w = w[valid]
    bins = bins[valid]

    if len(y) <= subset_size:
        return X, y, w

    per_bin_target = max(1, subset_size // N_BINS)
    selected = []

    for b in range(N_BINS):
        idx_b = np.flatnonzero(bins == b)
        if len(idx_b) == 0:
            continue

        take = min(per_bin_target, len(idx_b))
        chosen = rng.choice(idx_b, size=take, replace=False)
        selected.append(chosen)

    selected = np.concatenate(selected) if selected else np.array([], dtype=int)

    if len(selected) < subset_size:
        remaining = np.setdiff1d(np.arange(len(y)), selected, assume_unique=False)
        need = min(subset_size - len(selected), len(remaining))
        if need > 0:
            extra = rng.choice(remaining, size=need, replace=False)
            selected = np.concatenate([selected, extra])

    selected = np.unique(selected)

    if len(selected) > subset_size:
        selected = rng.choice(selected, size=subset_size, replace=False)

    return X[selected], y[selected], w[selected]


# -----------------------------
# Three-curve expectile helpers
# -----------------------------
def fit_expectile_model(
    terms,
    expectile: float,
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    w_sub: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    *,
    rng_seed: int,
    n_draws: int = 50,
    lam_low: float = 1e-7,
    lam_high: float = 1e2,
    best_lambda=None,
):
    if best_lambda is None:
        rng = np.random.default_rng(rng_seed)

        print(f"[INFO] Running lambda grid search on subset for tau={expectile:.1f}...")
        gam = ExpectileGAM(terms, expectile=expectile)

        n_lams = len(flatten(gam.lam))
        lams = np.exp(rng.uniform(np.log(lam_low), np.log(lam_high), size=(n_draws, n_lams)))

        gam.gridsearch(
            X_sub,
            y_sub,
            weights=w_sub,
            lam=lams,
        )

        best_lambda = gam.lam
        print(f"[INFO] Best lambda found for tau={expectile:.1f}: {best_lambda}")
    else:
        print(f"[INFO] Reusing lambda for tau={expectile:.1f}: {best_lambda}")

    gam_final = ExpectileGAM(terms, expectile=expectile, lam=best_lambda)
    print(f"[INFO] Training final model on full dataset for tau={expectile:.1f}...")
    gam_final.fit(X_train, y_train, weights=w_train)

    return gam_final, best_lambda


def predict_ordered_expectiles(models: Dict[float, ExpectileGAM], X: np.ndarray) -> Dict[float, np.ndarray]:
    taus = sorted(models.keys())
    pred_stack = np.vstack([models[tau].predict(X) for tau in taus])

    # enforce monotone ordering across expectiles
    pred_stack = np.maximum.accumulate(pred_stack, axis=0)

    return {tau: pred_stack[i] for i, tau in enumerate(taus)}

def print_missing_drivers_by_bis_bin(X: np.ndarray, y: np.ndarray, features: List[str], edges, top_k: int = 8):
    bins = np.digitize(y, edges, right=False) - 1
    print("\n[INFO] Missing-value drivers by BIS bin:")
    for b in range(len(edges) - 1):
        m = bins == b
        if not np.any(m):
            continue

        Xb = X[m]
        bad_rows = ~np.all(np.isfinite(Xb), axis=1)
        n_total = int(m.sum())
        n_bad = int(bad_rows.sum())

        lo, hi = edges[b], edges[b + 1]
        print(f"\n  [{lo:>5.1f}, {hi:>5.1f}): total={n_total:>8d}  rows_dropped_by_X={n_bad:>8d}")

        if n_bad == 0:
            continue

        miss_rate = np.mean(~np.isfinite(Xb[bad_rows]), axis=0)
        top_idx = np.argsort(-miss_rate)[:top_k]

        for j in top_idx:
            if miss_rate[j] <= 0:
                continue
            print(f"    {features[j]:30s} missing_in_dropped_rows={miss_rate[j]:.3f}")
            
# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
	FEATURES_ALL = build_feature_list(TEMPORAL_PICK, STATIC_FEATURES)

	FEATURES_USED, TERMS, NODE_TO_COL = build_used_columns_and_terms(
		FEATURES_ALL,
		TEMPORAL_PICK,
		RESTRICTORS,
		categorical_nodes={"sex"},
		node_n_splines_main=NODE_N_SPLINES_MAIN,
		node_n_splines_mod=NODE_N_SPLINES_MOD,
		default_n_splines_main=DEFAULT_N_SPLINES_MAIN,
		default_n_splines_mod=DEFAULT_N_SPLINES_MOD,
	)

	print("\n[INFO] NODE -> COLUMNS:")
	for node in sorted(NODE_TO_COL.keys()):
		print(f"  {node:20s} -> {NODE_TO_COL[node]}")

	print(f"\n[INFO] FEATURES_USED by GAM ({len(FEATURES_USED)}):")
	for i, col in enumerate(FEATURES_USED):
		print(f"  {i:02d} -> {col}")

	print_mechanism_term_plan_multi(FEATURES_USED, NODE_TO_COL, RESTRICTORS)

	clinical_map = load_clinical_info_map(STATIC_FEATURES)

	train_case_ids = collect_case_ids("Training")
	val_case_ids = collect_case_ids("Validation")
	test_case_ids = collect_case_ids("Testing")

	print("[INFO] Scanning per-case BIS bucket counts...")
	case_bucket_counts = scan_case_bucket_counts(train_case_ids)

	print("[INFO] Allocating global per-bucket quotas...")
	case_quota_map = allocate_global_bucket_quotas(
		case_bucket_counts,
		max_rows_per_bucket=MAX_ROWS_PER_BIS_BUCKET,
	)
	print_global_bucket_cap_summary(case_bucket_counts, case_quota_map)

	case_len_map, bin_counts = scan_training_stats(
		train_case_ids,
		case_quota_map,
		rng_seed=RNG_SEED,
	)

	bin_w_lookup = calculate_bin_weight(bin_counts, min_count=200, alpha=1.0)

	X_train, y_train, w_train = load_training_arrays(
		train_case_ids,
		FEATURES_USED,
		case_len_map,
		bin_w_lookup,
		clinical_map,
		case_quota_map,
		rng_seed=RNG_SEED,
	)
	X_val, y_val = load_split_arrays(val_case_ids, FEATURES_USED, clinical_map)
	X_test, y_test = load_split_arrays(test_case_ids, FEATURES_USED, clinical_map)
	print_missing_drivers_by_bis_bin(X_train, y_train, FEATURES_USED, BIS_EDGES)
	X_train, y_train, w_train = filter_finite(X_train, y_train, w_train)
	X_val, y_val = filter_finite(X_val, y_val)
	X_test, y_test = filter_finite(X_test, y_test)

	print(f"\n[INFO] Rows: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")
	print_weight_mass_by_bin(y_train, w_train, BIS_EDGES)

	SUBSET_SIZE = 200_000
	rng = np.random.default_rng(RNG_SEED)
	X_sub, y_sub, w_sub = stratified_bis_subset(X_train, y_train, w_train, SUBSET_SIZE, rng)
	print(f"[INFO] Using BIS-stratified subset of {len(y_sub):,} rows for lambda tuning")

	models: Dict[float, ExpectileGAM] = {}
	best_lambdas: Dict[float, object] = {}

	# Run gridsearch only once for tau=0.5
	model_05, best_lambda_05 = fit_expectile_model(
		TERMS,
		0.5,
		X_sub,
		y_sub,
		w_sub,
		X_train,
		y_train,
		w_train,
		rng_seed=RNG_SEED + int(round(1000 * 0.5)),
		n_draws=50,
		lam_low=1e-7,
		lam_high=1e2,
	)

	models[0.5] = model_05
	best_lambdas[0.5] = best_lambda_05

	# Reuse the same lambda for the other expectiles
	for tau in [0.1, 0.9]:
		model, best_lambda = fit_expectile_model(
			TERMS,
			tau,
			X_sub,
			y_sub,
			w_sub,
			X_train,
			y_train,
			w_train,
			rng_seed=RNG_SEED + int(round(1000 * tau)),
			n_draws=50,
			lam_low=1e-7,
			lam_high=1e2,
			best_lambda=best_lambda_05,
		)
		models[tau] = model
		best_lambdas[tau] = best_lambda

	rng = np.random.default_rng(RNG_SEED)
	support_n = min(100000, len(X_train))
	support_idx = rng.choice(len(X_train), size=support_n, replace=False)

	bundle = {
		"models": models,
		"best_lambdas": best_lambdas,
		"expectiles": EXPECTILES,
		"features_used": FEATURES_USED,
		"temporal_pick": TEMPORAL_PICK,
		"restrictors": RESTRICTORS,
		"version": 4,
		"X_support_sample": X_train[support_idx],
		"w_support_sample": w_train[support_idx],
	}
	joblib.dump(bundle, "bis_causal_expectilegam.pkl")
	print("[INFO] Model bundle saved")

	y_val_pred_map = predict_ordered_expectiles(models, X_val)
	y_test_pred_map = predict_ordered_expectiles(models, X_test)

	print_metrics("VAL tau=0.5", y_val, y_val_pred_map[0.5])
	print_metrics("TEST tau=0.5", y_test, y_test_pred_map[0.5])

	print_rmse_by_bis_bin("VAL tau=0.5", y_val, y_val_pred_map[0.5], BIS_EDGES)
	print_rmse_by_bis_bin("TEST tau=0.5", y_test, y_test_pred_map[0.5], BIS_EDGES)

	val_band_coverage = np.mean(
		(y_val >= y_val_pred_map[0.1]) & (y_val <= y_val_pred_map[0.9])
	)
	test_band_coverage = np.mean(
		(y_test >= y_test_pred_map[0.1]) & (y_test <= y_test_pred_map[0.9])
	)

	print(f"\n[METRIC] VAL 10-90 band coverage: {val_band_coverage:.4f}")
	print(f"[METRIC] TEST 10-90 band coverage: {test_band_coverage:.4f}")

	for tau in EXPECTILES:
		try:
			print(f"\n[INFO] Summary for tau={tau:.1f}")
			print("\n" + models[tau].summary())
		except Exception:
			pass
