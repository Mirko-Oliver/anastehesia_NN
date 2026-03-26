import os
import math
import numpy as np
import pandas as pd

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"

TIME_COL = "Time"
EMG_COL = "BIS/EMG"
BIS_COL = "BIS/BIS"

DERIV_COL = "emg_derivative"
LABEL_COL = "label"

LABEL_INVALID = "invalid"
LABEL_IRREGULAR = "irregular"
LABEL_REGULAR = "regular"

# -----------------------------
# Global configuration
# -----------------------------
BIS_BUCKET_SIZE = 10          # Example: 10 -> buckets 1-10, 11-20, ..., 91-100
TOP_PERCENT_PER_BUCKET = 5.0  # Label top 5% per BIS bucket as irregular

EXTEND_SEC = 30               # Extend only irregular labels forward by this many seconds
MIN_REGULAR_RUN_SEC = 30      # Since sampling is 1 Hz, this is also 30 seconds


def discover_case_ids(data_dir=DATA_DIR, suffix=SUFFIX):
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(data_dir)
        if f.endswith(suffix)
    )


def build_bis_buckets(bucket_size: int):
    """
    Returns bucket definitions as list of (start, end, label).
    Example for bucket_size=10:
      (1, 10, '1-10'), (11, 20, '11-20'), ...
    """
    if bucket_size <= 0:
        raise ValueError("BIS_BUCKET_SIZE must be > 0")
    if bucket_size > 100:
        raise ValueError("BIS_BUCKET_SIZE must be <= 100")

    buckets = []
    start = 1
    while start <= 100:
        end = min(start + bucket_size - 1, 100)
        buckets.append((start, end, f"{start}-{end}"))
        start += bucket_size

    return buckets


def print_bucket_info(bucket_size: int):
    buckets = build_bis_buckets(bucket_size)
    print("Created BIS buckets:")
    for start, end, label in buckets:
        print(f"  {label} (edges: {start} -> {end})")
    print()


def assign_bis_buckets(bis_values: np.ndarray, bucket_size: int) -> np.ndarray:
    """
    Assign each BIS value to a bucket label.
    Values outside [1, 100] or NaN get None.
    """
    buckets = build_bis_buckets(bucket_size)
    out = np.full(len(bis_values), None, dtype=object)

    finite_mask = np.isfinite(bis_values)
    in_range_mask = finite_mask & (bis_values >= 1) & (bis_values <= 100)

    for start, end, label in buckets:
        mask = in_range_mask & (bis_values >= start) & (bis_values <= end)
        out[mask] = label

    return out


def extend_mask_forward(t: np.ndarray, mask: np.ndarray, seconds: float = EXTEND_SEC):
    """
    For every True at index i, set subsequent indices True while t[j] <= t[i] + seconds.
    Assumes t is sorted ascending.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask

    seconds = float(seconds)
    if seconds <= 0:
        return mask

    out = mask.copy()
    true_idx = np.flatnonzero(mask)
    if true_idx.size == 0:
        return out

    n = len(t)
    for i in true_idx:
        if not np.isfinite(t[i]):
            continue
        t_end = t[i] + seconds
        j = i + 1
        while j < n and np.isfinite(t[j]) and t[j] <= t_end:
            out[j] = True
            j += 1

    return out


def relabel_short_regular_runs(labels: np.ndarray, min_len: int = MIN_REGULAR_RUN_SEC):
    """
    Relabel short regular runs to irregular, but only if they are sandwiched
    between irregular regions. Invalid regions are not bridged.
    """
    labels = labels.copy()
    n = len(labels)
    i = 0

    while i < n:
        if labels[i] != LABEL_REGULAR:
            i += 1
            continue

        start = i
        while i < n and labels[i] == LABEL_REGULAR:
            i += 1
        end = i  # exclusive

        run_len = end - start
        left_label = labels[start - 1] if start > 0 else None
        right_label = labels[end] if end < n else None

        if (
            run_len < min_len
            and left_label == LABEL_IRREGULAR
            and right_label == LABEL_IRREGULAR
        ):
            labels[start:end] = LABEL_IRREGULAR

    return labels


def select_top_positive_derivative_per_bucket(
    dy_dt: np.ndarray,
    bis_bucket: np.ndarray,
    is_invalid: np.ndarray,
    top_percent: float
) -> np.ndarray:
    """
    Mark the top X% of datapoints in each BIS bucket as irregular,
    based only on the highest positive EMG derivatives.
    """
    if not (0 <= top_percent <= 100):
        raise ValueError("TOP_PERCENT_PER_BUCKET must be between 0 and 100")

    irregular = np.zeros(len(dy_dt), dtype=bool)

    valid_candidates = (
        (~is_invalid) &
        np.isfinite(dy_dt) &
        (dy_dt > 0) &
        pd.notna(bis_bucket)
    )

    if not np.any(valid_candidates) or top_percent == 0:
        return irregular

    unique_buckets = [b for b in pd.unique(bis_bucket) if b is not None]

    for bucket in unique_buckets:
        bucket_mask = valid_candidates & (bis_bucket == bucket)
        bucket_idx = np.flatnonzero(bucket_mask)

        if bucket_idx.size == 0:
            continue

        n_select = int(math.ceil(bucket_idx.size * (top_percent / 100.0)))
        if n_select <= 0:
            continue

        bucket_deriv = dy_dt[bucket_idx]
        top_local_order = np.argsort(bucket_deriv)[-n_select:]
        selected_idx = bucket_idx[top_local_order]
        irregular[selected_idx] = True

    return irregular


def main():
    print_bucket_info(BIS_BUCKET_SIZE)

    case_ids = discover_case_ids(DATA_DIR)
    total_cases = len(case_ids)

    if total_cases == 0:
        print("No matching parquet files found.")
        return

    print(f"Found {total_cases} case(s) in '{DATA_DIR}'.")
    print()

    for i, cid in enumerate(case_ids, start=1):
        print(f"[{i}/{total_cases}] Processing case {cid}...")

        path = os.path.join(DATA_DIR, f"{cid}{SUFFIX}")
        df = pd.read_parquet(path)

        t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
        emg = pd.to_numeric(df[EMG_COL], errors="coerce").to_numpy(dtype=float)
        bis = pd.to_numeric(df[BIS_COL], errors="coerce").to_numpy(dtype=float)

        dy_dt = np.gradient(emg, t)
        bis_bucket = assign_bis_buckets(bis, BIS_BUCKET_SIZE)

        labels = np.full(len(df), LABEL_REGULAR, dtype=object)

        # Invalid means ONLY BIS == 0 or NaN
        is_invalid = np.isnan(bis) | np.isclose(bis, 0.0, atol=1e-9)
        labels[is_invalid] = LABEL_INVALID

        # Irregular is selected only from non-invalid rows
        is_irregular = select_top_positive_derivative_per_bucket(
            dy_dt=dy_dt,
            bis_bucket=bis_bucket,
            is_invalid=is_invalid,
            top_percent=TOP_PERCENT_PER_BUCKET
        )

        # Extend only irregular labels, never invalid
        is_irregular = extend_mask_forward(t, is_irregular, EXTEND_SEC)
        is_irregular &= ~is_invalid
        labels[is_irregular] = LABEL_IRREGULAR

        # Fill only short regular gaps between irregular regions
        labels = relabel_short_regular_runs(labels, MIN_REGULAR_RUN_SEC)

        df[DERIV_COL] = dy_dt
        df[LABEL_COL] = labels
        df.to_parquet(path, index=False)

        n_invalid = int(np.sum(labels == LABEL_INVALID))
        n_irregular = int(np.sum(labels == LABEL_IRREGULAR))
        n_regular = int(np.sum(labels == LABEL_REGULAR))

        print(
            f"[{i}/{total_cases}] Done case {cid} | "
            f"invalid={n_invalid}, irregular={n_irregular}, regular={n_regular}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
