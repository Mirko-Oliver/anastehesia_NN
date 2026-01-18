"""
Adds two new columns to every {caseid}_rawdata.parquet in DATA_DIR:
  - emg_derivative: d(BIS/EMG)/dt 
  - label:
        * 'Invalid'   if BIS/BIS is NaN or 0
        * 'irregular' if |emg_derivative| is in the top DERIV_Q quantile for that case
        * 'regular'   otherwise

- labels are extended by EXTENDED_SEC after a detected spike
- small gaps (MIN_REGULAR_RUN_SEC in regular are filled as irregular
- Quantile is computed per case on |emg_derivative|.
- Derivative uses numpy.gradient with the Time column.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"

TIME_COL = "Time"
EMG_COL = "BIS/EMG"
BIS_COL = "BIS/BIS"

DERIV_COL = "emg_derivative"
LABEL_COL = "label"

DERIV_Q = 0.95  # top-quantile threshold (per case)

LABEL_INVALID = "invalid"
LABEL_IRREGULAR = "irregular"
LABEL_REGULAR = "regular"

EXTEND_SEC = 30	# How many Samples after an irregular/invalid flags should also be labeled invalid/irregular
MIN_REGULAR_RUN_SEC = 30	#Shortest Allowed Regular Run


def discover_case_ids(data_dir=DATA_DIR, suffix=SUFFIX):
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(data_dir)
        if f.endswith(suffix)
    )


def per_case_abs_deriv_quantile(dy_dt: np.ndarray, q: float) -> float:
    """
    Compute quantile of |dy/dt| ignoring NaNs. Returns NaN if no finite values.
    """
    v = np.abs(dy_dt)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.quantile(v, q))

def extend_mask_forward(t, mask, seconds= EXTEND_SEC):
    """
    For every True at index i, set subsequent indices True while t[j] <= t[i] + seconds.
    Assumes t is sorted ascending (as per your wide-format contract).
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
        t_end = t[i] + seconds
        j = i + 1
        while j < n and t[j] <= t_end:
            out[j] = True
            j += 1

    return out

def relabel_short_regular_runs(labels, min_len = MIN_REGULAR_RUN_SEC):
    """
    Any contiguous run of 'regular' shorter than min_len samples is relabeled to 'irregular'.
    
    """
    labels = labels.copy()
    is_reg = labels == LABEL_REGULAR
    idx = np.flatnonzero(is_reg)
    if idx.size == 0:
        return labels

    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)

    for seg in segments:
        if len(seg) < min_len:
            labels[seg] = LABEL_IRREGULAR

    return labels


def main():
	case_ids = discover_case_ids(DATA_DIR)
	n = len(case_ids)
	updated = 0

	for cid in case_ids:
		path = os.path.join(DATA_DIR, f"{cid}{SUFFIX}")
		df = pd.read_parquet(path)

		# Parse columns
		t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
		emg = pd.to_numeric(df[EMG_COL], errors="coerce").to_numpy(dtype=float)
		bis = pd.to_numeric(df[BIS_COL], errors="coerce").to_numpy(dtype=float)

		# Derivative
		dy_dt = np.gradient(emg, t)

		# Threshold on absolute derivative (per case)
		thr = per_case_abs_deriv_quantile(dy_dt, DERIV_Q)

		# Labels
		labels = np.full(len(df), LABEL_REGULAR, dtype=object)

		is_invalid = np.isnan(bis) | np.isclose(bis, 0.0, atol=1e-9)	#Mask of Rows that have BIS == 0 or NAN
		is_invalid = extend_mask_forward(t, is_invalid)	#Label the next 30 seconds afte an invalid as invalid
		labels[is_invalid] = LABEL_INVALID
		
		is_irregular = (~is_invalid) & np.isfinite(dy_dt) & (np.abs(dy_dt) >= thr)	#Label Rows irregular with irregular EMG
		is_irregular = extend_mask_forward(t, is_irregular)	# Extend irregular Lebels 30 seconds
		labels[is_irregular] = LABEL_IRREGULAR		
		
		labels = relabel_short_regular_runs(labels)
		# Write columns back
		df[DERIV_COL] = dy_dt
		df[LABEL_COL] = labels


		df.to_parquet(path, index=False)


		thr_str = f"{thr:.6g}" if np.isfinite(thr) else "NaN"
		n_invalid = int(np.sum(labels == LABEL_INVALID))
		n_irreg = int(np.sum(labels == LABEL_IRREGULAR))
		n_reg = int(np.sum(labels == LABEL_REGULAR))
		updated += 1
		print(f"case {cid}:  updated | thr(q{DERIV_Q:.2f})={thr_str} | {updated}/{n}")
        

	print("\nDone.")


if __name__ == "__main__":
    main()
