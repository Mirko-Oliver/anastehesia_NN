"""
Plot BIS/EMG for 5 cases:
- emg_derivative 
- label (Invalid / irregular / regular)

Shading:
- Shade regions where label == 'irregular' 
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
SUFFIX = "_rawdata.parquet"

TIME_COL = "Time"
EMG_COL = "BIS/EMG"
DERIV_COL = "emg_derivative"
LABEL_COL = "label"

N_CASES = 5
RANDOM_SEED = None

LABEL_IRREGULAR = "irregular"
LABEL_INVALID = "Invalid"


SHADE_INVALID = True


def discover_case_ids(data_dir=DATA_DIR, suffix=SUFFIX):
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(data_dir)
        if f.endswith(suffix)
    )


def load_case_for_plot(case_id, data_dir=DATA_DIR):
    path = data_dir / f"{case_id}{SUFFIX}"
    df = pd.read_parquet(path)

    missing = [c for c in (TIME_COL, EMG_COL, DERIV_COL, LABEL_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"Case {case_id}: missing columns {missing} in {path.name}")

    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    emg = pd.to_numeric(df[EMG_COL], errors="coerce").to_numpy(dtype=float)
    deriv = pd.to_numeric(df[DERIV_COL], errors="coerce").to_numpy(dtype=float)
    labels = df[LABEL_COL].astype(str).to_numpy()

    m = np.isfinite(t)
    return t[m], emg[m], deriv[m], labels[m], path


def contiguous_true_segments(mask: np.ndarray):
    """
    Return list of (start_idx, end_idx) inclusive segments where mask is True.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)
    return [(int(seg[0]), int(seg[-1])) for seg in segments]


def shade_segments(ax, t, segments, alpha=0.20):
    for i0, i1 in segments:
        ax.axvspan(t[i0], t[i1], alpha=alpha)


def main():
    case_ids = discover_case_ids(DATA_DIR)
    if len(case_ids) < N_CASES:
        raise RuntimeError(f"Found only {len(case_ids)} cases in {DATA_DIR}, need {N_CASES}.")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    selected = random.sample(case_ids, N_CASES)
    print("Selected case IDs:", selected)

    fig, axes = plt.subplots(N_CASES, 1, sharex=False, figsize=(14, 3.2 * N_CASES))
    if N_CASES == 1:
        axes = [axes]

    for ax, cid in zip(axes, selected):
        t, emg, deriv, labels, path = load_case_for_plot(cid, DATA_DIR)

        # Shade irregular regions based on labels
        irregular_mask = labels == LABEL_IRREGULAR
        irr_segments = contiguous_true_segments(irregular_mask)
        shade_segments(ax, t, irr_segments, alpha=0.20)

        # Optional: shade invalid regions too 
        if SHADE_INVALID:
            invalid_mask = labels == LABEL_INVALID
            inv_segments = contiguous_true_segments(invalid_mask)
            shade_segments(ax, t, inv_segments, alpha=0.10)

        # Plot EMG and derivative 
        ax.plot(t, emg, linewidth=1.0, label=EMG_COL)
        ax.plot(t, deriv, linewidth=1.0, label=DERIV_COL)

        n_irr = int(np.sum(irregular_mask))
        n_inv = int(np.sum(labels == LABEL_INVALID))
        ax.set_title(f"Case {cid} ({path.name}) | irregular={n_irr} invalid={n_inv}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
