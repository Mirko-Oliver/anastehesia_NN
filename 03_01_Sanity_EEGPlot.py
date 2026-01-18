"""
Sanity check script:

- Pick a Case (Or Random)
- Loads relative bandpowers (EEG1/EEG2/EEGavg), BIS, and EMG 
- plots them
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("data")


CASE_ID = 4662  # Set to None for rng

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
EEG_PREFIXES = ["EEG1", "EEG2", "EEGavg"]
BIS_COL = "BIS/BIS"
EMG_COL = "BIS/EMG"

def collect_case_ids(data_dir=DATA_DIR):
    """
    Return a sorted list of case IDs from existing *_rawdata.parquet files.
    """
    case_ids = sorted(
        int(fname.split("_")[0])
        for fname in os.listdir(data_dir)
        if fname.endswith("_rawdata.parquet")
    )
    return case_ids


def load_case_parquet(case_id, data_dir=DATA_DIR):
    """
    Load the parquet file for a given case ID.
    """
    path = data_dir / f"{case_id}_rawdata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def extract_bandpowers_bis_emg(df):
	"""
	Extract the nexessary columns from the casefile
	"""
	# Find the time axis
	t = pd.to_numeric(df["Time"], errors="coerce")
	out = df.copy()
	out.index = t.astype(float).to_numpy()
	out.index.name = "Time"  
	
	#Generatle list of the EEG Bands Column Names
	band_cols = [f"{p}_{b}_rel" for p in EEG_PREFIXES for b in BANDS]
	#Filter list for Columns that exist (eg. EEG2 doesn't exist)
	existing_band_cols = [c for c in band_cols if c in out.columns]
	
	#Extract EEG Columns
	band_wide = out[existing_band_cols].copy() if existing_band_cols else pd.DataFrame(index=out.index)
	band_wide = band_wide.reindex(columns=[c for c in band_cols if c in band_wide.columns])
	
	#Find BIS and EMG  
	bis = pd.to_numeric(out[BIS_COL], errors="coerce")
	bis.name = "BIS"  
	emg = pd.to_numeric(out[EMG_COL], errors="coerce")
	emg.name = "EMG"

	return band_wide, bis, emg  
	
	
# Plotting

def _add_emg_axis(ax, offset=1.10):
    """
    Create a third y-axis for EMG, sharing the same x-axis with 'ax'
    """
    ax_emg = ax.twinx()
    ax_emg.spines["right"].set_position(("axes", offset))
    ax_emg.spines["right"].set_visible(True)
    return ax_emg


def plot_all_channels(case_id, band_wide, bis, emg):
    """
      - Stacked area plot of the relative bands 
      - BIS on a first right y-axis with its own y-limits.
      - EMG on a second right y-axis with its own y-limits.
    """
    prefixes = EEG_PREFIXES
    n_rows = len(prefixes)
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(12, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(f"Case {case_id} - Stacked relative EEG bandpowers with BIS & EMG")

    for ax, prefix in zip(axes, prefixes):
        t_band = band_wide.index.to_numpy()
        y_stacks = []
        labels = []
        for band in BANDS:
            col = f"{prefix}_{band}_rel"
            if col not in band_wide.columns:
                y = np.zeros_like(t_band, dtype=float)
            else:
                y = band_wide[col].fillna(0.0).to_numpy()
            y_stacks.append(y)
            labels.append(band)

        # LEFT axis: stacked relative bandpowers
        ax.stackplot(t_band, *y_stacks, labels=labels, baseline="zero")
        ax.set_ylabel(f"{prefix}\nrel. power")
        ax.set_ylim(0.0, 1.05)       
        ax.grid(True, alpha=0.3)

        # FIRST RIGHT axis: BIS (independent y-limits)
        ax_bis = ax.twinx()
        if not bis.empty:
            t_bis = bis.index.to_numpy()
            y_bis = bis.to_numpy()
            ax_bis.plot(
                t_bis, y_bis,
                color="black",
                linewidth=1.0,
                alpha=0.7,
                label="BIS",
            )

            finite_bis = y_bis[np.isfinite(y_bis)]
            if finite_bis.size > 0:
                ymin = finite_bis.min()
                ymax = finite_bis.max()
                span = ymax - ymin
                if span <= 0:
                    span = max(abs(ymax), 1.0)
                margin = 0.05 * span
                ax_bis.set_ylim(ymin - margin, ymax + margin)

            ax_bis.set_ylabel("BIS")
        else:
            ax_bis.set_ylabel("BIS (none)")

        # SECOND RIGHT axis: EMG (offset, independent y-limits)
        ax_emg = _add_emg_axis(ax, offset=1.10)
        if not emg.empty:
            t_emg = emg.index.to_numpy()
            y_emg = emg.to_numpy()
            ax_emg.plot(
                t_emg, y_emg,
                color="magenta",
                linewidth=1.0,
                alpha=0.7,
                label="EMG",
            )

            finite_emg = y_emg[np.isfinite(y_emg)]
            if finite_emg.size > 0:
                ymin = finite_emg.min()
                ymax = finite_emg.max()
                span = ymax - ymin
                if span <= 0:
                    span = max(abs(ymax), 1.0)
                margin = 0.05 * span
                ax_emg.set_ylim(ymin - margin, ymax + margin)

            ax_emg.set_ylabel("EMG")
        else:
            ax_emg.set_ylabel("EMG (none)")

        # legend
        band_handles, band_labels = ax.get_legend_handles_labels()
        bis_handles, bis_labels = ax_bis.get_legend_handles_labels()
        emg_handles, emg_labels = ax_emg.get_legend_handles_labels()
        ax.legend(
            band_handles + bis_handles + emg_handles,
            band_labels + bis_labels + emg_labels,
            loc="upper right",
            fontsize=8,
        )

        ax.set_title(f"{prefix} stacked relative bandpowers + BIS + EMG")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    case_ids = collect_case_ids(DATA_DIR)


    if CASE_ID is not None:
        case_id = CASE_ID
        print(f"Using configured case: {case_id}")
    else:
        case_id = random.choice(case_ids)
        print(f"Selected random case: {case_id}")

    df = load_case_parquet(case_id, DATA_DIR)
    band_wide, bis, emg = extract_bandpowers_bis_emg(df)

    plot_all_channels(case_id, band_wide, bis, emg)


