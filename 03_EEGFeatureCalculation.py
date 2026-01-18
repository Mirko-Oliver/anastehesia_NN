"""
This Script imports the EEG1 and EEG2 tracks from VitalDB and calculates the relative Bandpowers
by Track and a average of both Tracks
for any cases with existing Parquet Files in the Data Folder.

the environ config and main function allow for multiprocessing of cases which increases the scripts speed significantly
Operations:
1) Savitzky–Golay Detrending
2) Epoching, Split Signal into overlapping 2 second windows
3) FFT 
4) PSD
5) Relative PSD / Band
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import vitaldb

DATA_DIR = "data"

# Epoching: 2 s epochs, 75% overlap 
EPOCH_LEN_S = 2.0
EPOCH_HOP_S = 0.5

# Smoothing window for PSD-based features ,trailing 30 s
PSD_SMOOTH_S = 30.0

# EEG Sampling frequency
fs = 128

# EEG bands for relative band powers
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Total band 
TOTAL_BAND = (0.5, 47.0)


def collect_case_ids(data_dir=DATA_DIR):
	return sorted(
		int(fname.split("_")[0])
		for fname in os.listdir("data")
		if fname.endswith("_rawdata.parquet")
	)


def load_eeg_channels(case_id, fs=fs):
    """
    Load raw EEG1 and EEG2 waveforms for a given case from VitalDB.
    """
    tracks = ["BIS/EEG1_WAV", "BIS/EEG2_WAV"]
    vf = vitaldb.VitalFile(case_id, tracks)
    df = vf.to_pandas(tracks, 1.0 / fs)
    out = {}
    for track_name, key in zip(tracks, ["EEG1", "EEG2"]):
        if track_name in df.columns:
            x = pd.to_numeric(df[track_name], errors="coerce").to_numpy()
            out[key] = x[np.isfinite(x)]
        else:  # protect against missing EEG data
            out[key] = np.array([], dtype=float)
    return out


def detrend_savgol(x, win_s=0.5, poly=3):
    """
    Remove slow trend from EEG using a Savitzky–Golay filter.
    """
    win_len = int(round(win_s * fs))
    win_len = max(5, win_len | 1)  # ensure odd length, >= 5
    trend = savgol_filter(x, window_length=win_len, polyorder=poly, mode="interp")
    return x - trend


def make_epochs(x, epoch_s=EPOCH_LEN_S, hop_s=EPOCH_HOP_S):
    """
    Cut the continuous signal into overlapping epochs.

    - epoch_s: epoch length in seconds
    - hop_s: step in seconds between consecutive epoch starts
    - Returns array (n_epochs, epoch_len_samples).
    """
    L = int(round(epoch_s * fs))   # samples per epoch
    H = int(round(hop_s * fs))     # hop in samples
    if L <= 0 or H <= 0 or len(x) < L:
        return np.empty((0, L), dtype=float)
    starts = np.arange(0, len(x) - L + 1, H, dtype=int)
    return np.stack([x[s:s + L] for s in starts], axis=0)


def epoch_centers_seconds(n_epochs, epoch_s=EPOCH_LEN_S, hop_s=EPOCH_HOP_S):
    """
    Compute center time of each epoch
    """
    return (np.arange(n_epochs) * hop_s) + (epoch_s / 2.0)


def compute_epoch_spectra(epochs):
    """
    Compute per-epoch FFT and power spectral density proxy.

    - Applies a Blackman window per epoch.
    - Returns (freqs, X, P) where:
        freqs: frequency array for rFFT
        X: complex FFT coefficients (epochs x freqs)
        P: power spectrum estimates (epochs x freqs)
    """
    if epochs.size == 0:
        return None, None, None
    L = epochs.shape[1]
    w = np.blackman(L)[None, :]
    X = np.fft.rfft(epochs * w, n=L, axis=1)               # (E, Fbins)
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    P = (np.abs(X) ** 2) / (np.sum(np.abs(w) ** 2))        # per-epoch PSD proxy
    return freqs, X, P


def band_power(freqs, Pavg, band):
    """
    Integrate spectral power.
    """
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return np.nan
    return np.trapz(Pavg[m], freqs[m])


def rolling_psd_features_trailing(
    x,
    prefix,
    epoch_s=EPOCH_LEN_S,
    hop_s=EPOCH_HOP_S,
    smooth_s=PSD_SMOOTH_S,
):
    """
    Compute trailing-30s EEG spectral features

    - relative bandpowers for delta/theta/alpha/beta/gamma.
    - `prefix` controls the column names, e.g. 'EEG1' -> EEG1_delta_rel

    Returns a DataFrame indexed by time_sec with columns:
        [f'{prefix}_delta_rel', f'{prefix}_theta_rel', ...]
    """
    epochs = make_epochs(x, epoch_s=epoch_s, hop_s=hop_s)
    if epochs.size == 0:
        return pd.DataFrame()

    freqs, X, P = compute_epoch_spectra(epochs)
    centers = epoch_centers_seconds(len(epochs), epoch_s=epoch_s, hop_s=hop_s)

    # Integer time grid 
    t_grid = np.arange(int(np.floor(centers[-1])) + 1, dtype=int)

    # Initialize empty output arrays 
    data = {
        f"{prefix}_delta_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_theta_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_alpha_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_beta_rel":  np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_gamma_rel": np.full_like(t_grid, np.nan, dtype=float),
    }

    for k, t in enumerate(t_grid):
        # Trailing window [t - smooth_s, t]
        t_end = float(t)
        t_start = t_end - smooth_s
        use = (centers > t_start) & (centers <= t_end)
        if not np.any(use):
            continue

        # Average PSD across epochs in the trailing window
        Pavg = np.nanmean(P[use, :], axis=0)

        # Total power 
        total = band_power(freqs, Pavg, TOTAL_BAND)
        if not (np.isfinite(total) and total > 0):
            continue

        # Band powers
        delta = band_power(freqs, Pavg, BANDS["delta"])
        theta = band_power(freqs, Pavg, BANDS["theta"])
        alpha = band_power(freqs, Pavg, BANDS["alpha"])
        beta  = band_power(freqs, Pavg, BANDS["beta"])
        gamma = band_power(freqs, Pavg, BANDS["gamma"])

        # Relative bandpowers 
        data[f"{prefix}_delta_rel"][k] = delta / total if np.isfinite(delta) else np.nan
        data[f"{prefix}_theta_rel"][k] = theta / total if np.isfinite(theta) else np.nan
        data[f"{prefix}_alpha_rel"][k] = alpha / total if np.isfinite(alpha) else np.nan
        data[f"{prefix}_beta_rel"][k]  = beta  / total if np.isfinite(beta)  else np.nan
        data[f"{prefix}_gamma_rel"][k] = gamma / total if np.isfinite(gamma) else np.nan

    out = pd.DataFrame(data, index=t_grid)
    out.index.name = "time_sec"
    return out


def process_case(case_id):
	"""
	Process a single case:

	1. Load EEG1 and EEG2 from VitalDB.
	2. Detrend both channels 
	3. Compute relative band powers for EEG1 and EEG2 separately
	4. Compute average relative bandpowers 
	"""
	t0 = time.perf_counter()
	try:
		# Load EEG1 & EEG2 waveforms
		eeg_dict = load_eeg_channels(case_id)
		x1_raw = eeg_dict["EEG1"]
		x2_raw = eeg_dict["EEG2"]

		if x1_raw.size == 0 and x2_raw.size == 0:
			return case_id, "No EEG1 or EEG2 samples found"

		psd_feats_list = []

		# EEG1 features
		if x1_raw.size > 0:
			x1 = detrend_savgol(x1_raw, win_s=0.5, poly=3)
			feats1 = rolling_psd_features_trailing(x1, prefix="EEG1")
			if not feats1.empty:
				psd_feats_list.append(feats1)
		else:
			feats1 = pd.DataFrame()

		# EEG2 features
		if x2_raw.size > 0:
			x2 = detrend_savgol(x2_raw, win_s=0.5, poly=3)
			feats2 = rolling_psd_features_trailing(x2, prefix="EEG2")
			if not feats2.empty:
				psd_feats_list.append(feats2)
		else:
			feats2 = pd.DataFrame()

		# Combined average features (EEGavg_*) 
		if not feats1.empty and not feats2.empty:
			# Align on common time index
			idx = feats1.index.intersection(feats2.index)
			if len(idx) > 0:
				avg_data = {}
				for band in BANDS.keys():
					c1 = f"EEG1_{band}_rel"
					c2 = f"EEG2_{band}_rel"
					cavg = f"EEGavg_{band}_rel"
					v1 = feats1.loc[idx, c1]
					v2 = feats2.loc[idx, c2]
					avg_data[cavg] = (v1 + v2) / 2.0
				feats_avg = pd.DataFrame(avg_data, index=idx)
				psd_feats_list.append(feats_avg)

		if not psd_feats_list:
			return case_id, "No features computed (all psd_feats empty)"

		# Combine all feature tables along columns, align by time
		psd_feats = pd.concat(psd_feats_list, axis=1).sort_index()
		if psd_feats.empty:
			return case_id, "No features after concatenation"
			
		parquet_path = Path(DATA_DIR) / f"{case_id}_rawdata.parquet"
		if not parquet_path.exists():
			return case_id, f"Missing parquet file: {parquet_path.name}"

		df_existing = pd.read_parquet(parquet_path)
		time_existing = pd.to_numeric(df_existing["Time"], errors="coerce")
		if time_existing.isna().all():
			return case_id, "Existing parquet Time column could not be parsed"

		# Use seconds as join key
		df_existing = df_existing.copy()
		df_existing["time_sec"] = np.round(time_existing).astype("int64")
		df_existing = df_existing.set_index("time_sec", drop=True)

		# Prepare features for join
		feats_wide = psd_feats.copy()
		feats_wide.index = feats_wide.index.astype("int64")
		feats_wide.index.name = "time_sec"

		# Left-join
		df_out = df_existing.join(feats_wide, how="left")

		# Restore original row order and Time column position (Time must remain first column)
		df_out = df_out.sort_index()

		# Drop helper index column? keep index as time_sec only temporarily
		df_out = df_out.reset_index(drop=True)

		# Ensure Time is first column (preserve original Time values)
		# Because reset_index(drop=True) kept columns, Time is still present;
		# we reorder explicitly to guarantee first position.
		cols = list(df_out.columns)
		if "Time" in cols:
			cols.remove("Time")
			df_out = df_out[["Time"] + cols]
		else:
			return case_id, "Time column disappeared unexpectedly after join"


		# Write back to parquet
		df_out.to_parquet(parquet_path)

		dt = time.perf_counter() - t0
		return case_id, f"OK ({dt:.1f}s → bandpowers (EEG1, EEG2, EEGavg) appended)"

	except Exception as e:
		return case_id, f"ERROR: {e}"


def main():
    """
    Main entry point:

    - Discovers case IDs from '{caseid}_rawdata.parquet'.
    - Spawns a process pool and runs process_case(case_id) in parallel.
    """
    case_ids = collect_case_ids()

    # Re-assert BLAS thread limits inside main
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if not case_ids:
        print("[WARN] No '*_rawdata.parquet' files found. Nothing to do.")
        return

    n_procs = min(os.cpu_count() or 1, len(case_ids))
    print(f"[INFO] Starting pool with {n_procs} processes")

    t0 = time.perf_counter()
    with get_context("spawn").Pool(processes=n_procs) as pool:
        for case_id, msg in pool.imap_unordered(process_case, case_ids, chunksize=1):
            print(f"[CASE {case_id}] {msg}")
    dt = time.perf_counter() - t0
    print(f"[INFO] Finished all cases in {dt:.1f}s")


if __name__ == "__main__":
    main()

