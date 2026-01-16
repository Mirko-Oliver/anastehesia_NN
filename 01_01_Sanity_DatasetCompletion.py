"""
This Script is a Sanity Check, to make sure everything is in order.
Specifically it loads the created parquet files from the previous step,
calculates data completion for each track of each case and plots them.
It also plots the length of each case in minutes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_folder = "data"  

# get case_ids from filenames like {caseid}_rawdata.parquet
case_ids = sorted(
    int(fname.split("_")[0])
    for fname in os.listdir(data_folder)
    if fname.endswith("_rawdata.parquet")
)


# 1) Collect union of all track names across all parquet files

def collect_all_tracks(case_ids, folder= data_folder):
    all_tracks = set()
    for caseid in case_ids:
        parquet_path = os.path.join(folder, f"{caseid}_rawdata.parquet")
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        all_tracks.update(df.columns)
    tracks_sorted = sorted(all_tracks)
    print(f"Found {len(tracks_sorted)} unique tracks across {len(case_ids)} cases.")
    return tracks_sorted



# 2) Compute completion matrix [tracks, cases] and case durations

def compute_completion(case_ids, tracks, folder= data_folder, ignore_sec= 20 * 60):
	"""
    Computes:
      - completion[track, case] = % data completeness 
      - case_durations[case]    = case length in seconds 
    """
	n_tracks = len(tracks)
	n_cases = len(case_ids)
	completion = np.zeros((n_tracks, n_cases), dtype=float)
	case_durations = np.zeros(n_cases, dtype=float)

	for j, caseid in enumerate(case_ids):
		parquet_path = os.path.join(folder, f"{caseid}_rawdata.parquet")
		print(f"Loading case {caseid} ({j+1}/{n_cases})...")
		
		# Load Case File
		df = pd.read_parquet(parquet_path)
		n_rows = len(df)
		case_durations[j] = float(n_rows - 1)

		# Exclude first & last ignore_sec seconds
		valid_start = ignore_sec
		valid_end   = n_rows - ignore_sec

		if valid_end <= valid_start:
			print("Case too short for valid window. Marking 0%.")
			completion[:, j] = 0.0
			continue

		expected_samples = (valid_end - valid_start) + 1
		df_valid = df.loc[valid_start:valid_end]

		if df_valid.empty:
			print(" No rows in valid region, marking all 0%.")
			completion[:, j] = 0.0
			continue

		# Compute completeness per track column
		for i, trk in enumerate(tracks):
			if trk not in df_valid.columns:
				completion[i, j] = 0.0
				continue
			n_present = int(df_valid[trk].notna().sum())
			completion[i, j] = (n_present / expected_samples) * 100.0

	return completion, case_durations



# 3) Plot Completion heatmap 

def plot_completion_heatmap(completion, case_ids, tracks):
	""" Plots the Completion Matrix as a Heatmap.
	Tracks on the Y Axis, Cases on the X Axis"""
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        completion,
        aspect="auto",
        origin="upper",
        vmin=0,
        vmax=100,
        cmap="viridis",
    )

    ax.set_xlabel("Case ID")
    ax.set_ylabel("Track")

    ax.set_xticks(np.arange(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=90, fontsize=5)

    ax.set_yticks(np.arange(len(tracks)))
    ax.set_yticklabels(tracks, fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Data completeness (%)")

    plt.tight_layout()
    plt.show()


# 4) Plot case lengths

def plot_case_durations(case_ids, case_durations_sec):
    """
    Plots a bar chart of case length in minutes for each case.
    """
    durations_min = case_durations_sec / 60.0

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.bar(np.arange(len(case_ids)), durations_min)

    ax.set_xlabel("Case ID")
    ax.set_ylabel("Case length (minutes)")

    ax.set_xticks(np.arange(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=90, fontsize=5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) find all tracks present
    all_tracks = collect_all_tracks(case_ids)

    # 2) compute completion matrix and case duations
    completion, case_durations = compute_completion(case_ids, all_tracks)

    # 3) plot completion heatmap
    plot_completion_heatmap(completion, case_ids, all_tracks)

    # 4) plot case length per case
    plot_case_durations(case_ids, case_durations)

