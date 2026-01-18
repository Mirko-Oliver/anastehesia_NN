"""
This Script filters the Dataset to remove cases that are too short, or missing critical data
It also removes Tracks that are only present in some cases. 
"""

import os
import shutil
import numpy as np
import pandas as pd

DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "Archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

min_case_length = 90 * 60	#Minimum Case Length in Sec 
ignore_sec = 30 * 60		#ignore first and last minutes in data completion calculation 

#Minimum Track Completion [%] in Minimum of Cases to keep the Track in the Dataset 
min_track_completion = 50.0
min_cases = 250

#Tracks That are required in a case 
required_tracks= [
	"BIS/BIS",
	"Solar8000/ART_MBP",
]
# The min completion of required Tracks
req_min_completion = 50.0


# infer case_ids from filenames like {caseid}_rawdata.parquet
def get_case_ids():
	return sorted(
		int(fname.split("_")[0])
		for fname in os.listdir("data")
		if fname.endswith("_rawdata.parquet")
	)


def archive_short_files(case_ids, data_dir=DATA_DIR, archive_dir=ARCHIVE_DIR, min_seconds= min_case_length): 
	""" Move any parquet file shother than the time threshold"""
	for caseid in case_ids:
		fpath = os.path.join(data_dir, f"{caseid}_rawdata.parquet")
		df = pd.read_parquet(fpath)
		n_rows = len(df)
		if n_rows < min_seconds:
			shutil.move(fpath, os.path.join(archive_dir, f"{caseid}_rawdata.parquet"))
			print(f"{caseid}-> moved to Archive, too short")
	print("\n moved all short Files.")

def collect_all_tracks(case_ids, folder= DATA_DIR):
    all_tracks = set()
    for caseid in case_ids:
        parquet_path = os.path.join(folder, f"{caseid}_rawdata.parquet")
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        all_tracks.update(df.columns)
    tracks_sorted = sorted(all_tracks)
    print(f"Found {len(tracks_sorted)} unique tracks across {len(case_ids)} cases.")
    return tracks_sorted


def compute_completion(case_ids, tracks, folder= DATA_DIR, ignore_sec=ignore_sec):
	"""
    Computes:
      - completion[track, case] = % data completeness 
      - case_durations[case]    = case length in seconds 
    """
	n_tracks = len(tracks)
	n_cases = len(case_ids)
	completion = np.zeros((n_tracks, n_cases), dtype=float)

	for j, caseid in enumerate(case_ids):
		parquet_path = os.path.join(folder, f"{caseid}_rawdata.parquet")
		print(f"Loading parquet for case {caseid} ({j+1}/{n_cases})...")
		

		df = pd.read_parquet(parquet_path)
		n_rows = len(df)

		# Exclude first & last ignore_sec seconds
		valid_start = ignore_sec
		valid_end   = n_rows - ignore_sec

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

	return completion


def find_tracks_to_remove(tracks,
                          completion,
                          min_cases=min_cases,
                          min_completion=min_track_completion):
    """
	returns tracks that have >min_completion in >min_cases
    """
    n_cases_per_track = np.sum(completion >= min_completion, axis=1)

    tracks_to_remove = [
        trk for trk, n_ok in zip(tracks, n_cases_per_track)
        if n_ok < min_cases
    ]

    print("\nTracks with low coverage (candidates for removal):")
    for trk, n_ok in zip(tracks, n_cases_per_track):
        if n_ok < min_cases:
            print(f"  {trk}: {n_ok} cases with ≥ {min_completion:.1f}%")

    return tracks_to_remove




def archive_cases_by_track_completion(case_ids,
								   tracks,
								   completion,
								   required_tracks=required_tracks,
								   min_completion=req_min_completion,
								   data_dir=DATA_DIR,
								   archive_dir=ARCHIVE_DIR):
	"""
	Move cases into Archive if certain tracks in that case do NOT reach
	a completion threshold.
	"""
	# map track name -> index in completion matrix
	track_to_idx = {t: i for i, t in enumerate(tracks)}
	print("\nChecking cases for required-track completion")

	for j, caseid in enumerate(case_ids):
		case_ok = True
		missing_info = []

		for trk in required_tracks:
			i = track_to_idx[trk]
			pct = completion[i, j]

			if pct < min_completion:
				case_ok = False
				missing_info.append((trk, pct))

		if not case_ok:
			src = os.path.join(data_dir, f"{caseid}_rawdata.parquet")
			if os.path.exists(src):		#If File still at source (not moved)
				dst = os.path.join(archive_dir, f"{caseid}_rawdata.parquet")
				shutil.move(src, dst)
				print(f"Case {caseid} -> moved to Archive (below threshold for tracks: "
					  f"{', '.join(f'{t} ({p:.1f}%)' for t, p in missing_info)})")


def remove_tracks_from_all_cases(tracks_to_remove, data_dir=DATA_DIR):
	""" This function removes a list of tracks from all Cases"""
	print("\nRemoving tracks from all cases")

	files = [
		f for f in os.listdir(data_dir)
		if f.endswith("_rawdata.parquet")
	]

	for fname in files:
		fpath = os.path.join(data_dir, fname)
		df = pd.read_parquet(fpath)
		
		cols_before = set(df.columns)
		removable = cols_before.intersection(tracks_to_remove)
		df = df.drop(columns=list(removable))
		df.to_parquet(fpath)
		


if __name__ == "__main__":
	#1) get caseids
	case_ids = get_case_ids()
	#2) Archive cases that are too short
	archive_short_files(case_ids)
	# 3) Collect all tracks
	case_ids = get_case_ids()
	tracks = collect_all_tracks(case_ids)
	# 4) Compute completion matrix 
	completion = compute_completion(case_ids, tracks)
	# 5) Find tracks to remove )low coverage)
	tracks_to_remove = find_tracks_to_remove(tracks, completion)
	print("\nTracks recommended for removal:")
	for t in tracks_to_remove:
		print(" -", t)
	 #6) Ask if we should remove those tracks from all cases
	answer = input("Remove these tracks from ALL remaining cases? (yes/no): ").strip().lower()
	if answer == "yes":
		remove_tracks_from_all_cases(tracks_to_remove)
	else:
		print("No cahnges.")
	# 7) Move cases to Archive if required tracks are too incomplete
	case_ids = get_case_ids()
	archive_cases_by_track_completion(case_ids, tracks, completion)


