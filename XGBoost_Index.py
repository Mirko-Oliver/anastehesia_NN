""" This Script creates a Baseline Index with XGBoost, it does not have Causal restrictions, and does not use time dependendcies
1) Collect Case_IDS in dataset 
2) Import Static Features
3) Create Trainingweights

"""
import os
import csv
import sys
import json
import joblib
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

import xgboost as xgb

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

Static_Features = ["age", "sex"] #MISSING LBM
Dynamic_Features = [
#'Solar8000/ART_DBP', 
'Solar8000/ART_MBP', 
#'Solar8000/ART_SBP', 
'Solar8000/BT', 
'Solar8000/ETCO2', 
#'Solar8000/FEO2', 
#'Solar8000/FIO2', 
'Solar8000/HR', 
#'Solar8000/INCO2', 
#'Solar8000/PLETH_HR', 
#'Solar8000/PLETH_SPO2', 
#'Solar8000/RR_CO2', 
#'Solar8000/ST_II', 
#'Solar8000/VENT_INSP_TM', 
#'Solar8000/VENT_MAWP', 
#'Solar8000/VENT_MV', 
#'Solar8000/VENT_PIP', 
#'Solar8000/VENT_PPLAT', 
#'Solar8000/VENT_RR', 
#'Solar8000/VENT_TV', 
#'Orchestra/PPF20_CE', 
#'Orchestra/PPF20_CP', 
#'Orchestra/PPF20_CT', 
#'Orchestra/PPF20_RATE', 
#'Orchestra/PPF20_VOL', 
#'Orchestra/RFTN20_CE', 
#'Orchestra/RFTN20_CP', 
#'Orchestra/RFTN20_CT', 
#'Orchestra/RFTN20_RATE', 
#'Orchestra/RFTN20_VOL', 
'BIS/EMG', 
'BIS/SEF', 
#'BIS/SQI', 
'BIS/SR', 
#'BIS/TOTPOW', 
'EEG1_delta_rel', 
'EEG1_theta_rel', 
'EEG1_alpha_rel', 
'EEG1_beta_rel', 
'EEG1_gamma_rel', 
#'EEG2_delta_rel', 
#'EEG2_theta_rel', 
#'EEG2_alpha_rel', 
#'EEG2_beta_rel', 
#'EEG2_gamma_rel', 
#'EEGavg_delta_rel', 
#'EEGavg_theta_rel', 
#'EEGavg_alpha_rel', 
#'EEGavg_beta_rel', 
#'EEGavg_gamma_rel', 
#'emg_derivative', 
]

TARGET_COL = 'BIS/BIS'

BIS_EDGES = np.array([16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101], dtype=float)
N_BINS = len(BIS_EDGES) - 1

def collect_case_ids(dataset):
	# This Function collects all Case_Ids within a dataset (Training/ Testing/ Validation)
	case_ids = []
	file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

	with open(file_path, newline="", encoding="utf-8") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row.get("Dataset") == dataset:
				case_ids.append(row.get("caseid"))

	return case_ids

def create_mask(df):
	#Filter for Regular Datapoints, and Remove Start of Case
	mask = (df["label"] == "regular") & (df["Time"] > 1500)
	return mask
	
def make_bis_bins(y):
    y_num = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)

    b = np.digitize(y_num, BIS_EDGES, right=False) - 1  
    b[np.isnan(y_num)] = -1
    b[(y_num < BIS_EDGES[0]) | (y_num >= BIS_EDGES[-1])] = -1
    return b.astype(int)
    
def scan_training_stats(case_ids):
	"""
	create dicct {caseid -> usable case length}
	create BIS BIN Size Array with each index being a Bin as defined in make_bis_bins()
	"""
	case_len_map = {}
	bin_counts = None

	for cid in case_ids:
		path = os.path.join(DATA_DIR, cid + SUFFIX)
		df = pd.read_parquet(path)

		mask = create_mask(df)
		df = df.loc[mask]

		n = len(df)
		if n == 0:
			continue

		case_len_map[cid] = n
		if bin_counts is None:
			bin_counts = np.zeros(N_BINS, dtype=int)
			
		bins = make_bis_bins(df[TARGET_COL].to_numpy())
		valid = bins >= 0
		counts = np.bincount(bins[valid], minlength=N_BINS)

		bin_counts += counts

	return case_len_map, bin_counts

def calculate_bin_weight(bin_counts, min_count=200, alpha=0.5):
	"""Takes the BIS Bin Counts and calculates the corresponding weights.
	Uses min_count so weight doesn't become too heavy for small bins,
	 and alpha to set aggressiveness in balancing"""
	freq = bin_counts.astype(float)
	freq[freq < min_count] = min_count
	return 1.0 / np.power(freq, alpha)
    
def load_training_arrays(case_ids, features, case_len_map, bin_w_lookup, clinical_map):
	X_list, y_list, w_list = [], [], []

	for cid in case_ids:
		if cid not in case_len_map:
			continue

		path = os.path.join(DATA_DIR, cid + SUFFIX)
		if not os.path.exists(path):
			continue

		df = pd.read_parquet(path)

		mask = create_mask(df)
		df = df.loc[mask].copy()
		if len(df) == 0:
			continue

		# add statics
		df = add_static_features(df, cid, clinical_map)
		df = coerce_static_types(df)

		# build arrays
		# if any feature missing, raise with a clear message
		missing = [c for c in features if c not in df.columns]
		if missing:
			raise KeyError(f"Case {cid} missing feature columns: {missing}")

		X = df[features].to_numpy(dtype=np.float32)
		y = df[TARGET_COL].to_numpy(dtype=np.float32)

		# case weight (based on masked length)
		w_case = np.full(len(df), 1.0 / max(case_len_map[cid], 1), dtype=np.float32)

		# bin weight
		bins = make_bis_bins(y)
		w_bin = np.ones(len(df), dtype=np.float32)
		valid = bins >= 0
		w_bin[valid] = bin_w_lookup[bins[valid]].astype(np.float32)
		w_bin[~valid] = 0.0

		w = w_case * w_bin

		X_list.append(X)
		y_list.append(y)
		w_list.append(w)

	if not X_list:
		raise RuntimeError("No training data loaded (X_list empty). Check mask, paths, and columns.")

	X_all = np.vstack(X_list)
	y_all = np.concatenate(y_list)
	w_all = np.concatenate(w_list)

	#stabilization: normalize weights to mean 1
	nz = w_all > 0
	if np.any(nz):
		w_all = w_all * (np.count_nonzero(nz) / w_all[nz].sum())

	return X_all, y_all, w_all

def load_split_arrays(case_ids, features, clinical_map):
	#basically load_training_arrays w/o weight
	X_list, y_list = [], []

	for cid in case_ids:
		path = os.path.join(DATA_DIR, cid + SUFFIX)
		if not os.path.exists(path):
			continue

		df = pd.read_parquet(path)

		mask = create_mask(df)
		df = df.loc[mask].copy()
		if len(df) == 0:
			continue

		# add statics
		df = add_static_features(df, cid, clinical_map)
		df = coerce_static_types(df)

		missing = [c for c in features if c not in df.columns]
		if missing:
			raise KeyError(f"Case {cid} missing feature columns: {missing}")

		X = df[features].to_numpy(dtype=np.float32)
		y = df[TARGET_COL].to_numpy(dtype=np.float32)

		X_list.append(X)
		y_list.append(y)

	if not X_list:
		raise RuntimeError("No data loaded (X_list empty). Check mask, paths, and columns.")

	return (
		np.vstack(X_list),
		np.concatenate(y_list),
	)
def load_clinical_info_map(static_cols = Static_Features):
    """
    Returns: Dicc mapping relevant static Features to case_ids
    """
    fpath = os.path.join(DATA_DIR, CLINICAL_INFO)
    info = {}
    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get('caseid')
            if cid is None:
                continue
            info[cid] = {c: row.get(c) for c in static_cols}
    return info

def add_static_features(df: pd.DataFrame, caseid: str, clinical_map: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """
    Adds static columns to every row of df for this caseid.
    """

    statics = clinical_map[caseid]
    for k, v in statics.items():
        df[k] = v
    return df
    
def coerce_static_types(df: pd.DataFrame) -> pd.DataFrame:
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({"m": 1, "f": 0, })
    return df
    
def mae(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.nanmean(np.abs(y_pred - y_true)))

def rmse(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))

def print_metrics(name, y_true, y_pred):
	print(f"[METRIC] {name} MAE:  {mae(y_true, y_pred):.4f}")
	print(f"[METRIC] {name} RMSE: {rmse(y_true, y_pred):.4f}")

def print_metrics_by_bin(name, y_true, y_pred):
	b = make_bis_bins(y_true)
	for bi in range(N_BINS):
		idx = (b == bi)
		n = int(np.sum(idx))
		if n == 0:
			continue
		print(f"[BIN] {name} bin={bi:02d} n={n:7d}  MAE={mae(y_true[idx], y_pred[idx]):.4f}  RMSE={rmse(y_true[idx], y_pred[idx]):.4f}")

def print_metrics_by_sex(name, X, y_true, y_pred, features):
	if "sex" not in features:
		print(f"[WARN] {name} cannot print by sex because 'sex' not in FEATURES")
		return
	sex_i = features.index("sex")
	sex = X[:, sex_i]

	# sex expected as 0/1 (float32). Ignore NaNs.
	mask_m = (sex == 1)
	mask_f = (sex == 0)

	nm = int(np.sum(mask_m))
	nf = int(np.sum(mask_f))

	if nm > 0:
		print(f"[SEX] {name} sex=1 (M) n={nm:7d}  MAE={mae(y_true[mask_m], y_pred[mask_m]):.4f}  RMSE={rmse(y_true[mask_m], y_pred[mask_m]):.4f}")
	if nf > 0:
		print(f"[SEX] {name} sex=0 (F) n={nf:7d}  MAE={mae(y_true[mask_f], y_pred[mask_f]):.4f}  RMSE={rmse(y_true[mask_f], y_pred[mask_f]):.4f}")
    
if __name__ == "__main__":
	# 1) Create Map of Static Features
	clinical_map = load_clinical_info_map(Static_Features)
	# 2) Collect training case ids
	train_case_ids = collect_case_ids("Training")

	# 3) find usable lengths and BIS bin counts
	case_len_map, bin_counts = scan_training_stats(train_case_ids)

	# 4) Build global bin weight lookup
	bin_w_lookup = calculate_bin_weight(bin_counts)

	# 5) load training arrays 
	FEATURES = Static_Features + Dynamic_Features
	X_train, y_train, w_train = load_training_arrays(
		train_case_ids, FEATURES, case_len_map, bin_w_lookup, clinical_map
	)
	
	# 6) Load Validation Arrays 
	val_case_ids = collect_case_ids("Validation")
	X_val, y_val = load_split_arrays(val_case_ids, FEATURES, clinical_map)

	# 7) Build Baseline XGBoost Model
	model = xgb.XGBRegressor(
		objective="reg:squarederror",
		eval_metric="mae",
		n_estimators=5000,
		learning_rate=0.05,
		max_depth=6,
		subsample=0.8,
		colsample_bytree=0.8,
		reg_lambda=1.0,
		random_state=42,
		n_jobs=-1,
		early_stopping_rounds=100, 
	)
	# 8) Train with Early Stopping on Validation
	model.fit(
		X_train, y_train,
		sample_weight=w_train,
		eval_set=[(X_val, y_val)],
		verbose=50,
	)
	# 9) Evaluate on Validation
	y_val_pred = model.predict(X_val)
	print_metrics("VAL", y_val, y_val_pred)
	print_metrics_by_bin("VAL", y_val, y_val_pred)
	print_metrics_by_sex("VAL", X_val, y_val, y_val_pred, FEATURES)
	
	# 10)Load Test Arrays
	test_case_ids = collect_case_ids("Testing")
	X_test, y_test = load_split_arrays(test_case_ids, FEATURES, clinical_map)

	y_test_pred = model.predict(X_test)
	print_metrics("TEST", y_test, y_test_pred)
	print_metrics_by_bin("TEST", y_test, y_test_pred)
	print_metrics_by_sex("TEST", X_test, y_test, y_test_pred, FEATURES)




