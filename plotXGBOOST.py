"""
Plot XGBoost BIS prediction vs. actual BIS and EMG for a single case.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib.patches import Patch

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"
TARGET_COL = "BIS/BIS"

Static_Features = ["age", "sex"]
CASE_ID = 1614


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

def add_static_features(df, caseid, clinical_map):
    """
    Adds static columns to every row of df for this caseid.
    """
    if str(caseid) not in clinical_map:
        raise KeyError(f"caseid {caseid} not found in clinical_map")

    statics = clinical_map[str(caseid)]
    for k, v in statics.items():
        df[k] = v
    return df

def coerce_static_types(df):
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.strip().str.lower().map({"m": 1, "f": 0})
    return df

def load_model_bundle(model_dir = "model"):
	model_path = os.path.join(model_dir, "bis_xgb_model.json")
	meta_path  = os.path.join(model_dir, "bis_xgb_meta.json")

	model = xgb.XGBRegressor()
	model.load_model(model_path)

	with open(meta_path, "r", encoding="utf-8") as f:
		meta = json.load(f)

	return model, meta

def shade_background(ax, t, mask, color, alpha, zorder = 0):
    mask = np.asarray(mask, bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return

    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)

    for seg in segments:
        ax.axvspan(t[seg[0]], t[seg[-1]], color = color, alpha=alpha, zorder = 0)

def plot_case(caseid, model_dir = "model"):
	# 1) Load model + metadata (
	model, meta = load_model_bundle(model_dir)
	features = meta["features"]

	# 2) Load clinical map for statics
	clinical_map = load_clinical_info_map(Static_Features)

	# 3) Load case parquet
	path = os.path.join(DATA_DIR, str(caseid) + SUFFIX)
	df = pd.read_parquet(path)
	df = add_static_features(df, caseid, clinical_map)
	df = coerce_static_types(df)

	# 4) Build X, y, emg
	t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
	bis_true = pd.to_numeric(df["BIS/BIS"], errors="coerce").to_numpy(dtype=float)
	emg = pd.to_numeric(df["BIS/EMG"], errors="coerce").to_numpy(dtype=float)
	labels = df["label"].astype(str).to_numpy()
	irr_mask = labels == "irregular"
	
	X = df[features].to_numpy(dtype=np.float32)
	bis_pred = model.predict(X)



	x_label = "Time"
	fig, (ax, ax_rib) = plt.subplots(
		2, 1,
		figsize=(14, 5.2),
		sharex=True,
		gridspec_kw={"height_ratios": [6, 1], "hspace": 0.05},
	)

	line_bis, = ax.plot(t, bis_true, label="Actual BIS/BIS")
	line_pred, = ax.plot(t, bis_pred, label="Predicted BIS/BIS")

	ax.set_xlabel(x_label)
	ax.set_ylabel("BIS")

	ax2 = ax.twinx()
	line_emg, = ax2.plot(t, emg, label="EMG", color="brown", alpha=0.5)
	ax2.set_ylabel("EMG")


	# Ribbon PLot
	ax_rib.set_ylim(0, 1)
	ax_rib.set_yticks([])
	ax_rib.set_ylabel("")  
	ax_rib.grid(False)
	shade_background(ax_rib, t, irr_mask, color="#990000", alpha=.25, zorder=1)

	ax_rib.spines["bottom"].set_linewidth(1.5)
	ax_rib.spines["bottom"].set_color("black")
	ax_rib.spines["left"].set_linewidth(1.5)
	ax_rib.spines["left"].set_color("black")
	
	proxy_irr = Patch(facecolor="#990000", alpha=0.25, label="Irregulärer EMG")
	ax.legend(
		handles=[line_bis, line_emg, proxy_irr],
		loc="center left",
		bbox_to_anchor=(1.02, 0.5),    
		borderaxespad=0.0,
		frameon=False,
		fontsize=9,
	)
	ticks = ax.get_xticks()
	labels = [f"{int(t)}" for t in ticks]
	labels[-2] = "s"
	
	ax.set_xticks(ticks)
	ax.set_xticklabels(labels)
	fig.tight_layout(rect=[0, 0.06, 0.9, 1])
	plt.show()

if __name__ == "__main__":
	plot_case(CASE_ID, model_dir="model")
