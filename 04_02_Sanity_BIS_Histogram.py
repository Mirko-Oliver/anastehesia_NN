"""
BIS/BIS distribution across ALL cases and plot a stacked histogram by label.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"

BIS_COL = "BIS/BIS"
LABEL_COL = "label"

LABEL_REGULAR = "regular"
LABEL_INVALID = "invalid"
LABEL_IRREGULAR = "irregular"


def discover_case_paths(data_dir=DATA_DIR, suffix=SUFFIX):
    files = [f for f in os.listdir(data_dir) if f.endswith(suffix)]
    # sort by numeric case id if possible
    def _key(fname):
        try:
            return int(fname.split("_")[0])
        except Exception:
            return fname
    return [os.path.join(data_dir, f) for f in sorted(files, key=_key)]


def _clean_bis_discrete(bis: np.ndarray) -> np.ndarray:
    """
    Treat BIS as discrete integer 0..100:
    - drop NaN
    """
    bis = bis[np.isfinite(bis)]
    if bis.size == 0:
        return bis.astype(np.int64)
    bis = np.rint(bis).astype(np.int64)
    return np.clip(bis, 0, 100)


def main():
	paths = discover_case_paths(DATA_DIR, SUFFIX)
	if not paths:
		raise RuntimeError(f"No *{SUFFIX} files found in {DATA_DIR}")

	# counts[label, bis_value]
	counts_regular = np.zeros(101, dtype=np.int64)
	counts_invalid = np.zeros(101, dtype=np.int64)
	counts_irregular = np.zeros(101, dtype=np.int64)

	processed = 0

	for path in paths:
		df = pd.read_parquet(path, columns=[BIS_COL, LABEL_COL])
		bis = pd.to_numeric(df[BIS_COL], errors="coerce").to_numpy(dtype=float)
		labels = df[LABEL_COL].astype(str).to_numpy()

		bis_int = _clean_bis_discrete(bis)

		# Align labels to the same filtering used for bis_int:
		finite_mask = np.isfinite(bis)
		labels = labels[finite_mask]

		# Accumulate counts per label
		for lab, target in (
			(LABEL_REGULAR, counts_regular),
			(LABEL_INVALID, counts_invalid),
			(LABEL_IRREGULAR, counts_irregular),
		):
			m = labels == lab
			if np.any(m):
				vals = bis_int[m]
				target += np.bincount(vals, minlength=101).astype(np.int64)

		processed += 1

	# Plot stacked bars per BIS value
	x = np.arange(101, dtype=int)

	# Bottom: regular, Middle: invalid, Top: irregular
	plt.figure(figsize=(16, 6))
	plt.bar(x, counts_regular, label="regular")
	plt.bar(x, counts_invalid, bottom=counts_regular, label="invalid")
	plt.bar(x, counts_irregular, bottom=counts_regular + counts_invalid, label="irregular")

	plt.title(f"Total BIS/BIS distribution across cases (processed={processed})")
	plt.xlabel("BIS/BIS (discrete 0–100)")
	plt.ylabel("Count")
	plt.xlim(-0.5, 100.5)
	plt.xticks(np.arange(0, 101, 5))
	plt.grid(True, axis="y", alpha=0.3)
	plt.legend(loc="upper right")

	# Print totals
	total_regular = int(counts_regular.sum())
	total_invalid = int(counts_invalid.sum())
	total_irregular = int(counts_irregular.sum())
	total_all = total_regular + total_invalid + total_irregular
	print("Totals:")
	print(f"  regular:   {total_regular:,}")
	print(f"  invalid:   {total_invalid:,}")
	print(f"  irregular: {total_irregular:,}")
	print(f"  all:       {total_all:,}")

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
    main()
