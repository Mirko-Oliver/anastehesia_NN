import pandas as pd
import numpy as np
import os
""" manualy download clinical info csv from vitalbd,
this script filters any caseids out of the clinical info csv that are not part of the dataset"""
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"


clinical_path = os.path.join(DATA_DIR, CLINICAL_INFO)
df = pd.read_csv(clinical_path)


case_ids = {
    int(f.split("_")[0])
    for f in os.listdir(DATA_DIR)
    if f.endswith(SUFFIX)
}
print(len(case_ids))

def add_lbm(df, weight_col="weight", height_col="height", sex_col="sex", out_col="lbm"):
    """
    Adds a lean body mass (lbm) column to the dataframe.
        Male (M):   1.0 * weight - 128 * (weight / height)^2
        Female (F): 1.07 * weight - 148 * (weight / height)^2

    """
    df = df.copy()
    weight = pd.to_numeric(df[weight_col], errors="coerce")
    height = pd.to_numeric(df[height_col], errors="coerce")

    ratio_sq = (weight / height) ** 2

    sex = df[sex_col].astype(str).str.strip().str.upper()

    # Compute LBM
    df[out_col] = np.where(
        sex == "M",
        1.0 * weight - (128 * ratio_sq),
        np.where(
            sex == "F",
            1.07 * weight - (148 * ratio_sq),
            np.nan  
        )
    )

    return df


df["caseid"] = pd.to_numeric(df["caseid"], errors="coerce").astype("Int64")
df_filtered = df[df['caseid'].isin(case_ids)]
df_filtered = add_lbm(df_filtered) # ADD LBM Column
df_filtered.to_csv(clinical_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Remaining rows: {len(df_filtered)}")


