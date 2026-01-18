import pandas as pd
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

df["caseid"] = pd.to_numeric(df["caseid"], errors="coerce").astype("Int64")
df_filtered = df[df['caseid'].isin(case_ids)]
df_filtered.to_csv(clinical_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Remaining rows: {len(df_filtered)}")


