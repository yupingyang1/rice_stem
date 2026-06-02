import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ==========
# Settings
# ==========
INFILE = "R4_endpointA_table.csv"
ENDPOINT_COL = "A_logit_VIS_minus_RE"  # 改成你主文B1用的端点列
TARGET_COL = "SBI"

df = pd.read_csv(INFILE)

# 基本检查
required = {"sample_id", ENDPOINT_COL, TARGET_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# 全体 Spearman
r_all, p_all = spearmanr(df[ENDPOINT_COL].values, df[TARGET_COL].values)

rows = []
for sid in df["sample_id"].values:
    sub = df[df["sample_id"] != sid].copy()

    r, p = spearmanr(sub[ENDPOINT_COL].values, sub[TARGET_COL].values)

    rows.append({
        "deleted_sample_id": sid,
        "n_used": len(sub),
        "spearman_r": r,
        "spearman_p": p
    })

out = pd.DataFrame(rows)

# 汇总指标，便于写SI一句话
summary = {
    "endpoint": ENDPOINT_COL,
    "n_total": len(df),
    "spearman_r_all": r_all,
    "spearman_p_all": p_all,
    "r_min": out["spearman_r"].min(),
    "r_median": out["spearman_r"].median(),
    "r_max": out["spearman_r"].max(),
    "r_sd": out["spearman_r"].std(ddof=1),
    "most_influential_delete": out.loc[(out["spearman_r"] - r_all).abs().idxmax(), "deleted_sample_id"],
    "max_abs_delta_r": float((out["spearman_r"] - r_all).abs().max())
}

out.to_csv("SI_delete1_spearman_table.csv", index=False)
pd.DataFrame([summary]).to_csv("SI_delete1_spearman_summary.csv", index=False)

print("Saved: SI_delete1_spearman_table.csv")
print("Saved: SI_delete1_spearman_summary.csv")
print(summary)
