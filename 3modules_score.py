import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from Preprocessing_package import D2

MODEL_PATH = "best_cnn.h5"
ATT_LAYER_NAME = "att_weights"
DATA_PATH = "data.csv"
META_PATH = "meta.csv"
DE = [
    "(+)-Pinoresinol",
    "Sinapyl alcohol",
    "Naringin",
    "Ononin",
    "Trifolirhizin",
    "Tyramine",
    "Melilotoside",
]

DA = [
    "Trans-2-Octenal",
    "9-Oxo-nonanoic acid",
    "9,10-eot",
    "Lpc(18:3)",
    "LysoPC(18:1(11Z))",
    "Lpc(16:1)",
]

RE = [
    "Oxiglutatione",
    "1,4-diaminobutane",
    "L-(+)-Arginine",
    "Feruloylputrescine",
]

df_data = pd.read_csv(DATA_PATH, header=None)
X_raw = df_data.values
print("Raw spectra shape:", X_raw.shape)
X_raw = D2(X_raw)
print("After D2 shape:", X_raw.shape)
n_samples = X_raw.shape[0]
sample_id = [f"S{i+1:02d}" for i in range(n_samples)]
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
expected_bands = model.input_shape[1]
print("Model expected bands:", expected_bands)
if X_raw.shape[1] > expected_bands:
    X_raw = X_raw[:, :expected_bands]
elif X_raw.shape[1] < expected_bands:
    pad = expected_bands - X_raw.shape[1]
    X_raw = np.pad(X_raw, ((0, 0), (0, pad)), mode="edge")
print("Aligned spectra shape:", X_raw.shape)

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X_raw)
X_cnn = X_scaled.reshape(n_samples, expected_bands, 1)

att_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(ATT_LAYER_NAME).output)
att_raw = att_model.predict(X_cnn, verbose=0)
print("Attention raw shape:", att_raw.shape)

att_raw_df = pd.DataFrame(att_raw, columns=["att_1", "att_2", "att_3"])
att_raw_df.insert(0, "sample_id", sample_id)
att_raw_df.to_csv("attention_raw_neutral.csv", index=False)

att_df = pd.DataFrame({
    "sample_id": sample_id,
    "att_VIS": att_raw_df["att_1"].values,
    "att_RE":  att_raw_df["att_2"].values,
    "att_NIR": att_raw_df["att_3"].values
})
att_df.to_csv("attention_per_sample.csv", index=False)
att_df.to_excel("attention_per_sample.xlsx", index=False)
print("Saved: attention_per_sample.*")

meta = pd.read_csv(META_PATH, header=0)

if len(meta) != n_samples:
    print(f"meta rows={len(meta)} but spectra samples={n_samples}; align to min length.")
    min_n = min(len(meta), n_samples)
    meta = meta.iloc[:min_n].copy()
    att_df = att_df.iloc[:min_n].copy()
    sample_id = sample_id[:min_n]
    n_samples = min_n

meta.insert(0, "sample_id", sample_id)

def check_cols(df, cols, tag):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Missing {tag} metabolites in meta2.csv:", missing)
    return [c for c in cols if c in df.columns]

DE = check_cols(meta, DE, "DE")
DA   = check_cols(meta, DA,   "DA")
RE   = check_cols(meta, RE,   "RE")
meta_num = meta.drop(columns=["sample_id"]).copy()
eps = 1e-8
z = (meta_num - meta_num.mean(axis=0)) / (meta_num.std(axis=0) + eps)
z.insert(0, "sample_id", meta["sample_id"])
scores = pd.DataFrame({
    "sample_id": meta["sample_id"],
    "De_score": z[DE].mean(axis=1) if DE else np.nan,
    "Da_score":   z[DA].mean(axis=1) if DA else np.nan,
    "Re_score":   z[RE].mean(axis=1) if RE else np.nan,
})

scores.to_csv("module_scores_3modules.csv", index=False)
scores.to_excel("module_scores_3modules.xlsx", index=False)
print("Saved: module_scores_3modules.*")
bridge = pd.merge(att_df, scores, on="sample_id", how="inner")
bridge.to_csv("bridge_att_scores_table_3modules.csv", index=False)
bridge.to_excel("bridge_att_scores_table_3modules.xlsx", index=False)
print("Saved: bridge_att_scores_table_3modules.*")
att_cols = ["att_VIS", "att_RE", "att_NIR"]
score_cols = ["De_score", "Da_score", "Re_score"]
rows = []
for a in att_cols:
    for s in score_cols:
        if bridge[a].isna().all() or bridge[s].isna().all():
            continue
        r_s, p_s = spearmanr(bridge[a], bridge[s])
        r_p, p_p = pearsonr(bridge[a], bridge[s])
        rows.append([a, s, r_s, p_s, r_p, p_p])
corr_res = pd.DataFrame(rows, columns=[
    "attention", "score",
    "spearman_r", "spearman_p",
    "pearson_r", "pearson_p"
])
corr_res.to_csv("att_score_correlation_3modules.csv", index=False)
corr_res.to_excel("att_score_correlation_3modules.xlsx", index=False)
print("Saved: att_score_correlation_3modules.*")
print(corr_res.sort_values("spearman_p").head(12))
pairs_to_plot = [
    ("att_VIS", "De_score"),
    ("att_VIS", "Da_score"),
    ("att_RE",  "Re_score"),
]
for a, s in pairs_to_plot:
    if a not in bridge.columns or s not in bridge.columns:
        continue
    r_s, p_s = spearmanr(bridge[a], bridge[s])
    plt.figure(figsize=(5.2, 4.6))
    plt.scatter(bridge[a], bridge[s], alpha=0.85, edgecolor="k")
    plt.xlabel(a)
    plt.ylabel(s)
    plt.title(f"{a} vs {s}\nSpearman r={r_s:.2f}, p={p_s:.3g}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Fig_{a}_vs_{s}.png", dpi=300)
plt.close("all")
print("Done")
