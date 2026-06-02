import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("module_scores_3modules.csv")
module_cols = ["De_score", "Da_score", "Re_score"]
X = df[module_cols].values
sc = StandardScaler()
Z = sc.fit_transform(X)
pca = PCA(n_components=1, random_state=0)
ssi = pca.fit_transform(Z).ravel()
group = np.array(["CK"] * 6 + ["Cd1"] * 6 + ["Cd2"] * 6 + ["Cd3"] * 6)
if ssi[group == "Cd3"].mean() < ssi[group == "CK"].mean():
    ssi = -ssi
df_out = df[["sample_id"]].copy()
df_out["group"] = group
df_out["SSI"] = ssi
loadings = pca.components_.ravel()
df_load = pd.DataFrame({"module": module_cols, "loading_PC1": loadings})
df_out.to_csv("ssi_table.csv", index=False)
df_load.to_csv("ssi_module_loadings.csv", index=False)
print("Saved: ssi_table.csv, ssi_module_loadings.csv")
