import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
spec_df = pd.read_csv("data.csv", header=None)
X_raw = spec_df.values
print("shape:", X_raw.shape)
from Preprocessing_package import D2
X_raw = D2(X_raw)
n_samples, n_bands = X_raw.shape
print("D2 shape:", X_raw.shape)
wavelengths = np.linspace(388.34, 1036.34, n_bands)
module_df = pd.read_csv("module_scores_3modules.csv")
sample_ids = module_df.iloc[:, 0].values
Y_raw = module_df.iloc[:, 1:].values
module_names = module_df.columns[1:].tolist()
n_modules = Y_raw.shape[1]
print("module shape:", Y_raw.shape, "module name:", module_names)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = scaler_X.fit_transform(X_raw)
Y = scaler_Y.fit_transform(Y_raw)
n_components = 1
pls = PLSRegression(n_components=n_components)
pls.fit(X, Y)
X_loadings = pls.x_loadings_
Y_loadings = pls.y_loadings_
X_scores   = pls.x_scores_
Y_scores   = pls.y_scores_
print("X_loadings shape:", X_loadings.shape)
print("Y_loadings shape:", Y_loadings.shape)

x_load1 = X_loadings[:, 0]
y_load1 = Y_loadings[:, 0]
t1      = X_scores[:, 0]
u1      = Y_scores[:, 0]

with pd.ExcelWriter("JLM_outputs.xlsx") as writer:
    df_xload = pd.DataFrame({
        "wavelength_nm": wavelengths,
        "loading_comp1": x_load1
    })
    df_xload.to_excel(writer, sheet_name="X_loading_comp1", index=False)
    df_yload = pd.DataFrame({
        "module": module_names,
        "loading_comp1": y_load1
    })
    df_yload.to_excel(writer, sheet_name="Y_loading_comp1", index=False)
    groups = ["CK"] * 6 + ["Cd1"] * 6 + ["Cd2"] * 6 + ["Cd3"] * 6
    if len(groups) != n_samples:
        groups = ["Group"] * n_samples

    df_scores = pd.DataFrame({
        "sample_id": sample_ids,
        "group": groups,
        "t1_X_score": t1,
        "u1_Y_score": u1
    })
    df_scores.to_excel(writer, sheet_name="Scores_comp1", index=False)

print("JLM saved JLM_outputs.xlsx")

plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Arial"

plt.figure(figsize=(7, 4))
plt.plot(wavelengths, x_load1, linewidth=1.8)

plt.axvline(400, color="grey", linestyle="--", alpha=0.5)
plt.axvline(700, color="grey", linestyle="--", alpha=0.5)
plt.axvline(850, color="grey", linestyle="--", alpha=0.5)

ymax = np.max(x_load1)
ymin = np.min(x_load1)
ytxt = ymin + 0.9 * (ymax - ymin)

plt.text(450, ytxt, "VIS", ha="center")
plt.text(775, ytxt, "RE",  ha="center")
plt.text(950, ytxt, "NIR", ha="center")

plt.xlabel("Wavelength (nm)", fontsize=12)

plt.figure(figsize=(5, 4))
x_pos = np.arange(n_modules)
plt.bar(x_pos, y_load1)
plt.xticks(x_pos, module_names, rotation=30, ha="right")
plt.ylabel("Loading (Component 1)", fontsize=12)
plt.title("Module loadings on joint component", fontsize=13)
plt.tight_layout()
plt.savefig("JLM_Y_loading_comp1.png")
plt.close()
print("saved：JLM_Y_loading_comp1.png")
marker_map = {"CK": "o", "Cd1": "^", "Cd2": "s", "Cd3": "D"}
group_order = ["CK", "Cd1", "Cd2", "Cd3"]
plt.figure(figsize=(5, 4))

for g in group_order:
    mask = np.array(groups) == g
    if np.any(mask):
        plt.scatter(
            t1[mask], u1[mask],
            marker=marker_map[g],
            label=g,
            alpha=0.8,
            edgecolors="k",
            s=60
        )

plt.xlabel("X-score t1 (spectral block)", fontsize=12)
plt.ylabel("Y-score u1 (module block)", fontsize=12)
plt.title("Joint component scores (Comp 1)", fontsize=13)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("JLM_scores_comp1_t1_vs_u1.png")
plt.close()
print("saved：JLM_scores_comp1_t1_vs_u1.png")

plt.figure(figsize=(5, 4))
data_groups = [t1[np.array(groups) == g] for g in group_order]

plt.boxplot(
    data_groups,
    labels=group_order,
    showfliers=False
)
plt.ylabel("X-score t1", fontsize=12)
plt.title("t1 scores across four groups", fontsize=13)
plt.tight_layout()
plt.savefig("JLM_t1_group_boxplot.png")
plt.close()
print("saved：JLM_t1_group_boxplot.png")