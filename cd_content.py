import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

R4_TABLE = "R4_endpointA_table.csv"
CD_TABLE = "Cd_stem.csv"
ENDPOINT_COL = "A_logit_VIS_minus_RE"
FIG_MATCHED_LOAD = "Fig_SI_matched_load_strip.png"
MATCHED_LOAD_DATA = "matched_load_strip_data.csv"

def main():
    df_r4 = pd.read_csv(R4_TABLE)
    df_cd = pd.read_csv(CD_TABLE)
    required_r4 = {"sample_id", "group", "SSI", ENDPOINT_COL}
    missing = required_r4 - set(df_r4.columns)
    if missing:
        raise ValueError(f"Missing columns in {R4_TABLE}: {missing}")
    if not {"sample_id", "Cd_stem"}.issubset(df_cd.columns):
        raise ValueError(f"{CD_TABLE} must contain columns: sample_id, Cd_stem")

    df = df_r4.merge(df_cd[["sample_id", "Cd_stem"]], on="sample_id", how="inner").copy()
    df["SSI"] = df["SSI"].astype(float)
    df["Cd_stem"] = df["Cd_stem"].astype(float)
    df[ENDPOINT_COL] = df[ENDPOINT_COL].astype(float)
    df["group"] = df["group"].astype(str)
    if set(df["group"].unique()) - {"CK", "Cd3"}:
        print("Warning: group is not exactly {CK, Cd3}. Current groups:", df["group"].unique())
    m_load = smf.ols("SSI ~ Cd_stem + C(group)", data=df).fit()
    m_plus = smf.ols(f"SSI ~ Cd_stem + C(group) + {ENDPOINT_COL}", data=df).fit()
    m_int  = smf.ols(f"SSI ~ Cd_stem + C(group) + {ENDPOINT_COL} + {ENDPOINT_COL}:C(group)", data=df).fit()
    delta_r2 = m_plus.rsquared - m_load.rsquared
    ftest_plus_vs_load = m_plus.compare_f_test(m_load)  # (F, p, df_diff)
    ftest_int_vs_plus = m_int.compare_f_test(m_plus)
    rows = []
    def add_model_row(name, model):
        rows.append({
            "model": name,
            "n": int(model.nobs),
            "R2": float(model.rsquared),
            "adj_R2": float(model.rsquared_adj),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "coef_Cd_stem": float(model.params.get("Cd_stem", np.nan)),
            "p_Cd_stem": float(model.pvalues.get("Cd_stem", np.nan)),
            "coef_endpointA": float(model.params.get(ENDPOINT_COL, np.nan)),
            "p_endpointA": float(model.pvalues.get(ENDPOINT_COL, np.nan)),
        })

    add_model_row("Load: SSI ~ Cd_stem + group", m_load)
    add_model_row("Load+Readout: + endpointA", m_plus)
    add_model_row("Load+Readout+Interaction", m_int)
    out_models = pd.DataFrame(rows)
    out_models["delta_R2_plus_vs_load"] = np.nan
    out_models.loc[out_models["model"].str.contains("Load\\+Readout: \\+ endpointA"), "delta_R2_plus_vs_load"] = delta_r2
    out_models["Ftest_plus_vs_load_F"] = np.nan
    out_models["Ftest_plus_vs_load_p"] = np.nan
    out_models.loc[out_models["model"].str.contains("Load\\+Readout: \\+ endpointA"), "Ftest_plus_vs_load_F"] = float(ftest_plus_vs_load[0])
    out_models.loc[out_models["model"].str.contains("Load\\+Readout: \\+ endpointA"), "Ftest_plus_vs_load_p"] = float(ftest_plus_vs_load[1])
    out_models["Ftest_int_vs_plus_F"] = np.nan
    out_models["Ftest_int_vs_plus_p"] = np.nan
    out_models.loc[out_models["model"].str.contains("Interaction"), "Ftest_int_vs_plus_F"] = float(ftest_int_vs_plus[0])
    out_models.loc[out_models["model"].str.contains("Interaction"), "Ftest_int_vs_plus_p"] = float(ftest_int_vs_plus[1])
    df_sorted = df.sort_values(["group", "Cd_stem"]).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    cd_norm = (df_sorted["Cd_stem"] - df_sorted["Cd_stem"].mean()) / (df_sorted["Cd_stem"].std(ddof=0) + 1e-12)
    plot_data = pd.DataFrame({
        'sample_id': df_sorted['sample_id'].values,
        'group': df_sorted['group'].values,
        'position': x,
        'Cd_stem': df_sorted['Cd_stem'].values,
        'Cd_stem_zscore': cd_norm.values,
        'SSI': df_sorted['SSI'].values,
        'endpointA': df_sorted[ENDPOINT_COL].values
    })
    grp = df_sorted["group"].values
    change_idx = np.where(grp[:-1] != grp[1:])[0]
    plot_data['group_boundary'] = False
    plot_data.loc[plot_data['position'].isin(change_idx + 0.5), 'group_boundary'] = True
    plot_data.to_csv(MATCHED_LOAD_DATA, index=False)
    print(f"Saved: {MATCHED_LOAD_DATA}")
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(x, cd_norm.values, marker="o", linewidth=1, label="Cd_stem (z-scored)")
    plt.plot(x, df_sorted["SSI"].values, marker="o", linewidth=1, label="SSI")
    grp = df_sorted["group"].values
    change_idx = np.where(grp[:-1] != grp[1:])[0]
    for idx in change_idx:
        plt.axvline(idx + 0.5, linewidth=1)
    plt.xticks(x, df_sorted["sample_id"].values, rotation=45, ha="right")
    plt.xlabel("Samples ordered within group by Cd_stem")
    plt.ylabel("Value")
    plt.title("Matched-load illustration: load vs buffering state")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_MATCHED_LOAD, dpi=600)
    plt.close()
    print("Saved:", FIG_MATCHED_LOAD)
    print("\nKey results:")
    print(f"Model-Load R2 = {m_load.rsquared:.3f}")
    print(f"Model-Load+Readout R2 = {m_plus.rsquared:.3f}, delta_R2 = {delta_r2:.3f}")
    print(f"F-test (plus vs load): F={ftest_plus_vs_load[0]:.3f}, p={ftest_plus_vs_load[1]:.4g}")
    print(f"F-test (interaction vs plus): F={ftest_int_vs_plus[0]:.3f}, p={ftest_int_vs_plus[1]:.4g}")
if __name__ == "__main__":
    main()
