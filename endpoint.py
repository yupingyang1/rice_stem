import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tensorflow import keras

MODEL_PATH = "best_cnn.h5"
TRAIN_CSV  = "stem_data.csv"
SPEC_CSV = "data.csv"
SSI_CSV    = "ssi_table.csv"
USE_D2 = True
RIDGE_ALPHA = 100.0
N_PCS = 2
N_PERM = 5000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
def apply_D2(X: np.ndarray) -> np.ndarray:
    if not USE_D2:
        return X
    from Preprocessing_package import D2
    return D2(X)
def make_sample_ids(n: int) -> np.ndarray:
    return np.array([f"S{i:02d}" for i in range(1, n + 1)])
def make_groups(n: int) -> np.ndarray:
    if n != 24:
        raise ValueError(f"Expected n=24 for 4 groups x 6 samples, but got n={n}")
    g = ["CK"] * 6 + ["Cd1"] * 6 + ["Cd2"] * 6 + ["Cd3"] * 6
    return np.array(g)

def build_scaler_x_from_training_csv(train_csv: str) -> StandardScaler:
    df = pd.read_csv(train_csv)
    X_raw = df.iloc[:, :-1].values
    X_raw = apply_D2(X_raw)
    scaler_x = StandardScaler()
    scaler_x.fit(X_raw)
    return scaler_x

def load_12_spectra(spec_csv: str, scaler_x: StandardScaler) -> np.ndarray:
    X12_raw = pd.read_csv(spec_csv, header=None).values
    X12_raw = apply_D2(X12_raw)
    X12 = scaler_x.transform(X12_raw)
    return X12.reshape(X12.shape[0], X12.shape[1], 1)

def load_ssi(ssi_csv: str, expected_ids: np.ndarray) -> np.ndarray:
    ssi_df = pd.read_csv(ssi_csv)
    ssi_df = ssi_df.set_index("sample_id").loc[expected_ids].reset_index()
    return ssi_df["SSI"].values.astype(float)

def load_model_and_extractors(model_path: str):
    model = keras.models.load_model(model_path, compile=False)

    layer_names = [l.name for l in model.layers]
    required = ["att_logits", "att_weights", "merged"]
    for r in required:
        if r not in layer_names:
            raise ValueError(f"Layer {r} not found in model. Existing names: {layer_names[:50]} ...")
    logits_model = keras.Model(model.input, model.get_layer("att_logits").output)
    att_model    = keras.Model(model.input, model.get_layer("att_weights").output)
    merged_model = keras.Model(model.input, model.get_layer("merged").output)
    return model, logits_model, att_model, merged_model
def compute_endpointA(logits: np.ndarray, att: np.ndarray) -> pd.DataFrame:
    eps = 1e-12
    out = pd.DataFrame({
        "logit_VIS": logits[:, 0],
        "logit_RE":  logits[:, 1],
        "logit_NIR": logits[:, 2],
        "att_VIS":   att[:, 0],
        "att_RE":    att[:, 1],
        "att_NIR":   att[:, 2],
    })

    out["A_logit_VIS_minus_RE"] = out["logit_VIS"] - out["logit_RE"]
    out["A_logit_VIS_minus_NIR"] = out["logit_VIS"] - out["logit_NIR"]
    out["A_log_att_VIS_over_RE"] = np.log((out["att_VIS"] + eps) / (out["att_RE"] + eps))
    out["A_log_att_VIS_over_NIR"] = np.log((out["att_VIS"] + eps) / (out["att_NIR"] + eps))
    return out

def main():
    scaler_x = build_scaler_x_from_training_csv(TRAIN_CSV)

    X12_cnn = load_12_spectra(SPEC_CSV, scaler_x)
    n12 = X12_cnn.shape[0]
    sample_ids = make_sample_ids(n12)
    groups = make_groups(n12)
    ssi = load_ssi(SSI_CSV, sample_ids)

    model, logits_model, att_model, merged_model = load_model_and_extractors(MODEL_PATH)

    logits = logits_model.predict(X12_cnn, verbose=0)
    att    = att_model.predict(X12_cnn, verbose=0)
    merged = merged_model.predict(X12_cnn, verbose=0)

    endA_df = compute_endpointA(logits, att)
    endA_df.insert(0, "sample_id", sample_ids)
    endA_df.insert(1, "group", groups)
    endA_df["SSI"] = ssi

    for col in ["A_logit_VIS_minus_RE", "A_logit_VIS_minus_NIR", "A_log_att_VIS_over_RE", "A_log_att_VIS_over_NIR"]:
        r, p = spearmanr(endA_df[col].values, endA_df["SSI"].values)
        print(f"[EndpointA vs SSI] {col}: Spearman r={r:.3f}, p={p:.4g}")
    endA_df.to_csv("R4_endpointA_table.csv", index=False)
    print("Saved: R4_endpointA_table.csv")
    print("\nDone.")
if __name__ == "__main__":
    main()