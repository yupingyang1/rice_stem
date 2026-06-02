import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, Lambda, concatenate
from keras.layers import Softmax, Multiply
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv("stem_data.csv")
X_raw = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42)
from Preprocessing_package import D2
X_train_clean = D2(X_train_raw)
X_test_clean = D2(X_test_raw)
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_clean)
X_test_scaled = scaler_x.transform(X_test_clean)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
X_train = X_train_scaled
X_test = X_test_scaled
y_train = y_train_scaled
y_test = y_test_scaled
n_samples, n_bands = X_train.shape
print("n_bands =", n_bands)
X_train_cnn = X_train.reshape(X_train.shape[0], n_bands, 1)
X_test_cnn  = X_test.reshape(X_test.shape[0],  n_bands, 1)
wavelengths = np.linspace(388.34, 1036.34, n_bands)
vis_idx = np.where((wavelengths >= 388.34) & (wavelengths <= 700))[0].tolist()
re_idx  = np.where((wavelengths > 700) & (wavelengths <= 850))[0].tolist()
nir_idx = np.where((wavelengths > 850))[0].tolist()
def slice_idx(x, idx):
    return tf.gather(x, idx, axis=1)
def branch_block(x_slice, name):
    x = Conv1D(32, 5, padding='same', activation='relu', name=f'{name}_conv1')(x_slice)
    x = Conv1D(32, 5, padding='same', activation='relu', name=f'{name}_conv2')(x)
    x = Flatten(name=f'{name}_flat')(x)
    x = Dense(32, activation='relu', name=f'{name}_dense2')(x)
    return x
def build_model():
    inp = Input(shape=(n_bands,1), name='spec_input')
    vis = Lambda(slice_idx, arguments={'idx': vis_idx}, name='slice_vis')(inp)
    re  = Lambda(slice_idx, arguments={'idx': re_idx},  name='slice_re')(inp)
    nir = Lambda(slice_idx, arguments={'idx': nir_idx}, name='slice_nir')(inp)
    f_vis = branch_block(vis, 'vis')
    f_re  = branch_block(re,  're')
    f_nir = branch_block(nir, 'nir')
    logits = concatenate([
        Dense(1, name='vis_logit')(f_vis),
        Dense(1, name='re_logit')(f_re),
        Dense(1, name='nir_logit')(f_nir)
    ], axis=1, name='att_logits')
    att = Softmax(axis=1, name='att_weights')(logits)
    α_vis = Lambda(lambda x: x[:,0:1], name='alpha_vis')(att)
    α_re  = Lambda(lambda x: x[:,1:2], name='alpha_re')(att)
    α_nir = Lambda(lambda x: x[:,2:3], name='alpha_nir')(att)
    f_vis_w = Multiply(name='vis_weighted')([f_vis, α_vis])
    f_re_w  = Multiply(name='re_weighted')([f_re, α_re])
    f_nir_w = Multiply(name='nir_weighted')([f_nir, α_nir])
    merged = concatenate([f_vis_w, f_re_w, f_nir_w], name='merged')
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='linear', name='output')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model
model = build_model()
model.summary()
es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15)
history = model.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_cnn, y_test),
    callbacks=[es, rl],
    verbose=1
)
y_train_pred = model.predict(X_train_cnn).flatten()
y_test_pred  = model.predict(X_test_cnn).flatten()
y_train_pred_real = scaler_y.inverse_transform(y_train_pred.reshape(-1,1)).ravel()
y_test_pred_real  = scaler_y.inverse_transform(y_test_pred.reshape(-1,1)).ravel()
y_train_real = scaler_y.inverse_transform(y_train.reshape(-1,1)).ravel()
y_test_real  = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
train_rmse = mean_squared_error(y_train_real, y_train_pred_real) ** 0.5
test_rmse = mean_squared_error(y_test_real, y_test_pred_real) ** 0.5
print("\n===== Final Performance =====")
print("Train R2:", r2_score(y_train_real, y_train_pred_real))
print("Test  R2:", r2_score(y_test_real,  y_test_pred_real))
print("Train Rmse:", train_rmse)
print("Test  Rmse:", test_rmse)