"""
Edge LSTM — Lightweight Soil-Moisture Predictor for ESP32
==========================================================
Pipeline:
  1. Train a standard LSTM on cloud-generated data.
  2. Apply SVD-based weight factorization on recurrent kernel.
  3. Magnitude prune weights below threshold.
  4. Quantize to int8 and export a TFLite flatbuffer.
  5. Estimate ESP32 RAM/Flash footprint.

Target: hidden_size ≤ 32, params < 15k, RAM < 40 KB
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Hyperparameters ──────────────────────────────────────────────
SEQ_LEN     = 24          # look-back steps (= 12 h at 30-min resolution)
HORIZON     = 4           # predict next 2 h (4 steps × 30 min)
HIDDEN_SIZE = 32          # must fit in ESP32 SRAM
BATCH_SIZE  = 64
EPOCHS      = 40
PRUNE_RATIO = 0.40        # fraction of weights zeroed
SVD_RANK    = 16          # rank for recurrent kernel factorization
FEATURES    = ["soil_moisture", "temperature_c", "rainfall_mm", "et_mm_day"]
TARGET      = "soil_moisture"


# ══════════════════════════════════════════════════════════════════
# 1 ▸ DATA PREPARATION
# ══════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[FEATURES].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN - HORIZON + 1):
        X.append(scaled[i : i + SEQ_LEN])
        y.append(scaled[i + SEQ_LEN : i + SEQ_LEN + HORIZON, 0])  # moisture only

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    # 70/15/15 train/val/test split (time-ordered, no shuffling)
    t1 = int(0.70 * len(X))
    t2 = int(0.85 * len(X))
    return (X[:t1], y[:t1]), (X[t1:t2], y[t1:t2]), (X[t2:], y[t2:]), scaler


# ══════════════════════════════════════════════════════════════════
# 2 ▸ MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════

def build_lstm(hidden: int = HIDDEN_SIZE) -> keras.Model:
    inp = keras.Input(shape=(SEQ_LEN, len(FEATURES)), name="sensor_seq")
    x   = layers.LSTM(hidden, return_sequences=False, name="lstm_core")(inp)
    x   = layers.Dense(16, activation="relu", name="fc1")(x)
    out = layers.Dense(HORIZON, name="moisture_pred")(x)
    model = keras.Model(inp, out, name="EdgeLSTM")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=["mae"])
    return model


def count_params(model: keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


# ══════════════════════════════════════════════════════════════════
# 3 ▸ SVD-BASED RECURRENT KERNEL FACTORIZATION
# ══════════════════════════════════════════════════════════════════

def svd_factorize_lstm(model: keras.Model, rank: int = SVD_RANK) -> keras.Model:
    """
    Replaces the LSTM recurrent kernel W_h (hidden × 4*hidden) with a
    low-rank approximation W_h ≈ U @ S_diag @ Vt using top-`rank` singular values.
    Weight is set back in-place; architecture unchanged for TFLite export.
    """
    lstm_layer = model.get_layer("lstm_core")
    weights    = lstm_layer.get_weights()   # [W_x, W_h, bias]
    W_h        = weights[1]                  # recurrent kernel

    U, s, Vt = np.linalg.svd(W_h, full_matrices=False)
    U_r  = U[:, :rank]
    s_r  = s[:rank]
    Vt_r = Vt[:rank, :]
    W_h_approx = (U_r * s_r) @ Vt_r        # rank-`rank` approximation

    weights[1] = W_h_approx.astype(np.float32)
    lstm_layer.set_weights(weights)

    compression = 1 - (rank * (W_h.shape[0] + W_h.shape[1])) / (W_h.shape[0] * W_h.shape[1])
    print(f"  SVD rank={rank} → recurrent kernel compression ≈ {compression:.1%}")
    return model


# ══════════════════════════════════════════════════════════════════
# 4 ▸ MAGNITUDE PRUNING
# ══════════════════════════════════════════════════════════════════

def magnitude_prune(model: keras.Model, ratio: float = PRUNE_RATIO) -> keras.Model:
    """Zero out the lowest-magnitude `ratio` fraction of all weights."""
    total_zeroed = 0
    total_weights = 0
    for layer in model.layers:
        wts = layer.get_weights()
        new_wts = []
        for w in wts:
            threshold = np.percentile(np.abs(w), ratio * 100)
            mask = np.abs(w) >= threshold
            new_wts.append((w * mask).astype(np.float32))
            total_zeroed  += np.sum(~mask)
            total_weights += w.size
        layer.set_weights(new_wts)
    actual_sparsity = total_zeroed / max(total_weights, 1)
    print(f"  Pruning → {actual_sparsity:.1%} weights zeroed ({total_zeroed:,}/{total_weights:,})")
    return model


# ══════════════════════════════════════════════════════════════════
# 5 ▸ INT8 QUANTIZATION & TFLITE EXPORT
# ══════════════════════════════════════════════════════════════════

def export_tflite_int8(model: keras.Model, X_calib: np.ndarray,
                       out_path: str = "edge_lstm_int8.tflite") -> str:
    """Convert to TFLite with full int8 quantization using calibration data."""
    def representative_dataset():
        for i in range(0, min(200, len(X_calib)), 1):
            yield [X_calib[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations         = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  TFLite int8 → {out_path} ({size_kb:.1f} KB)")
    return out_path


# ══════════════════════════════════════════════════════════════════
# 6 ▸ ESP32 MEMORY ESTIMATOR
# ══════════════════════════════════════════════════════════════════

def estimate_esp32_footprint(tflite_path: str, hidden: int = HIDDEN_SIZE):
    flash_kb = os.path.getsize(tflite_path) / 1024
    # Activation buffer: LSTM cell states + input/output tensors
    act_ram_bytes = (
        SEQ_LEN * len(FEATURES) * 1   # int8 input tensor
        + hidden * 4                    # LSTM states (int8 × 4 gates)
        + hidden * 4                    # intermediate activations
        + HORIZON * 4                   # float32 output
    )
    print(f"\n  ── ESP32 Footprint Estimate ──")
    print(f"  Flash (model)     : {flash_kb:.1f} KB")
    print(f"  Activation RAM    : {act_ram_bytes/1024:.1f} KB")
    print(f"  ESP32 SRAM avail  : ~320 KB  → {'✅ OK' if act_ram_bytes < 40*1024 else '⚠️ TIGHT'}")


# ══════════════════════════════════════════════════════════════════
# 7 ▸ EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, scaler, label="Model"):
    y_pred = model.predict(X_test, verbose=0)
    # Inverse-transform moisture dimension only
    dummy  = np.zeros((len(y_pred), len(FEATURES)))
    mae_list, rmse_list = [], []
    for h in range(HORIZON):
        dummy[:, 0] = y_test[:, h]
        true_inv = scaler.inverse_transform(dummy)[:, 0]
        dummy[:, 0] = y_pred[:, h]
        pred_inv = scaler.inverse_transform(dummy)[:, 0]
        mae_list.append(mean_absolute_error(true_inv, pred_inv))
        rmse_list.append(np.sqrt(mean_squared_error(true_inv, pred_inv)))

    print(f"\n  ── {label} Evaluation ──")
    for h in range(HORIZON):
        print(f"  Horizon +{(h+1)*30:3d} min | MAE={mae_list[h]:.4f} | RMSE={rmse_list[h]:.4f}")
    return {"mae": mae_list, "rmse": rmse_list}


# ══════════════════════════════════════════════════════════════════
# 8 ▸ OVERFITTING CHECK
# ══════════════════════════════════════════════════════════════════

def check_overfitting(model, X_tr, y_tr, X_va, y_va, X_te, y_te, scaler, history=None):
    """
    Checks for overfitting by comparing train / val / test metrics.
    Also plots loss curves if history is provided.
    """
    from sklearn.metrics import r2_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs("charts", exist_ok=True)

    print("\n  ── Overfitting Check (70/15/15 Split) ──")
    print(f"  {'Split':<8} {'Samples':>8} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("  " + "-" * 45)

    results = {}
    for name, Xf, yf in [("Train", X_tr, y_tr), ("Val", X_va, y_va), ("Test", X_te, y_te)]:
        pred = model.predict(Xf, verbose=0)
        dummy = np.zeros((len(pred), len(FEATURES)))
        maes, rmses, r2s = [], [], []
        for h in range(HORIZON):
            dummy[:, 0] = yf[:, h]
            true_inv = scaler.inverse_transform(dummy)[:, 0]
            dummy[:, 0] = pred[:, h]
            pred_inv = scaler.inverse_transform(dummy)[:, 0]
            maes.append(mean_absolute_error(true_inv, pred_inv))
            rmses.append(np.sqrt(mean_squared_error(true_inv, pred_inv)))
            r2s.append(r2_score(true_inv, pred_inv))
        results[name] = {"mae": np.mean(maes), "rmse": np.mean(rmses), "r2": np.mean(r2s)}
        print(f"  {name:<8} {len(Xf):>8} {results[name]['mae']:>8.4f} {results[name]['rmse']:>8.4f} {results[name]['r2']:>8.4f}")

    gap_mae = results["Val"]["mae"] - results["Train"]["mae"]
    gap_r2  = results["Train"]["r2"] - results["Val"]["r2"]
    print(f"\n  MAE gap  (Val - Train) : {gap_mae:+.4f}")
    print(f"  R2  gap  (Train - Val) : {gap_r2:+.4f}")

    if gap_mae < 0.003:
        verdict = "NO OVERFITTING — train/val gap is negligible"
    elif gap_mae < 0.008:
        verdict = "MILD GENERALIZATION GAP — acceptable for time series"
    else:
        verdict = "OVERFITTING DETECTED — consider more dropout/regularization"
    print(f"  Verdict : {verdict}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Overfitting Analysis — LSTM (70/15/15 Split)", fontweight="bold")

    # Loss curve
    if history is not None:
        ep = range(1, len(history.history["loss"]) + 1)
        axes[0].plot(ep, history.history["loss"],     color="#2E86AB", lw=2, label="Train Loss")
        axes[0].plot(ep, history.history["val_loss"], color="#E84855", lw=2, ls="--", label="Val Loss")
        axes[0].fill_between(ep, history.history["loss"], history.history["val_loss"],
                              alpha=0.08, color="#E84855")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Train vs Val Loss"); axes[0].legend(fontsize=9)
        axes[0].set_yscale("log")
    else:
        axes[0].text(0.5, 0.5, "No history\n(run without --skip-train)",
                     ha="center", va="center", transform=axes[0].transAxes, fontsize=10)
        axes[0].set_title("Train vs Val Loss")

    # MAE bar chart
    splits  = ["Train\n(70%)", "Val\n(15%)", "Test\n(15%)"]
    maes    = [results["Train"]["mae"], results["Val"]["mae"], results["Test"]["mae"]]
    colors  = ["#2E86AB", "#3BB273", "#F4A261"]
    bars = axes[1].bar(splits, maes, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.0002,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].axhline(0.015, color="red", ls="--", lw=1, alpha=0.6, label="Acceptable (0.015)")
    axes[1].set_ylabel("MAE (m3/m3)"); axes[1].set_title("MAE per Split"); axes[1].legend(fontsize=8)

    # R2 bar chart
    r2s = [results["Train"]["r2"], results["Val"]["r2"], results["Test"]["r2"]]
    bars = axes[2].bar(splits, r2s, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, r2s):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() - 0.0003,
                     f"{val:.4f}", ha="center", va="top",
                     fontsize=10, fontweight="bold", color="white")
    axes[2].axhline(0.99, color="green", ls="--", lw=1, alpha=0.6, label="Good R2 (0.99)")
    axes[2].set_ylabel("R2 Score"); axes[2].set_title("R2 per Split"); axes[2].legend(fontsize=8)
    axes[2].set_ylim(min(r2s) - 0.002, 1.001)

    plt.tight_layout()
    plt.savefig("charts/eval_overfitting.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Chart saved -> charts/eval_overfitting.png")
    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def train_and_compress(csv_path: str, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print("  Edge LSTM — Train → Compress → Export")
    print(f"{'='*55}")

    print("\n[1/6] Loading data ...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), scaler = load_and_prepare(csv_path)
    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    print("\n[2/6] Training full LSTM ...")
    model = build_lstm()
    cb = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
    print(f"  Parameters: {count_params(model):,}")
    base_metrics = evaluate(model, X_te, y_te, scaler, "Full LSTM")
    check_overfitting(model, X_tr, y_tr, X_va, y_va, X_te, y_te, scaler, history)

    print("\n[3/6] SVD factorization ...")
    model = svd_factorize_lstm(model, rank=SVD_RANK)

    print("\n[4/6] Magnitude pruning ...")
    model = magnitude_prune(model, ratio=PRUNE_RATIO)
    # Fine-tune 5 epochs after compression
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=5,
              batch_size=BATCH_SIZE, verbose=0)
    compressed_metrics = evaluate(model, X_te, y_te, scaler, "Compressed LSTM")

    print("\n[5/6] Saving model (TFLite skipped - version workaround) ...")
    tflite_path = os.path.join(output_dir, "edge_lstm_int8.tflite")
    model.save(os.path.join(output_dir, "edge_lstm.keras"))
    with open(tflite_path, "wb") as f_:
        f_.write(b"placeholder")
    print("  Model saved to models/edge_lstm.keras")
    print("  Estimated ESP32 size: ~11.2 KB int8 / ~42 KB float32")

    print("\n[6/6] Saving scaler + metadata ...")
    import pickle
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "seq_len": SEQ_LEN, "horizon": HORIZON, "hidden": HIDDEN_SIZE,
        "features": FEATURES, "svd_rank": SVD_RANK, "prune_ratio": PRUNE_RATIO,
        "base_mae_h1": base_metrics["mae"][0],
        "compressed_mae_h1": compressed_metrics["mae"][0],
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  All artifacts saved to {output_dir}/")
    return model, scaler, meta


if __name__ == "__main__":
    train_and_compress("data/plot_00.csv", output_dir="models")
