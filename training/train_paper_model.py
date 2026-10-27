"""Train the paper's multi-task LSTM (Fig. 1) and its ablations.

Variants:
    multitask     weather head (MSE) + EPL head (Gaussian NLL)   [main model]
    no_itu        multitask without the six ITU-R features        [Table 10]
    epl_only      EPL head only                                  [Table 9]
    weather_only  weather head only                              [Table 9]
    multitask_msl     multitask on the MSL-pressure dataset (Table 2 reading)
    minute_multitask  multitask on the 1-minute interpolated dataset (Fig. 2 reading)
    multitask_fig1heads  multitask with a hidden Dense(256, ReLU) layer per head (Fig. 1)

Usage:
    python training/train_paper_model.py multitask        # seed 42 (config.SEED)
    python training/train_paper_model.py multitask 1      # repeat with seed 1

Runs with a seed other than config.SEED go to models/paper/seeds/<variant>_seed<N>/,
so the original runs are never overwritten.

Outputs (models/paper/<variant>/): model.keras (best weights), best.keras,
training_log.csv (all epochs), history.json, run_config.json,
test_predictions.npz (and val_predictions.npz).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Reshape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

VARIANTS = ("multitask", "no_itu", "epl_only", "weather_only", "multitask_msl", "minute_multitask",
            "multitask_fig1heads")


def dataset_dir(variant):
    """Every variant uses the main dataset except the two separate experiments."""
    return {"multitask_msl": config.DATASET_DIR_MSL,
            "minute_multitask": config.DATASET_DIR_MINUTE}.get(variant, config.DATASET_DIR)


H = config.HORIZON


def gaussian_nll(y_true, y_pred):
    """Eq. 10: mean of 0.5 * (log sigma^2 + (y - mu)^2 / sigma^2)."""
    mean = y_pred[:, :H]
    log_var = tf.clip_by_value(y_pred[:, H:], *config.LOG_VAR_CLIP)
    return tf.reduce_mean(0.5 * (log_var + tf.square(y_true - mean) / tf.exp(log_var)))


def build_model(n_features, heads, head_hidden=0):
    """head_hidden > 0 adds a Dense(head_hidden, ReLU) layer in each head, as
    drawn in the paper's Fig. 1 ("Dense: 256 -> 256 -> 25 / 10")."""
    inputs = Input(shape=(config.SEQ_LEN, n_features), name="history")
    x = inputs
    for layer in range(config.LSTM_LAYERS):
        last = layer == config.LSTM_LAYERS - 1
        x = LSTM(config.LSTM_UNITS, return_sequences=not last)(x)
        if not last:   # dropout between LSTM layers, as in models/multitask_lstm.py
            x = Dropout(config.DROPOUT)(x)

    def head_input():
        return Dense(head_hidden, activation="relu")(x) if head_hidden else x

    outputs = {}
    if "weather" in heads:
        w = Dense(H * len(config.WEATHER_TARGETS))(head_input())
        outputs["weather"] = Reshape((H, len(config.WEATHER_TARGETS)), name="weather")(w)
    if "epl" in heads:
        outputs["epl"] = Dense(2 * H, name="epl")(head_input())   # 5 means + 5 log-variances
    return Model(inputs, outputs)


def load_split(split, feature_idx, d=config.DATASET_DIR):
    X = np.load(d / f"X_{split}.npy")[:, :, feature_idx]
    targets = {"weather": np.load(d / f"weather_{split}.npy"),
               "epl": np.load(d / f"epl_{split}.npy")}
    return X, targets


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "multitask"
    if variant not in VARIANTS:
        raise SystemExit(f"Variant must be one of {VARIANTS}")
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else config.SEED

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    features = [f for f in config.FEATURES
                if not (variant == "no_itu" and f in config.ITU_FEATURES)]
    feature_idx = [config.FEATURES.index(f) for f in features]
    heads = {"multitask": ["weather", "epl"], "no_itu": ["weather", "epl"],
             "epl_only": ["epl"], "weather_only": ["weather"],
             "multitask_msl": ["weather", "epl"], "minute_multitask": ["weather", "epl"],
             "multitask_fig1heads": ["weather", "epl"]}[variant]

    X_train, y_train = load_split("train", feature_idx, dataset_dir(variant))
    X_val, y_val = load_split("val", feature_idx, dataset_dir(variant))
    X_test, _ = load_split("test", feature_idx, dataset_dir(variant))
    y_train = {k: y_train[k] for k in heads}
    y_val = {k: y_val[k] for k in heads}
    print(f"Variant {variant}: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")

    model = build_model(len(features), heads,
                        head_hidden=config.LSTM_UNITS if variant == "multitask_fig1heads" else 0)
    losses = {"weather": "mse", "epl": gaussian_nll}
    weights = {"weather": config.LOSS_WEIGHT_WEATHER, "epl": config.LOSS_WEIGHT_EPL}
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss={k: losses[k] for k in heads},
                  loss_weights={k: weights[k] for k in heads})
    model.summary()

    out_dir = ROOT / "models" / "paper" / variant
    if seed != config.SEED:
        out_dir = ROOT / "models" / "paper" / "seeds" / f"{variant}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # BackupAndRestore saves the state after every epoch; re-running the same
    # command after an interruption resumes from the last finished epoch.
    # (The early-stopping patience counter restarts at 0 after a resume.)
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(str(out_dir / "backup")),
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best.keras"), monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv"), append=True),
    ]

    start = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE,
                        callbacks=callbacks, shuffle=True, verbose=2)
    train_hours = (time.time() - start) / 3600

    model.save(out_dir / "model.keras")
    with open(out_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    for split, X in (("val", X_val), ("test", X_test)):
        pred = model.predict(X, batch_size=512, verbose=0)
        arrays = {}
        if "epl" in pred:
            log_var = np.clip(pred["epl"][:, H:], *config.LOG_VAR_CLIP)
            arrays["epl_mean"] = pred["epl"][:, :H]
            arrays["epl_std"] = np.exp(0.5 * log_var)
        if "weather" in pred:
            arrays["weather_scaled"] = pred["weather"]
        np.savez(out_dir / f"{split}_predictions.npz", **arrays)

    log = pd.read_csv(out_dir / "training_log.csv")   # full log, also across resumes
    best_epoch = int(log["epoch"].iloc[log["val_loss"].idxmin()]) + 1
    run_config = {
        "variant": variant,
        "seed": seed,
        "dataset": str(dataset_dir(variant).relative_to(ROOT)),
        "features": features,
        "heads": heads,
        "epochs_run": int(len(log)),
        "best_epoch": best_epoch,
        "best_val_loss": float(log["val_loss"].min()),
        "train_hours": round(train_hours, 3),
        "params": int(model.count_params()),
        "tensorflow": tf.__version__,
        "settings": {k: getattr(config, k) for k in (
            "SEQ_LEN", "HORIZON", "LSTM_UNITS", "LSTM_LAYERS", "DROPOUT", "LEARNING_RATE",
            "BATCH_SIZE", "MAX_EPOCHS", "PATIENCE", "LOSS_WEIGHT_WEATHER",
            "LOSS_WEIGHT_EPL", "LOG_VAR_CLIP")} | {"SEED": seed},
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)
    print(json.dumps(run_config, indent=2))


if __name__ == "__main__":
    main()
