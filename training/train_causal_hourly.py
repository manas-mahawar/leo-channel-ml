"""Train multitask_fig1heads on the strictly causal protocol causal_hourly_v1 (one seed).

Data: data_pipeline/causal_forecast_source.CausalHourlySource for the six training
cities (all hourly origins; inputs t-59..t; EPL and weather targets t+1..t+5 h).
Model, loss and optimiser: unchanged from training/train_paper_model.py
(build_model with head_hidden=256, MSE weather + Gaussian NLL EPL, Adam 1e-4,
batch 128, <= 50 epochs, early stopping on validation loss with patience 6).
Scalers: features on training input rows only; weather targets on training
target rows only; EPL in dB. The test split is only predicted after training.

Outputs (refuses to overwrite a finished run; resumes an interrupted one):
    results/<name>/  run_config.json, training_log.csv, scalers, test_predictions.npz
    models/<name>/   best.keras, model.keras, backup/ (resume state, removed by Keras at the end)

Usage:
    python training/train_causal_hourly.py --dry-run     # data, shapes, key check, estimates; no training
    python training/train_causal_hourly.py               # train seed 42
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import config
from data_pipeline.causal_forecast_source import CausalHourlySource, fit_feature_scaler
from data_pipeline.cities import CITIES
from sklearn.preprocessing import StandardScaler

H, SEQ = config.HORIZON, config.SEQ_LEN
W_IDX = [config.FEATURES.index(c) for c in config.WEATHER_TARGETS]
BASELINE_KEYS = ROOT / "results" / "causal_baseline_audit" / "aligned_keys_hourly.npz"


def peak_memory_gb():
    """Peak working set of this process (Windows), without extra dependencies."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        c = Counters()
        c.cb = ctypes.sizeof(c)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not get_info(get_process(), ctypes.byref(c), c.cb):
            return None
        return round(c.PeakWorkingSetSize / 1e9, 2)
    except (AttributeError, OSError):
        return None


def load_sources():
    sources = []
    for city, c in CITIES.items():
        table = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
        sources.append(CausalHourlySource(table, city, c["latitude"], c["longitude"]))
    return sources


def build_split(sources, split, x_scaler, w_scaler):
    X, W, Y, keys = [], [], [], []
    for c, src in enumerate(sources):
        o = src.splits[split]
        w = src.windows(o)
        x = x_scaler.transform(w["X"].reshape(-1, w["X"].shape[-1])).reshape(w["X"].shape).astype(np.float32)
        wt = src.feats[o[:, None] + np.arange(1, H + 1)][:, :, W_IDX]          # weather at t+1..t+5
        W.append(w_scaler.transform(wt.reshape(-1, len(W_IDX))).reshape(wt.shape).astype(np.float32))
        X.append(x); Y.append(w["y"].astype(np.float32))
        keys.append(np.stack([np.full(len(o), c), w["origin_time"].astype("datetime64[h]").astype(np.int64)], axis=1))
    return np.concatenate(X), np.concatenate(W), np.concatenate(Y), np.concatenate(keys)


def fit_weather_scaler(sources):
    """Training target rows only: first train origin + 1 .. last train origin + H."""
    scaler = StandardScaler()
    for src in sources:
        t = src.splits["train"]
        scaler.partial_fit(src.feats[t[0] + 1:t[-1] + H + 1][:, W_IDX])
    return scaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="causal_hourly_fig1heads_seed42")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    res_dir, model_dir = ROOT / "results" / args.name, ROOT / "models" / args.name
    if (res_dir / "run_config.json").exists():
        raise SystemExit(f"{res_dir} holds a finished run; not overwriting.")
    if not args.dry_run and (res_dir.exists() or model_dir.exists()) and not (model_dir / "backup").exists():
        raise SystemExit(f"{res_dir} or {model_dir} exists without resume state; not overwriting.")

    t0 = time.time()
    sources = load_sources()
    x_scaler, w_scaler = fit_feature_scaler(sources), fit_weather_scaler(sources)
    data = {s: build_split(sources, s, x_scaler, w_scaler) for s in ("train", "val", "test")}
    prep_s = time.time() - t0

    # The test keys must be exactly the baseline evaluation's keys
    base = np.load(BASELINE_KEYS)
    test_keys = data["test"][3]
    same_keys = bool(np.array_equal(test_keys[:, 0], base["city_index"]) and
                     np.array_equal(test_keys[:, 1], base["origin_hour_since_epoch"]) and
                     np.allclose(data["test"][2], base["y"], atol=1e-5))
    finite = all(np.isfinite(a).all() for s in data.values() for a in s[:3])
    n = {s: len(d[0]) for s, d in data.items()}
    gb = sum(a.nbytes for d in data.values() for a in d[:3]) / 1e9
    # measured: 3.65 min per epoch for 42,000 + 8,898 windows on mains power (hourly fig1heads)
    min_per_epoch = 3.65 * (n["train"] + n["val"]) / (42_000 + 8_898)
    info = {"windows": n, "shapes": {s: {"X": list(d[0].shape), "weather": list(d[1].shape), "epl": list(d[2].shape)}
                                     for s, d in data.items()},
            "array_memory_gb": round(gb, 2), "peak_memory_gb_after_build": peak_memory_gb(),
            "prep_seconds": round(prep_s, 1), "test_keys_equal_baseline_keys": same_keys, "all_finite": finite,
            "estimated_min_per_epoch_mains": round(min_per_epoch, 1),
            "estimated_min_per_epoch_battery": round(2 * min_per_epoch, 1),
            "output_paths": {"results": str(res_dir.relative_to(ROOT)), "models": str(model_dir.relative_to(ROOT))}}
    print(json.dumps(info, indent=2), flush=True)
    if not (same_keys and finite):
        raise SystemExit("Test keys differ from the baseline keys or data are not finite; stopping.")
    if args.dry_run:
        return

    import tensorflow as tf
    from train_paper_model import build_model, gaussian_nll

    res_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, res_dir / "feature_scaler.pkl")
    joblib.dump(w_scaler, res_dir / "weather_scaler.pkl")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    model = build_model(len(config.FEATURES), ["weather", "epl"], head_hidden=config.LSTM_UNITS)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss={"weather": "mse", "epl": gaussian_nll},
                  loss_weights={"weather": config.LOSS_WEIGHT_WEATHER, "epl": config.LOSS_WEIGHT_EPL})
    callbacks = [tf.keras.callbacks.BackupAndRestore(str(model_dir / "backup")),
                 tf.keras.callbacks.ModelCheckpoint(str(model_dir / "best.keras"), monitor="val_loss", save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(str(res_dir / "training_log.csv"), append=True)]
    Xtr, Wtr, Ytr, _ = data["train"]
    Xva, Wva, Yva, _ = data["val"]
    t = time.time()
    history = model.fit(Xtr, {"weather": Wtr, "epl": Ytr}, validation_data=(Xva, {"weather": Wva, "epl": Yva}),
                        epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE, shuffle=True,
                        callbacks=callbacks, verbose=2)
    train_hours = (time.time() - t) / 3600
    model.save(model_dir / "model.keras")

    Xte, Wte, Yte, keys = data["test"]
    pred = model.predict(Xte, batch_size=512, verbose=0)
    log_var = np.clip(pred["epl"][:, H:], *config.LOG_VAR_CLIP)
    np.savez(res_dir / "test_predictions.npz", city_index=keys[:, 0], origin_hour_since_epoch=keys[:, 1],
             y=Yte, epl_mean=pred["epl"][:, :H], epl_std=np.exp(0.5 * log_var), weather_scaled=pred["weather"],
             weather_true_scaled=Wte)
    log = pd.read_csv(res_dir / "training_log.csv")
    run_config = {**info, "protocol": "causal_hourly_v1", "variant": "multitask_fig1heads", "name": args.name,
                  "seed": args.seed, "params": int(model.count_params()), "tensorflow": tf.__version__,
                  "epochs_run": int(len(log)), "best_epoch": int(log.epoch[log.val_loss.idxmin()]) + 1,
                  "best_val_loss": float(log.val_loss.min()), "train_hours": round(train_hours, 3),
                  "peak_memory_gb": peak_memory_gb(),
                  "model_selection": "validation loss only (early stopping, best checkpoint); test predicted once after training",
                  "scaling": "features: training input rows; weather targets: training target rows; EPL in dB",
                  "settings": {k: getattr(config, k) for k in ("SEQ_LEN", "HORIZON", "LSTM_UNITS", "LSTM_LAYERS",
                               "DROPOUT", "LEARNING_RATE", "BATCH_SIZE", "MAX_EPOCHS", "PATIENCE",
                               "LOSS_WEIGHT_WEATHER", "LOSS_WEIGHT_EPL", "LOG_VAR_CLIP")},
                  "history": history.history}
    with open(res_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, default=float)
    print(json.dumps({k: run_config[k] for k in ("epochs_run", "best_epoch", "train_hours", "peak_memory_gb")}, indent=2))


if __name__ == "__main__":
    main()
