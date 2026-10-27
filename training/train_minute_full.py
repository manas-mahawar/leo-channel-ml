"""Train the full-series 1-minute reconstruction (variant minute_full_fig1heads).

Same model, loss and optimiser as the hourly `multitask_fig1heads`
(training/train_paper_model.build_model with head_hidden=256 and gaussian_nll;
the hourly script and its results are not touched). Data come lazily from
data_pipeline/minute_full_loader.py. Every output goes to two new folders:

    results/minute_full_<name>/   run_config.json, scalers, training_log.csv,
                                  metrics (model + baselines), sample predictions
    models/minute_full_<name>/    best.keras, model.keras

The script refuses to run if results/minute_full_<name>/run_config.json exists.

Smoke test (1 % of anchors, one epoch, stops after --time-limit-min minutes):
    python training/train_minute_full.py --name smoke --train-fraction 0.01 --val-fraction 0.01 --test-fraction 0.01 --epochs 1
Full run (not started; needs approval):
    python training/train_minute_full.py --name full --epochs 50 --time-limit-min 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))

import config
from data_pipeline.minute_full_loader import MinuteBatches, MinuteFullData
from evaluate_minute_full import evaluate
from train_paper_model import build_model, gaussian_nll

VARIANT = "minute_full_fig1heads"


class TimeAndThroughput(tf.keras.callbacks.Callback):
    """Records per-batch timing and stops training once the time limit is reached."""

    def __init__(self, limit_s):
        super().__init__()
        self.limit_s, self.batch_times, self.stopped_early = limit_s, [], False

    def on_train_begin(self, logs=None):
        self.start = self.last = time.time()

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        self.batch_times.append(now - self.last)
        self.last = now
        if logs and not np.isfinite(logs.get("loss", 0.0)):
            raise RuntimeError(f"Non-finite training loss at batch {batch}: {logs}")
        if self.limit_s and now - self.start > self.limit_s:
            self.stopped_early = True
            self.model.stop_training = True

    def on_test_begin(self, logs=None):
        self.val_start = time.time()

    def on_test_end(self, logs=None):
        self.val_seconds = time.time() - self.val_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=1.0)
    p.add_argument("--test-fraction", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--time-limit-min", type=float, default=5.0, help="0 = no limit")
    args = p.parse_args()

    res_dir = ROOT / "results" / f"minute_full_{args.name}"
    model_dir = ROOT / "models" / f"minute_full_{args.name}"
    if (res_dir / "run_config.json").exists():
        raise SystemExit(f"{res_dir} already holds a finished run; choose another --name.")
    res_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    t = time.time()
    data = MinuteFullData()
    timings["load_sources_s"] = time.time() - t
    t = time.time()
    scaler_rows = data.fit_scalers()
    data.save_scalers(res_dir)
    timings["fit_scalers_s"] = time.time() - t

    train = MinuteBatches(data, "train", args.batch_size, shuffle=True, fraction=args.train_fraction, seed=args.seed)
    val = MinuteBatches(data, "val", args.batch_size, fraction=args.val_fraction, seed=args.seed)
    test = MinuteBatches(data, "test", 512, fraction=args.test_fraction, seed=args.seed)
    counts = {s: {"windows": len(b.anchors), "batches": len(b),
                  "per_city": np.bincount(b.city_idx, minlength=len(data.cities)).tolist()}
              for s, b in (("train", train), ("val", val), ("test", test))}
    print("windows:", {s: c["windows"] for s, c in counts.items()}, "batch", args.batch_size, flush=True)

    model = build_model(len(config.FEATURES), ["weather", "epl"], head_hidden=config.LSTM_UNITS)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss={"weather": "mse", "epl": gaussian_nll},
                  loss_weights={"weather": config.LOSS_WEIGHT_WEATHER, "epl": config.LOSS_WEIGHT_EPL})
    clock = TimeAndThroughput(args.time_limit_min * 60)
    callbacks = [clock,
                 tf.keras.callbacks.ModelCheckpoint(str(model_dir / "best.keras"), monitor="val_loss", save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(str(res_dir / "training_log.csv"))]
    t = time.time()
    history = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks, verbose=2)
    timings["fit_s"] = time.time() - t
    timings["validation_s_last_epoch"] = getattr(clock, "val_seconds", None)
    steady = clock.batch_times[5:] if len(clock.batch_times) > 10 else clock.batch_times
    timings["train_batch_s_median"] = float(np.median(steady))
    timings["train_batches_run"] = len(clock.batch_times)

    # Save, reload and check that inference is identical after loading
    t = time.time()
    model.save(model_dir / "model.keras")
    reloaded = tf.keras.models.load_model(model_dir / "model.keras", custom_objects={"gaussian_nll": gaussian_nll},
                                          compile=False)
    X0, _ = test[0]
    a, b = model.predict_on_batch(X0), reloaded.predict_on_batch(X0)
    reload_diff = float(max(np.abs(a[k] - b[k]).max() for k in a))
    finite = bool(all(np.isfinite(a[k]).all() for k in a))
    timings["save_reload_s"] = time.time() - t

    t = time.time()
    test_result, preds = evaluate(test, reloaded, keep_predictions=2000)
    timings["test_eval_s"] = time.time() - t
    np.savez(res_dir / "test_predictions_sample.npz", **preds)
    with open(res_dir / "metrics_test.json", "w", encoding="utf-8") as file:
        json.dump(test_result, file, indent=2)

    # Baselines on every test anchor (cheap: no model), same definitions
    t = time.time()
    full_test = MinuteBatches(data, "test", 4096, seed=args.seed)
    baselines_full, _ = evaluate(full_test, None)
    timings["baselines_full_test_s"] = time.time() - t
    with open(res_dir / "baselines_full_test.json", "w", encoding="utf-8") as file:
        json.dump(baselines_full, file, indent=2)

    run_config = {
        "variant": VARIANT, "name": args.name, "seed": args.seed, "model_dir": str(model_dir.relative_to(ROOT)),
        "protocol": "full 2020-2024 hourly series per city, linearly interpolated to 1 minute; 60-minute inputs; "
                    "targets +1..+5 minutes; anchors every 5 minutes; chronological 70/15/15 per city; no hourly "
                    "row shared between splits",
        "fractions": {"train": args.train_fraction, "val": args.val_fraction, "test": args.test_fraction},
        "counts": counts, "scaler_rows": scaler_rows, "features": config.FEATURES,
        "weather_targets": config.WEATHER_TARGETS, "epochs_requested": args.epochs,
        "epochs_run": len(history.history["loss"]), "stopped_by_time_limit": clock.stopped_early,
        "batch_size": args.batch_size, "params": int(model.count_params()), "tensorflow": tf.__version__,
        "history": history.history, "timings": timings, "reload_max_abs_diff": reload_diff,
        "predictions_finite": finite,
        "settings": {k: getattr(config, k) for k in ("SEQ_LEN", "HORIZON", "LSTM_UNITS", "LSTM_LAYERS", "DROPOUT",
                                                     "LEARNING_RATE", "PATIENCE", "LOSS_WEIGHT_WEATHER",
                                                     "LOSS_WEIGHT_EPL", "LOG_VAR_CLIP")},
    }
    with open(res_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, default=float)
    print(json.dumps({"timings": timings, "test_metrics": {k: v["RMSE"] for k, v in test_result["metrics"].items()},
                      "baselines_full_test": {k: v["RMSE"] for k, v in baselines_full["metrics"].items()},
                      "reload_max_abs_diff": reload_diff}, indent=2, default=float))


if __name__ == "__main__":
    main()
