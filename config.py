"""Shared settings for the paper replication pipeline.

El-Sayed et al., "Prediction of channel propagation impairments in LEO
satellite networks using machine learning", EURASIP JWCN 2026:102.

Each value is marked:
  [PAPER]  stated in the paper
  [OURS]   not given in the paper; our documented choice
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------
# open_meteo_v2 holds a fresh download that adds the liquid "rain"
# column; its shared columns were checked against the original files.
WEATHER_DIR = ROOT / "data" / "processed" / "weather" / "open_meteo_v2"
UNSEEN_WEATHER_DIR = ROOT / "data" / "processed" / "weather" / "open_meteo"
ERA5_CDS_DIR = ROOT / "data" / "raw" / "era5"
PHYSICS_DIR = ROOT / "data" / "processed" / "physics_paper"
DATASET_DIR = ROOT / "data" / "processed" / "paper"
# Separate experiments (the main dataset above is never modified by them)
MSL_DIR = ROOT / "data" / "processed" / "weather" / "open_meteo_msl"
DATASET_DIR_MSL = ROOT / "data" / "processed" / "paper_msl"        # MSL pressure input
DATASET_DIR_MINUTE = ROOT / "data" / "processed" / "paper_minute"  # 1-min interpolation (Fig. 2)
RESULTS_DIR = ROOT / "results" / "paper"

YEARS = [2020, 2021, 2022, 2023, 2024]          # [PAPER] five years
UNSEEN_YEARS = [2023, 2024]                     # [OURS] covers the test period

# ---------------------------------------------------------------
# Link geometry and ITU-R settings
# ---------------------------------------------------------------
FREQUENCY_GHZ = 39.0          # [PAPER] Q-band
ELEVATION_DEG = 30.0          # [OURS] paper: "fixed elevation angle"
POLARIZATION_TILT_DEG = 45.0  # [OURS] circular polarisation (P.838 tau)
CLOUD_KL_TEMP_C = 0.0         # [OURS] P.840 K_l evaluated at 0 degC

# LWP proxy coefficients are fitted by physics/calibrate_lwp_proxy.py (needs ERA5
# CDS files in data/raw/era5/). The fitted file is kept with the source code so the
# labels can be rebuilt without CDS access.
LWP_PROXY_FILE = ROOT / "physics" / "lwp_proxy_calibration.json"
LWP_CALIBRATION_END = "2023-01-01"   # [OURS] fit only on 2020-2022

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
SAMPLES_PER_CITY = 10_000     # [PAPER] stratified by month and hour
SEQ_LEN = 60                  # [PAPER] 60 time steps (hourly data -> 60 h)
HORIZON = 5                   # [PAPER] 5 h ahead
TRAIN_FRAC = 0.70             # [PAPER]
VAL_FRAC = 0.15               # [PAPER]
STD_WINDOW_H = 24             # [OURS] trailing window for sigma features
SEED = 42                     # [PAPER]

FEATURES = [                  # [PAPER] Table 2, in this order
    "latitude",
    "longitude",
    "temperature_C",
    "relative_humidity_percent",
    "pressure_hPa",
    "precipitation_mm",
    "cloud_cover_percent",
    "gas_mean",
    "gas_std",
    "cloud_mean",
    "cloud_std",
    "rain_mean",
    "rain_std",
    "elevation_deg",
]
ITU_FEATURES = FEATURES[7:13]   # removed in the Table 10 ablation

WEATHER_TARGETS = [           # [PAPER] Fig. 1 weather head
    "temperature_C",
    "relative_humidity_percent",
    "pressure_hPa",
    "precipitation_mm",
    "cloud_cover_percent",
]
EPL_COLUMN = "epl_dB"

# ---------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------
LSTM_UNITS = 256              # [PAPER]
LSTM_LAYERS = 3               # [PAPER]
DROPOUT = 0.2                 # [PAPER]
LEARNING_RATE = 1e-4          # [PAPER]
BATCH_SIZE = 128              # [PAPER]
MAX_EPOCHS = 50               # [PAPER]
PATIENCE = 6                  # [PAPER]
LOSS_WEIGHT_WEATHER = 1.0     # [PAPER]
LOSS_WEIGHT_EPL = 1.0         # [PAPER]
LOG_VAR_CLIP = (-12.0, 6.0)   # [OURS] numerical safety for the NLL

# ---------------------------------------------------------------
# Coordinate-based inference for unseen cities
# ---------------------------------------------------------------
K_NEIGHBOURS = 3              # [PAPER]
TAU_KM = 500.0                # [OURS] paper does not give tau
