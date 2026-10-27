# Result tables

All values in dB unless stated. 'Paper' columns are copied from El-Sayed et al. (2026) for reference; every other number comes from our runs.

## Table 7: test RMSE per city and forecast hour (multi-task LSTM)

| City | Hour 1 | Hour 2 | Hour 3 | Hour 4 | Hour 5 | Average | MAE | Paper average |
|---|---|---|---|---|---|---|---|---|
| Athens | 0.315 | 0.519 | 0.620 | 0.793 | 0.729 | 0.619 | 0.161 | 0.017 |
| Bergen | 0.525 | 0.708 | 0.861 | 0.959 | 1.014 | 0.833 | 0.447 | 0.148 |
| Budapest | 0.986 | 1.200 | 1.119 | 1.001 | 1.069 | 1.078 | 0.349 | 0.005 |
| Chilbolton | 0.524 | 0.712 | 0.779 | 0.914 | 0.977 | 0.797 | 0.369 | 0.005 |
| Vigo | 1.021 | 1.249 | 1.398 | 1.371 | 1.412 | 1.298 | 0.442 | 0.003 |
| Warsaw | 0.545 | 0.800 | 0.939 | 0.905 | 0.925 | 0.836 | 0.337 | 0.039 |
| Overall | 0.702 | 0.905 | 0.985 | 1.007 | 1.041 | 0.936 | 0.351 | 0.026 |

### Table 7 layout, persistence baseline

| City | Hour 1 | Hour 2 | Hour 3 | Hour 4 | Hour 5 | Average | MAE |
|---|---|---|---|---|---|---|---|
| Athens | 0.379 | 0.576 | 0.675 | 0.860 | 0.790 | 0.678 | 0.151 |
| Bergen | 0.474 | 0.790 | 1.018 | 1.158 | 1.255 | 0.980 | 0.477 |
| Budapest | 0.785 | 1.324 | 1.437 | 1.410 | 1.461 | 1.308 | 0.372 |
| Chilbolton | 0.588 | 0.831 | 0.959 | 1.148 | 1.244 | 0.982 | 0.376 |
| Vigo | 0.661 | 1.090 | 1.350 | 1.466 | 1.539 | 1.262 | 0.421 |
| Warsaw | 0.603 | 0.888 | 0.960 | 0.965 | 1.068 | 0.911 | 0.316 |
| Overall | 0.596 | 0.946 | 1.097 | 1.188 | 1.251 | 1.042 | 0.352 |

### Table 7 layout, climatology baseline

| City | Hour 1 | Hour 2 | Hour 3 | Hour 4 | Hour 5 | Average | MAE |
|---|---|---|---|---|---|---|---|
| Athens | 0.564 | 0.663 | 0.740 | 0.872 | 0.800 | 0.735 | 0.264 |
| Bergen | 1.299 | 1.280 | 1.299 | 1.306 | 1.288 | 1.294 | 0.842 |
| Budapest | 1.373 | 1.434 | 1.281 | 1.127 | 1.186 | 1.285 | 0.510 |
| Chilbolton | 0.955 | 1.005 | 0.961 | 1.057 | 1.105 | 1.018 | 0.513 |
| Vigo | 1.703 | 1.793 | 1.836 | 1.787 | 1.741 | 1.773 | 0.889 |
| Warsaw | 0.923 | 1.056 | 1.149 | 1.078 | 1.027 | 1.049 | 0.524 |
| Overall | 1.194 | 1.256 | 1.258 | 1.239 | 1.225 | 1.235 | 0.590 |

### Table 7 layout, Fig. 2 reading (1-min interpolated data, steps = minutes)

| City | Min 1 | Min 2 | Min 3 | Min 4 | Min 5 | Average | MAE | Paper average |
|---|---|---|---|---|---|---|---|---|
| Athens | 0.0271 | 0.0449 | 0.0347 | 0.0377 | 0.0346 | 0.0363 | 0.0069 | 0.0170 |
| Bergen | 0.2338 | 0.3230 | 0.1824 | 0.1945 | 0.2207 | 0.2361 | 0.0252 | 0.1480 |
| Budapest | 0.6201 | 0.7377 | 0.5327 | 0.5517 | 0.5923 | 0.6112 | 0.0537 | 0.0050 |
| Chilbolton | 0.1016 | 0.1842 | 0.0369 | 0.0368 | 0.0922 | 0.1053 | 0.0129 | 0.0050 |
| Vigo | 0.8334 | 0.9991 | 0.7136 | 0.7361 | 0.7983 | 0.8223 | 0.0983 | 0.0030 |
| Warsaw | 0.1865 | 0.2357 | 0.1412 | 0.1410 | 0.1597 | 0.1764 | 0.0140 | 0.0390 |
| Overall | 0.4433 | 0.5381 | 0.3760 | 0.3887 | 0.4226 | 0.4375 | 0.0352 | 0.0260 |

#### persistence, same minute test set

| City | Min 1 | Min 2 | Min 3 | Min 4 | Min 5 | Average | MAE |
|---|---|---|---|---|---|---|---|
| Athens | 0.0063 | 0.0126 | 0.0190 | 0.0253 | 0.0316 | 0.0210 | 0.0040 |
| Bergen | 0.0079 | 0.0158 | 0.0237 | 0.0316 | 0.0395 | 0.0262 | 0.0118 |
| Budapest | 0.0131 | 0.0262 | 0.0392 | 0.0523 | 0.0654 | 0.0434 | 0.0104 |
| Chilbolton | 0.0098 | 0.0196 | 0.0294 | 0.0392 | 0.0490 | 0.0325 | 0.0102 |
| Vigo | 0.0110 | 0.0220 | 0.0330 | 0.0440 | 0.0550 | 0.0365 | 0.0109 |
| Warsaw | 0.0101 | 0.0201 | 0.0302 | 0.0402 | 0.0503 | 0.0334 | 0.0084 |
| Overall | 0.0099 | 0.0199 | 0.0298 | 0.0397 | 0.0496 | 0.0329 | 0.0093 |

#### linear_extrapolation, same minute test set

| City | Min 1 | Min 2 | Min 3 | Min 4 | Min 5 | Average | MAE |
|---|---|---|---|---|---|---|---|
| Athens | 0.0005 | 0.0010 | 0.0015 | 0.0020 | 0.0026 | 0.0017 | 0.0001 |
| Bergen | 0.0024 | 0.0049 | 0.0073 | 0.0097 | 0.0122 | 0.0081 | 0.0010 |
| Budapest | 0.0045 | 0.0090 | 0.0135 | 0.0180 | 0.0225 | 0.0149 | 0.0012 |
| Chilbolton | 0.0038 | 0.0076 | 0.0114 | 0.0152 | 0.0190 | 0.0126 | 0.0010 |
| Vigo | 0.0074 | 0.0148 | 0.0221 | 0.0295 | 0.0369 | 0.0245 | 0.0025 |
| Warsaw | 0.0014 | 0.0028 | 0.0042 | 0.0056 | 0.0070 | 0.0046 | 0.0005 |
| Overall | 0.0040 | 0.0080 | 0.0121 | 0.0161 | 0.0201 | 0.0133 | 0.0010 |

## Table 8: ten unseen cities (multitask)

Means over every hour Apr-Dec 2024 (6,385 forecasts per city). 'Abs. error' = |mean predicted - mean true| (the paper's measure); 'Hourly MAE' = mean |error| over hours and horizons.

| City | True EPL | Pred. (weighted) | Abs. error (weighted) | Abs. error (median) | Hourly MAE (weighted) | Hourly MAE (own inputs) | Hourly MAE (persistence) | Paper true | Paper pred. | Paper abs. error |
|---|---|---|---|---|---|---|---|---|---|---|
| Berlin | 1.152 | 1.191 | 0.039 | 0.066 | 0.535 | 0.325 | 0.376 | 0.963 | 1.049 | 0.087 |
| Oslo | 1.321 | 1.511 | 0.190 | 0.179 | 0.783 | 0.400 | 0.447 | 1.228 | 1.066 | 0.162 |
| Rome | 1.308 | 1.075 | 0.233 | 0.324 | 0.602 | 0.407 | 0.480 | 0.865 | 1.036 | 0.172 |
| Stockholm | 1.072 | 1.398 | 0.326 | 0.067 | 0.634 | 0.269 | 0.254 | 0.887 | 1.048 | 0.162 |
| Bucharest | 1.059 | 1.029 | 0.030 | 0.095 | 0.489 | 0.278 | 0.307 | 0.896 | 1.036 | 0.140 |
| London | 1.237 | 1.327 | 0.090 | 0.027 | 0.409 | 0.332 | 0.358 | 0.988 | 1.150 | 0.162 |
| Davos | 1.072 | 1.172 | 0.099 | 0.007 | 0.652 | 2.356 | 0.360 | 0.787 | 1.080 | 0.293 |
| Paris | 1.336 | 1.293 | 0.043 | 0.185 | 0.589 | 0.398 | 0.452 | 0.869 | 1.142 | 0.272 |
| Marseille | 1.193 | 1.328 | 0.135 | 0.009 | 0.659 | 0.312 | 0.347 | 0.820 | 1.143 | 0.323 |
| Madrid | 0.779 | 1.483 | 0.705 | 0.555 | 0.865 | 1.078 | 0.187 | 0.810 | 1.145 | 0.335 |
| Average | 1.153 | 1.281 | 0.189 | 0.151 | 0.622 | 0.615 | 0.357 | 0.911 | 1.089 | 0.211 |

## Table 8: ten unseen cities (multitask_msl)

Means over every hour Apr-Dec 2024 (6,385 forecasts per city). 'Abs. error' = |mean predicted - mean true| (the paper's measure); 'Hourly MAE' = mean |error| over hours and horizons.

| City | True EPL | Pred. (weighted) | Abs. error (weighted) | Abs. error (median) | Hourly MAE (weighted) | Hourly MAE (own inputs) | Hourly MAE (persistence) | Paper true | Paper pred. | Paper abs. error |
|---|---|---|---|---|---|---|---|---|---|---|
| Berlin | 1.152 | 1.196 | 0.043 | 0.073 | 0.538 | 0.349 | 0.376 | 0.963 | 1.049 | 0.087 |
| Oslo | 1.321 | 1.505 | 0.184 | 0.216 | 0.781 | 0.407 | 0.447 | 1.228 | 1.066 | 0.162 |
| Rome | 1.308 | 1.040 | 0.268 | 0.360 | 0.594 | 0.415 | 0.480 | 0.865 | 1.036 | 0.172 |
| Stockholm | 1.072 | 1.381 | 0.309 | 0.022 | 0.622 | 0.282 | 0.254 | 0.887 | 1.048 | 0.162 |
| Bucharest | 1.059 | 1.038 | 0.021 | 0.098 | 0.499 | 0.290 | 0.307 | 0.896 | 1.036 | 0.140 |
| London | 1.237 | 1.335 | 0.098 | 0.006 | 0.413 | 0.356 | 0.358 | 0.988 | 1.150 | 0.162 |
| Davos | 1.072 | 1.154 | 0.082 | 0.034 | 0.646 | 0.364 | 0.360 | 0.787 | 1.080 | 0.293 |
| Paris | 1.336 | 1.299 | 0.037 | 0.192 | 0.595 | 0.417 | 0.452 | 0.869 | 1.142 | 0.272 |
| Marseille | 1.193 | 1.296 | 0.103 | 0.045 | 0.641 | 0.323 | 0.347 | 0.820 | 1.143 | 0.323 |
| Madrid | 0.779 | 1.418 | 0.640 | 0.390 | 0.808 | 0.193 | 0.187 | 0.810 | 1.145 | 0.335 |
| Average | 1.153 | 1.266 | 0.179 | 0.144 | 0.614 | 0.340 | 0.357 | 0.911 | 1.089 | 0.211 |

## Table 9: single-task vs multi-task

| Configuration | EPL RMSE | EPL MAE | Weather RMSE (standardised) | Training time (h) | Epochs (best) | Paper EPL RMSE | Paper weather RMSE |
|---|---|---|---|---|---|---|---|
| EPL only | 0.942 | 0.358 |  | 1.229 | 19 (13) | 0.015 |  |
| Weather only |  |  | 0.480 | 0.984 | 18 (12) |  | 0.892 |
| Multi-task | 0.936 | 0.351 | 0.522 | 7.599 | 20 (14) | 0.026 | 0.761 |

Training times include any standby pauses of the laptop and are not comparable across runs.

## Table 10: removing the ITU-R-derived features

| Model configuration | Average RMSE | Average MAE | PICP 95 % | Paper RMSE | Paper MAE |
|---|---|---|---|---|---|
| Full feature set (14) | 0.936 | 0.351 | 96.613 | 0.026 | 0.009 |
| Without ITU-R-derived features (8) | 0.933 | 0.352 | 96.537 | 0.028 | 0.012 |

## Table 11: qualitative comparison with prior work

| Metric | Bai et al. MLP [11] | Bai et al. LSTM [12] | Paper (reported) | This replication |
|---|---|---|---|---|
| Trained RMSE (dB) | 0.75 | ~0.65 | 0.026 | 0.936 |
| Unseen-city error (dB) | Not tested | Not tested | 0.213 | 0.189 (error of means); 0.622 (hourly MAE) |
| Prediction horizon | 1 min | 1 min | 5 h | 5 h (hourly steps) |
| Geographic coverage | Single-site | Single-site | 10 cities | 6 training + 10 unseen cities |
| Uncertainty quantification | No | No | 99.03 % | PICP 96.6 % |
| Training cities | 1 | 1 | 6 | 6 |
| Test cities | 1 | 1 | 10 | 6 (held-out period) + 10 unseen |

Bai et al. values are as quoted in the paper (different datasets and targets; not re-run here).

## All trained variants and baselines (test sets)

| Model / baseline | Dataset | RMSE | MAE | PICP 95 % | NLL (Eq. 10) | Sharpness | Weather RMSE (std.) | Best epoch |
|---|---|---|---|---|---|---|---|---|
| Persistence | paper | 1.0420 | 0.3521 |  |  |  |  |  |
| Climatology | paper | 1.2347 | 0.5902 |  |  |  |  |  |
| epl_only | paper | 0.9420 | 0.3581 | 96.8955 | -0.6290 | 0.5898 |  | 13 |
| minute_multitask | paper_minute | 0.4375 | 0.0352 | 98.8672 | -4.2694 | 0.0453 | 0.0951 | 30 |
| multitask | paper | 0.9360 | 0.3506 | 96.6128 | -0.6320 | 0.5570 | 0.5222 | 14 |
| multitask_fig1heads | paper | 0.9097 | 0.3422 | 96.5859 | -0.6632 | 0.5400 | 0.5084 | 16 |
| multitask_msl | paper_msl | 0.9347 | 0.3442 | 96.0902 | -0.5590 | 0.5155 | 0.5197 | 16 |
| no_itu | paper | 0.9326 | 0.3521 | 96.5366 | -0.6352 | 0.5435 | 0.5124 | 18 |
| weather_only | paper |  |  |  |  |  | 0.4801 | 12 |
| persistence | paper_minute | 0.0329 | 0.0093 |  |  |  |  |  |
| linear_extrapolation | paper_minute | 0.0133 | 0.0010 |  |  |  |  |  |

## Repeated seeds (test set, EPL in dB)

Mean and sample standard deviation over the listed seeds; seed 42 is the original run.

| variant | n_runs | seeds | EPL_RMSE_dB_mean | EPL_RMSE_dB_std | EPL_RMSE_dB_min | EPL_RMSE_dB_max | EPL_MAE_dB_mean | EPL_RMSE_hour1_dB_mean | PICP_95_pct_mean |
|---|---|---|---|---|---|---|---|---|---|
| epl_only | 3 | 1 2 42 | 0.9457 | 0.0069 | 0.9415 | 0.9537 | 0.3564 | 0.7152 | 96.6457 |
| multitask | 3 | 1 2 42 | 0.9399 | 0.0037 | 0.9360 | 0.9433 | 0.3491 | 0.7078 | 96.3571 |
| multitask_fig1heads | 3 | 1 2 42 | 0.9161 | 0.0056 | 0.9097 | 0.9199 | 0.3430 | 0.6435 | 96.2823 |
| no_itu | 3 | 1 2 42 | 0.9397 | 0.0063 | 0.9326 | 0.9449 | 0.3486 | 0.7121 | 96.2674 |

### Paired differences (A - B, same seed; negative = A better)

| comparison | metric | seeds | differences | mean_difference | std_difference | same_sign_all_seeds |
|---|---|---|---|---|---|---|
| multitask_fig1heads - multitask | EPL_RMSE_dB | 1 2 42 | -0.0205 -0.0245 -0.0263 | -0.0238 | 0.0030 | True |
| multitask - epl_only | EPL_RMSE_dB | 1 2 42 | -0.0011 -0.0104 -0.0060 | -0.0058 | 0.0046 | True |
| no_itu - multitask | EPL_RMSE_dB | 1 2 42 | +0.0045 -0.0017 -0.0034 | -0.0002 | 0.0042 | False |
| multitask_fig1heads - multitask | EPL_RMSE_hour1_dB | 1 2 42 | -0.0620 -0.0643 -0.0666 | -0.0643 | 0.0023 | True |
