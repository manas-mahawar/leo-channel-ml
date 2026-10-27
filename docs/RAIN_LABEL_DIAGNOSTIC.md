# Rain label diagnostic

Date: 2026-10-27. Question: is the large EPL label discrepancy with the paper
(our test-period maximum about 25.6 dB vs about 6.7 dB visible in the paper's
Figs. 3-4; our unseen-city mean 1.15 dB vs 0.91 dB in its Table 8) caused by
an implementation error, or by choices the paper leaves unspecified?

**Answer: no implementation error was found.** The rain, gas and cloud
calculations reproduce independent references to numerical precision. The
discrepancy comes from how ITU-R P.618 is applied to hourly rain and from
unspecified choices (elevation, time percentage, rain-rate definition),
which the paper does not report.

Scripts (read-only; no dataset regenerated, no model trained):
`evaluation/rain_label_diagnostic.py` (outputs
`results/replication_audit/rain_label_diagnostic.csv` and `.json`),
`data_pipeline/minute_full_source.py`; tests in `tests/test_pipeline.py`
(`TestRainLabelReference`, `TestMinuteFullSource`).

## 1. What `physics/build_attenuation.py` computes (verified line by line)

`rain_attenuation(rain_rate, lat, lon)` follows ITU-R P.618-13 Sect. 2.2.1.1:

| Step | Quantity | Code | Inputs and units | Check |
|---|---|---|---|---|
| 1 | rain height h_R | `itu839.rain_height(lat, lon)` | km above sea level, P.839-4 (h_0 + 0.36 km) | equals P.839 to 4e-16 |
| - | station height h_s | `itu1511.topographic_altitude` | km, P.1511 | 0.075-0.160 km (training cities) |
| 2 | slant path L_s = (h_R - h_s)/sin(theta) | `dh / SIN_EL` | theta = 30 deg (`config.ELEVATION_DEG`) | equals reference |
| 3 | horizontal projection L_G = L_s cos(theta) | yes | km | equals reference |
| 4 | R_0.01 | the **hourly** ERA5 liquid rain `rain_mm` (mm in the preceding hour = mm/h) | **interpretation**: P.618 expects the rate exceeded 0.01 % of an average year | ambiguity (Sect. 3) |
| 5 | gamma_R = k R^alpha | `itu838.rain_specific_attenuation(R, f, el, tau)` | f = 39 GHz, el = 30 deg, tau = 45 deg (circular) | P.838-3 combination of k_H, k_V, alpha_H, alpha_V re-done by hand: k = 0.4136, alpha = 0.8617; gamma equals k R^alpha to 4e-16 |
| 6 | horizontal reduction r_0.01 | `r001` | uses L_G, gamma_R, f in GHz | equals reference |
| 7 | zeta, L_R, chi, nu_0.01 | `zeta`, `l_r`, `chi`, `nu` | deg, km | equals reference |
| 8 | effective path L_E = L_R nu | `l_e` | km | equals reference |
| 9 | A = gamma_R L_E | `a_rain` | dB | equals reference R1 (9e-16 dB) and `itur.models.itu618.rain_attenuation(..., p=0.01, R001=R)` (0.0 dB) |
| 10 | scaling to other time percentages | not applied | - | see Sect. 3 |

Elevation geometry is used consistently: the same 30 deg enters P.838
(polarisation-elevation term), L_s, L_G, zeta and nu. Rain is liquid only
(snow excluded); with total precipitation the label statistics change by
less than 1 % (`label_sensitivity.csv`).

## 2. Representative hours (stored values vs references)

R1 = independent P.618 implementation written from the Recommendation;
R2 = `itur`'s own P.618 function; alternative = gamma_R x L_s (geometric slant
path, no P.618 factors). Gas reference: `itur` P.676 approximate slant path with
dry-air pressure and an independent P.453 vapour pressure; cloud reference:
P.840 double-Debye K_l implemented independently.

| City, case | R (mm/h) | L_s (km) | gamma_R (dB/km) | r_0.01 | nu_0.01 | L_E (km) | A_rain ours / R1 / R2 (dB) | gamma x L_s (dB) | A_gas ours / ref | A_cloud ours / ref | EPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Athens dry | 0 | 4.67 | 0 | - | - | 0 | 0 / 0 / 0 | 0 | 0.676 / 0.677 | 0.128 / 0.128 | 0.80 |
| Athens moderate | 1.0 | 4.67 | 0.414 | 1.279 | 1.425 | 6.65 | 2.752 (all three) | 1.93 | 0.704 / 0.705 | 0.510 / 0.510 | 3.97 |
| Athens heavy (max) | 7.4 | 4.67 | 2.321 | 0.997 | 1.371 | 6.39 | 14.82 (all three) | 10.83 | 1.057 / 1.062 | 2.679 / 2.679 | 18.55 |
| Bergen dry | 0 | 3.30 | 0 | - | - | 0 | 0 | 0 | 0.770 / 0.772 | 0.156 / 0.156 | 0.93 |
| Bergen moderate | 1.0 | 3.30 | 0.414 | 1.321 | 1.431 | 4.73 | 1.955 (all three) | 1.37 | 0.696 / 0.697 | 0.539 / 0.539 | 3.19 |
| Bergen heavy (max) | 8.9 | 3.30 | 2.721 | 1.031 | 1.379 | 4.56 | 12.39 (all three) | 8.99 | 0.997 / 1.001 | 3.353 / 3.353 | 16.74 |
| Vigo dry | 0 | 5.04 | 0 | - | - | 0 | 0 | 0 | 0.817 / 0.819 | 0.227 / 0.227 | 1.04 |
| Vigo moderate | 1.0 | 5.04 | 0.414 | 1.269 | 1.423 | 7.17 | 2.966 (all three) | 2.08 | 0.885 / 0.888 | 0.532 / 0.532 | 4.38 |
| Vigo heavy (max) | 11.1 | 5.04 | 3.291 | 0.915 | 1.355 | 6.25 | 20.55 (all three) | 16.58 | 0.951 / 0.954 | 4.125 / 4.125 | 25.63 |

(Exact values with full precision: `results/replication_audit/rain_label_diagnostic.csv`.)

Maximum differences over these hours: rain vs R1 9e-16 dB, vs R2 0.0 dB;
gamma vs k R^alpha 4e-16; cloud vs double-Debye 4e-16 dB (K_l = 1.23267 both);
gas vs reference 0.005 dB (our vapour-pressure formula omits P.453's
enhancement factor of about 1.0007; a 0.5 % effect, left unchanged because
fixing it would regenerate every dataset for no material gain). An
apparent 0.033 dB gas difference in a first version of this diagnostic came
from passing *total* pressure to `itur`; P.676-12 Annex 1 and `itur` expect
*dry-air* pressure, which is what the pipeline uses (confirmed: 1e-16 dB with
dry pressure).

## 3. Confirmed errors vs unspecified choices

**Confirmed coding errors: none.** (Earlier bugs fixed on 2026-10-23 and
covered by tests: P.840 temperature in Kelvin instead of Celsius; P.676
ignoring measured T/P; total instead of dry pressure; snow counted as rain.)

**Unspecified choices that change the labels a lot** (paper Sect. 3.1 only
says "P.838 and P.618, with the rain height obtained from P.839"):

1. **Hourly rate used as R_0.01, with P.618's statistical path factors.** At
   39 GHz and 30 deg, nu_0.01 is about 1.36-1.47 and r_0.01 about 0.9-1.6, so the
   effective path L_E is 1.24-1.43 x the geometric slant path for the hours
   above (up to about 2.4 x at very light rain). This follows the formulas
   exactly; it is a consequence of using a statistical (0.01 %-of-time) method
   for instantaneous hourly rates. Using gamma x L_s instead lowers the maxima
   by about 19-27 % (Vigo 20.6 -> 16.6 dB) but does not approach 6.7 dB.
2. **Time percentage p (P.618 step 10).** If the authors called a P.618
   routine with the hourly rate as R_0.01 but a time percentage other than
   0.01 % (for example a library default), every value would be scaled down
   [Hypothesis, `itur.models.itu618.rain_attenuation`]: Vigo at its maximum
   11.1 mm/h gives 20.6 dB (p = 0.01 %), 7.4 dB (p = 0.1 %) or 1.9 dB (p = 1 %);
   at 1 mm/h 3.0 / 0.9 / 0.2 dB. A p of about 0.1 % would bring the tail near the
   paper's visible range. Nothing in the paper confirms this.
3. **Elevation.** Not given; 30 deg in our code. 45 deg lowers the mean by
   about 28 % and the maximum by about 19 %; no elevation up to 90 deg gives a
   maximum below 16 dB (`label_sensitivity.csv`).
4. **Rain-rate definition.** For our model to reach only 6.7 dB EPL, the
   hourly rate would have to stay below 0.5-1.25 mm/h (by city), but ERA5
   exceeds that in 406-5,301 hours per city (max 6.9-11.1 mm/h). A
   different rate (e.g. averaged, or in other units) would be needed
   [Hypothesis].
5. **Cloud proxy, sigma features, P.676/P.618 versions:** affect the level
   by at most a few tenths of a dB; not the cause of the tail.

**What would settle it:** the authors' label table (their dataset, "available
on request"), or their code for one hour with a known rain rate. With either,
one row is enough to identify which of 1-4 they used.

**Decision:** keep the current labels and the hourly results as the
primary measured baseline. Changing p, elevation or the rain path to match
the published numbers would be tuning to the target, so no label change is
made.

## 4. Full-series minute dataset without materialising windows

`data_pipeline/minute_full_source.py` (prototype; not used for training yet):

* Keeps each city's hourly table (43,848 x 14 features + targets, about 2 MB)
  and builds windows on demand with the same `interpolate` and `add_features`
  as `build_minute_dataset.py`; anchors every 5 minutes; inputs 60 minutes;
  targets +1..+5 minutes; chronological 70/15/15 per city with a 65-minute gap.
* Counts [Measured]: 2,208,678 train, 473,214 validation, **473,220 test**
  windows (paper Fig. 4: about 470,000). Materialised float32 inputs would need
  7.4 GB (train) and 1.6 GB (test).
* Tested: windows are identical to the existing minute dataset at the same
  anchor minutes; splits share no minute; stride is 5 minutes.

Still needed before training (not done, by instruction):

1. A `keras.utils.PyDataset` (or `tf.data`) wrapper that shuffles anchor
   indices across cities each epoch and yields batches of 128 scaled windows.
2. Scalers fitted on training data only: the interpolated training minutes
   per city, accumulated with `StandardScaler.partial_fit` (about 1.8 million
   rows x 14 per city, processed city by city).
3. A new variant in `train_paper_model.py` (for example
   `minute_full_fig1heads`) writing to its own folder, plus an evaluation path
   that streams the 473,220 test windows in batches and scores persistence and
   linear extrapolation on the same windows.
4. Cost: about 52 x the hourly dataset per epoch, i.e. about 2.8-3.6 h per
   epoch on this CPU (measured 3.2-4.1 min per 42,000 windows), about 55-70 h
   for 20 epochs. A GPU is justified for this run only.

## 5. Recommended next experiment

Implement items 1-3 above, then run a **one-epoch smoke test on a 1 %
subset of anchors** on this CPU (estimate under 5 minutes) to validate the
generator, scaling and evaluation end to end. Only then decide on a full
run (GPU). Its value: it tests whether a trained model on the paper's
apparent protocol lands near 0.026 dB, and whether it beats linear
extrapolation (0.011 dB) on the same windows. Either way it uses our
labels, so it cannot reproduce the paper's Table 7 city pattern.

Next command now (re-runs this diagnostic and its tests, no training):

```powershell
.venv\Scripts\python evaluation\rain_label_diagnostic.py; .venv\Scripts\python -m unittest discover -s tests -v
```
