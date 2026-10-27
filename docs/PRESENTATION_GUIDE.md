# Presentation guide (8-10 minutes + questions)

Format: a university research/project presentation followed by technical
questions.

* **Slides:** 9, about 9 minutes.
* **Figures:** all exist in the repository. Paths are relative to the
  repository root.
* **Numbers:** every number is taken from
  `results/causal_hourly_fig1heads_seed42/metrics_test.json` or
  `results/figures/figures_manifest.json`.
* **Preparation:** read `docs/PROJECT_EXPLAINED_FROM_SCRATCH.md` first.

> **Do not use the two older diagrams** `docs/assets/figures/fig1_architecture.png`
> and `fig2_pipeline.png` without changes:
> * the architecture diagram shows the earlier "main" model, without the
>   Dense(256) head layers (1,337,123 parameters);
> * the pipeline diagram shows the earlier stratified dataset (10,000
>   anchors per city).
>
> Use the simple diagrams given on slides 3-5 below. They match the causal
> experiment.

## Timing overview

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Why satellite links fade | 0:50 | 0:50 |
| 2 | Research question and reference paper | 0:50 | 1:40 |
| 3 | From weather to attenuation labels | 1:10 | 2:50 |
| 4 | Causal data construction | 1:00 | 3:50 |
| 5 | Model and baselines | 1:00 | 4:50 |
| 6 | Main results | 1:20 | 6:10 |
| 7 | Where it fails: heavy rain and uncertainty | 1:10 | 7:20 |
| 8 | Why we do not match the paper's 0.026 dB | 1:00 | 8:20 |
| 9 | Conclusions and future work | 0:40 | 9:00 |

## Opening (first 15 seconds)

> "Satellite internet at high frequencies has a weather problem: rain can wipe
> out most of the signal. My project asks whether a neural network can forecast
> that loss a few hours ahead, and, just as importantly, where it can't."

---

## Slide 1: Why satellite links fade

**Purpose:** motivate the problem in plain language.

**Bullets:**
* LEO satellites: hundreds of km up, thousands of them, broadband at Q-band
  (39 GHz).
* At 39 GHz, gas, cloud and especially **rain** absorb the signal.
* The extra loss is called the **excess path loss (EPL)** and is measured in
  dB (10 dB = one tenth of the power remains).
* Today operators use fixed margins of about 3-5 dB. A forecast could let
  links adapt in advance.

**Show:** `results/figures/12_attenuation_components.png`, left panel (mean
gas, cloud and rain per city).

**Speaker notes:**
> "At 39 GHz the atmosphere is not transparent. Oxygen and water vapour take
> a small, steady bite of about three quarters of a dB. Clouds add a little.
> Rain is the big one: it's zero most of the time but can reach over 20 dB.
> We call the sum the excess path loss. If we could forecast it, the link
> could change modulation or power before the fade hits, instead of wasting
> a large safety margin all the time."

**Transition:** "This idea comes from a recent paper, which is where I started."

---

## Slide 2: Research question and reference paper

**Purpose:** state what was attempted and frame the claim honestly.

**Bullets:**
* Reference: El-Sayed et al., *EURASIP JWCN* 2026:102. A multi-task LSTM
  forecasts EPL 5 h ahead and reports RMSE **0.026 dB**.
* Their data and code are not available, and key details are unspecified.
* My question: *can an LSTM forecast 39 GHz EPL 1-5 h ahead better than
  simple baselines, under a strictly causal evaluation?*
* This is an independent implementation, **not an exact reproduction**.

**Show:** a text slide; optionally the paper's title and DOI.

**Speaker notes:**
> "The paper describes a three-layer LSTM with two output heads and reports
> an error of only 0.026 dB. I implemented the same architecture from the
> description. But the authors' dataset isn't available, and several choices,
> like the elevation angle and the exact forecasting setup, aren't specified.
> So I set myself a cleaner question: under a strictly time-ordered
> evaluation, does the model beat simple baselines? I'll come back to the
> 0.026 at the end."

**Transition:** "First, where do the training targets come from?"

---

## Slide 3: From weather to attenuation labels

**Purpose:** explain the data and the label pipeline, and that the labels
are reconstructed.

**Diagram to draw (simple boxes):**

```text
ERA5 hourly weather        ITU-R models (39 GHz, 30 deg)         EPL label (dB)
(Open-Meteo, 2020-2024) -> gas P.676 + cloud P.840 (proxy) -> = gas + cloud + rain
6 cities                   + rain P.838 / P.618 / P.839
```

**Bullets:**
* ERA5 is an hourly weather reanalysis; we use six cities from 2020 to 2024.
* Standard ITU-R models convert weather into dB of attenuation.
* The labels are **reconstructed**, not measured satellite signals.
* Rain dominates: zero in 79 % of hours, but up to 25.6 dB.

**Show:** `results/figures/11_epl_distribution.png`.

**Explaining the figure:**
* Left: a histogram of all hourly EPL values, with a logarithmic vertical
  axis. Most hours are near 1 dB, and there is a long tail out to 25 dB.
* Right: the share of hours above each threshold. About 1 % of values
  exceed 6.7 dB.

**Speaker notes:**
> "There are no public satellite measurements for these cities, so like the
> paper I generate the targets from physics. I take hourly ERA5 weather and
> apply the ITU-R propagation models: one for gases, one for clouds, one for
> rain. The result is an hourly attenuation label. Two things matter. First,
> these labels are model outputs, not measurements. Second, the distribution
> is extremely skewed: the median is under 1 dB, but rain produces a tail up
> to 25 dB. That tail will matter a lot for the results."

**Transition:** "Next: how I turned this time series into forecasting examples
without cheating."

---

## Slide 4: Causal data construction

**Purpose:** show that the evaluation has no time leakage.

**Diagram to draw:**

```text
 ... hour t-59 ............ hour t | t+1  t+2  t+3  t+4  t+5
 [ 60 hours of 14 input features ] | [ 5 EPL targets         ]
             (past only)           |  forecast origin = hour t

 Per city: |---- train 70 % ----| gap |-- val 15 % --| gap |-- test 15 % --|
            2020-01 .. 2023-07        2023-07 .. 2024-04     2024-04 .. 2024-12
```

**Bullets:**
* Every hour is a forecast origin. The inputs are the past 60 hours of 14
  features; the targets are EPL at +1 to +5 h.
* The split is chronological per city, with gaps of at least 65 h, so no hour
  is shared between splits.
* Scalers are fitted on training data only.
* Unit tests check that changing future data does not change the inputs.
* Test set: **39,000 origins x 5 horizons = 195,000 values**.

**Show:** the diagram above.

**Speaker notes:**
> "Each training example is: the last 60 hours of weather and attenuation,
> and the attenuation for each of the next five hours. I split each city's
> timeline in order: about three and a half years for training, then
> validation, then the last nine months of 2024 for testing, with gaps so no
> hour is shared. Normalisation uses only training statistics, and I wrote
> tests that perturb future values and check the inputs don't change. The
> test set has 39,000 forecast times, and five horizons each gives 195,000
> values."

**Transition:** "Now the model, and what it has to beat."

---

## Slide 5: Model and baselines

**Purpose:** explain the architecture and why persistence is the key
baseline.

**Diagram to draw:**

```text
Input 60 x 14 -> LSTM 256 -> LSTM 256 -> LSTM 256 (last state)
                                 |-> Dense 256 -> 25  = weather, 5 h x 5 variables (MSE)
                                 |-> Dense 256 -> 10  = EPL mean + log-variance, 5 h (Gaussian NLL)
1,468,707 parameters; Adam 1e-4; batch 128; early stopping (patience 6); seed 42
```

**Bullets:**
* A three-layer LSTM with two heads: a weather forecast (auxiliary task) and
  an EPL forecast with uncertainty.
* The Gaussian negative log-likelihood loss trains both the mean and the
  spread.
* Early stopping on validation loss selected epoch **5 of 11**.
* Baselines on identical test cases:
  * **persistence**: repeat the current EPL;
  * the 24-h mean;
  * linear extrapolation.

**Show:** the diagram above; optionally
`results/figures/01_training_curves.png` (the dashed line marks epoch 5).

**Speaker notes:**
> "The model reads the 60-hour sequence through three stacked LSTM layers.
> One head predicts the weather, as in the paper's multi-task design. The
> other predicts, for each of the five hours, a mean and a variance, so we
> get an uncertainty. Training stopped automatically after validation loss
> stopped improving; the best epoch was 5. The most important comparison is
> persistence: just assume the attenuation stays as it is now. For weather,
> that's a surprisingly strong baseline."

**Transition:** "So, does it beat persistence?"

---

## Slide 6: Main results

**Purpose:** present the headline numbers fairly.

**Table on the slide:**

| Method | RMSE (dB) | MAE (dB) |
|---|---|---|
| LSTM | **0.859** | **0.335** |
| Persistence | 1.018 | 0.346 |
| 24-h mean | 1.126 | 0.494 |
| Linear extrapolation | 2.519 | 0.718 |

**Bullets:**
* RMSE is **15.6 % lower** than persistence; MAE is only **3.2 % lower**.
* The gain grows with horizon: -0.02 dB at +1 h, -0.22 dB at +5 h.
* RMSE is lower in all six cities.
* MAE is slightly *worse* than persistence at +1 h and +2 h.

**Show:** `results/figures/06_baseline_comparison.png`, then
`results/figures/05_horizon_rmse_mae.png`.

**Explaining Figure 5:** the horizontal axis is how far ahead the forecast
is, and the vertical axis is the error in dB. Orange is persistence, blue is
the LSTM. The gap opens as the horizon grows.

**Speaker notes:**
> "On the full test set the LSTM has an RMSE of 0.86 dB versus 1.02 for
> persistence, about 16 % lower, and it's lower in every city. But look at
> MAE: only 3 % better, and at one and two hours ahead persistence actually
> wins on MAE. RMSE punishes big errors, MAE treats all errors equally. So
> the model avoids some big mistakes, mostly when rain is changing and
> persistence repeats a stale value, while in steady clear sky persistence is
> already almost perfect. The benefit grows with lead time: by five hours,
> the RMSE gap is over 0.2 dB."

**Transition:** "Where does it go wrong? Exactly where it matters most."

---

## Slide 7: Where it fails: heavy rain and uncertainty

**Purpose:** show the limitations clearly.

**Bullets:**
* Heavy rain (>= 2.5 mm/h, 1,475 values): LSTM RMSE **6.70 dB** vs
  persistence **5.85 dB**, so the model is worse.
* Rain peaks are under-predicted: the true maximum is 25.6 dB, the largest
  prediction 8.9 dB.
* The nominal 95 % interval covers 97.4 % overall, but **46 %** in heavy
  rain and 99.5 % in dry hours.
* Overall coverage hides conditional miscalibration.

**Show:** `results/figures/08_precipitation_classes.png` and
`results/figures/09_calibration_reliability.png`. Optionally use
`results/figures/02_timeseries_examples.png`, panel B (the Vigo event).

**Explaining Figure 9:**
* The diagonal is perfect calibration.
* Points above it mean the intervals are too wide; points below mean too
  narrow.
* The dry line sits high, and the heavy-rain line sits far below.

**Speaker notes:**
> "Split by rain intensity, the picture changes. In heavy rain the LSTM is
> worse than persistence: it predicts a smoothed, much smaller peak. In the
> worst event in the test set, Vigo reached 25.6 dB and the model predicted
> about 8. The uncertainty has the same problem. Overall, the 95 % interval
> contains 97 % of the truth, which sounds great. But in heavy rain it
> contains only 46 %. In clear sky the intervals are too wide; in rain, far
> too narrow. Average calibration is not the same as useful calibration."

**Transition:** "Which brings me back to the paper's 0.026 dB."

---

## Slide 8: Why we do not match the paper's 0.026 dB

**Purpose:** explain the gap carefully, without accusing anyone.

**Bullets:**
* Not a controlled comparison: the data, labels and possibly the protocol
  differ.
* **Most likely** cause: a different task. The paper's figures suggest a
  series interpolated to 1 minute and forecast minutes ahead. On our labels
  under that reading, simple baselines score 0.011-0.035 dB.
* The labels differ: ours reach 25.6 dB, while the paper's plots stop near
  6.7 dB.
* The paper's own Table 7 overall (0.026) does not follow from its per-city
  values.
* No bug was found in our pipeline, and the units were verified in dB.

**Show:** `results/figures/13_paper_context.png`.

**Explaining the figure:**
* The axis is logarithmic, and each row is a different task.
* The paper's value sits next to the minute-interpolated baselines, not
  next to hourly forecasts.

**Speaker notes:**
> "Our error is about 30 times larger than the paper's. I spent a lot of time
> on why. I checked units, alignment and the physics against independent
> references, and found no bug. The paper's own figures show one-minute
> sampling and about 470,000 test samples, which matches interpolating the
> hourly data to minutes and forecasting a few minutes ahead. On that task,
> even 'repeat the last value' gets 0.035 dB on our labels, the same order
> as 0.026. So I think we're answering different questions. That's a
> hypothesis, not proof. I also chose not to tune my physics to match their
> number, because that would be fitting to the answer."

**Transition:** "To wrap up."

---

## Slide 9: Conclusions and future work

**Bullets:**
* Under a strictly causal evaluation, the LSTM beats persistence by
  **15.6 % RMSE** (3.2 % MAE).
* It fails where fades matter most, in heavy rain, and its uncertainty is
  unreliable there.
* The labels are reconstructed and this is one seed, so it is a
  methodological result, not an operational one.
* Next steps:
  * more seeds;
  * heavy-rain-aware losses and conditional calibration;
  * validation on measured satellite data.

**Show:** a text slide; optionally a small version of Figure 6.

**Speaker notes / closing:**
> "To sum up: a carefully evaluated LSTM does add value over the simplest
> baseline, especially a few hours ahead, but not yet in heavy rain, which is
> the case that matters most for real links. And published numbers need the
> same causal scrutiny before we compare against them. Thank you. I'm happy
> to take questions."

---

## Five-minute version

| Slide | Time | Change |
|---|---|---|
| 1 | 0:35 | Combine with slide 2 in one sentence |
| 3 + 4 merged | 1:00 | Show only Figure 11 and the window diagram |
| 5 | 0:40 | Show the architecture line and the persistence definition only |
| 6 | 1:00 | Show Figure 6 only; mention the horizon trend verbally |
| 7 | 0:50 | Show Figure 8 only; state the 46 % coverage verbally |
| 8 + 9 merged | 0:55 | One sentence on the paper (different task, not comparable), then the conclusions |

## Likely audience questions

1. **Are your labels real measurements?**
   No. They are ITU-R model outputs from ERA5 weather. That is the main
   limitation.
2. **Why 30 degrees elevation?**
   The paper doesn't give a value. 30 degrees is a typical LEO elevation,
   and the paper's clear-sky floor is consistent with it, but that is
   inferred.
3. **What is persistence and why is it strong?**
   Repeating the current value. Weather changes slowly most of the time, so
   it is hard to beat at short horizons.
4. **Why is the MAE gain so much smaller than the RMSE gain?**
   RMSE weights large errors. The model avoids some large errors but adds
   small ones in clear sky.
5. **How do you know there's no leakage?**
   Inputs use only past hours; there are gapped chronological splits;
   scalers use training data only; and unit tests perturb future values.
6. **Why does the model fail in heavy rain?**
   Heavy rain is rare (under 1 % of values) and hard to predict from past
   weather alone. The model predicts smoothed peaks, and the NLL loss lets
   it inflate sigma instead.
7. **What does the weather head contribute?**
   It is the paper's multi-task design. On our earlier protocol it gave at
   most a small EPL gain (0.006 dB).
8. **How many parameters, and how long did training take?**
   1,468,707 parameters; about 3.1 hours on a laptop CPU, 11 epochs.
9. **Is the improvement statistically significant?**
   Not tested; there is one seed. Three seeds on the earlier protocol varied
   by about 0.01 dB (standard deviation 0.006), which suggests but does not
   prove robustness.
10. **What is the uncertainty used for?**
    In principle, for setting link margins with a chosen risk. In practice,
    the intervals are unreliable in heavy rain, so not yet.
11. **Could you use a weather forecast instead of past weather?**
    Yes, that is a natural next step. Numerical weather or radar nowcasts
    would carry information about rain onset.
12. **Why not tune your labels to match the paper?**
    That would be fitting to a target number, not to physics.
13. **What is the 39,000 vs 195,000?**
    39,000 forecast times; each forecasts 5 hours, giving 195,000 predicted
    values.
14. **Does the model work for other cities?**
    It has not been tested under this protocol. The earlier protocol tested
    ten unseen cities with mixed results.
15. **Why an LSTM and not a Transformer or gradient boosting?**
    To follow the paper's design. Other models are reasonable future
    baselines.
16. **What was the biggest technical challenge?**
    Defining a leakage-free protocol. The minute interpolation suggested by
    the paper leaks future values.
17. **What would make this operational?**
    Real-time inputs, validation on measured links, and reliable
    heavy-rain performance.

## Acknowledge uncertainty; do not bluff

* **Exactly what protocol the paper used.** Say: "My evidence points to a
  minute-interpolated task, but I can't confirm it without their code."
* **Whether the result holds across seeds on this protocol.** Not measured.
* **How the model would perform on measured satellite data.** Unknown.
* **Whether our cloud proxy or rain-label interpretation matches reality.**
  These are documented assumptions; the cloud proxy is weak (held-out R^2
  about 0).
* **Detailed ITU-R internals beyond what the code computes.** Point to the
  Recommendation and `physics/build_attenuation.py`.
* **Why the paper's per-city ranking differs from ours** (for example, Vigo
  is their easiest city and our hardest). This is unexplained.
