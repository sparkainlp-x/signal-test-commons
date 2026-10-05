# Signal Test Commons

[![tests](https://github.com/sparkainlp-x/signal-test-commons/actions/workflows/tests.yml/badge.svg)](https://github.com/sparkainlp-x/signal-test-commons/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23057992.svg)](https://doi.org/10.5281/zenodo.23057992)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#challenge-contents-version-200)
[![Evidence: SYNTHETIC](https://img.shields.io/badge/evidence-SYNTHETIC-blue.svg)](#challenge-contents-version-200)

Signal Test Commons is a compact, deterministic, versioned **synthetic** challenge suite for evaluating 512-channel signal monitors offline. It uses only the Python standard library and generates frames of 512 channels arranged as a 16 × 32 row-major grid. A channel ID is one grid position, `row * 32 + column` (IDs 0–511); predictions and ground truth are always sets of individual channels.

## Quickstart

Requires Python 3.10 or later; no installation or network access is needed.

```sh
python3 signal_test_commons.py --seed 7 --report challenge_report.json
python3 signal_test_commons.py --seeds 1-10
python3 -m unittest -v
python3 -m py_compile signal_test_commons.py test_signal_test_commons.py
```

The single-seed run writes a JSON report with the seed, challenge version, a SHA-256 of the frame values, the scenario specs, every episode's ground truth, and the metrics of both reference detectors. Repeating the same seed with the same challenge version replays the same challenge. `--seeds` accepts a range (`1-10`), a list (`1,2,5`) or a mix (`1-3,8`) and writes a multi-seed summary with the mean, minimum and maximum of each metric. `--threshold` sets the absolute-threshold baseline limit and `--robust-threshold` the robust-z limit (both default to 4.5); `--report` chooses the output path (default `./challenge_report.json`).

To use the generator and evaluator from another local Python program:

```python
from signal_test_commons import generate_challenge, baseline_detect, robust_z_detect, evaluate

challenge = generate_challenge(seed=7)
metrics = evaluate(challenge, baseline_detect(challenge, threshold=4.5))
print(metrics["event_detection_recall"], metrics["per_scenario"]["weak-burst"])
```

## Challenge contents (version 2.0.0)

Every scenario is described once, in the `SCENARIOS` tuple of `ScenarioSpec` objects in `signal_test_commons.py`; the generator and the report both read from it. Each scenario is run **5 times** (`EPISODES_PER_SCENARIO`), giving 45 episodes of 20 frames (900 frames). Every channel is independent seeded Gaussian noise with standard deviation 1.0. Event windows below are inclusive frame offsets within the episode; the timing is fixed per scenario, while the location of each block or channel is drawn from the seed independently for every episode.

| Scenario | Event window | Region (seeded location per episode) | Injection |
| --- | --- | --- | --- |
| `stable-noise` | none | none | Noise only; clean reference frames |
| `localized-burst` | 7–9 | 1 channel | +8.0 |
| `block-drift` | 5–12 | 3 × 4 block | +5.0, rising by 0.35 per event frame |
| `global-shock` | 8–9 | all 512 channels | +6.0 |
| `weak-burst` | 6–9 | 2 × 2 block | +3.0 |
| `slow-ramp` | 4–15 | 3 × 4 block | +0.4, rising by 0.4 per event frame (to +4.8) |
| `negative-shift` | 7–10 | 3 × 4 block | −4.0 |
| `variance-change` | 6–11 | 4 × 4 block | noise sigma × 3, mean unchanged |
| `baseline-step` | 10–19 | 4 × 8 block | +3.5, stays shifted to the end of the episode |

Episodes are ordered by repetition, then by the table order (`stable-noise-1`, `localized-burst-1`, …, `baseline-step-5`). One `random.Random(seed)` stream draws all region locations first, then the frame noise.

**Episodes are contiguous.** Frame indices run 0–899 without gaps, and one episode's last frame is immediately followed by the next episode's first frame, which may belong to a very different scenario. Each `Frame` carries `episode_id`, `episode_offset` (0–19) and the property `episode_start` (`episode_offset == 0`), so a temporal detector can reset its state at every episode boundary. Each frame also records its scenario, event ID (when injected) and exact `injected_channels`.

**Versioning.** `CHALLENGE_VERSION` is `2.0.0`; version 1.0.0 had four single-episode scenarios, so v1 and v2 results are not comparable. The test suite pins the SHA-256 of all seed-7 frame values packed as little-endian IEEE-754 doubles (`frame_values_sha256`, also written to every report). Any change to the generator changes that digest and fails the test, which is the signal to bump the challenge version.

## Reference detectors and metrics

Two deliberately simple reference detectors are included to show that the suite separates different detector behaviors:

- **`absolute-threshold`** (`baseline_detect`): flags each channel with `abs(value) >= threshold` (default 4.5).
- **`robust-z`** (`robust_z_detect`): per frame, computes the median and the median absolute deviation (MAD) of the 512 values and flags channels with `abs(value − median) / (1.4826 × MAD) >= threshold` (default 4.5). It is relative to the frame, so it cannot see a shift shared by most channels.

Both alarm on a frame when at least one channel is flagged. `evaluate()` returns:

- **Event detection recall:** fault episodes with at least one alarm during the labeled event interval, divided by fault episodes.
- **False alarms per clean frame:** alarmed frames containing no injected channels, divided by all frames containing no injected channels.
- **Channel localization accuracy:** micro Jaccard over injected frames, `sum(|predicted ∩ injected|) / sum(|predicted ∪ injected|)`. It penalizes both missed and extra channels. Because it is micro-averaged, the overall value is dominated by the 512-channel `global-shock` frames, so read the per-scenario values too.
- **Mean detection latency:** for each detected fault episode, the number of frames from the event start to the first alarm inside the event interval, averaged over detected episodes (`null` when nothing is detected). Per-episode latencies are listed under `per_episode`.
- **Per-scenario breakdown:** the same metrics restricted to each scenario's episodes and frames (`per_scenario`), with `null` where a denominator is zero, for example recall for `stable-noise`.

The report also includes the counts used by these metrics and plain-text metric definitions. For the overall metrics, a zero denominator gives 0.0.

## Example results (synthetic)

These are real outputs of the commands above for challenge version 2.0.0. They are **synthetic** results for illustrative reference detectors, not measurements of any instrument or product.

Single seed (`python3 signal_test_commons.py --seed 7`):

```text
Wrote challenge_report.json
challenge 2.0.0  seed 7  frames 900  sha256 e203ae8c6c1a92d2…

[absolute-threshold (threshold=4.5)] recall 0.950  false-alarms/clean-frame 0.003  localization 0.654  mean latency 0.74
  scenario         recall local. latency FA/clean
  stable-noise          -      -       -    0.000
  localized-burst    1.00  1.000    0.00    0.000
  block-drift        1.00  0.902    0.00    0.000
  global-shock       1.00  0.929    0.00    0.000
  weak-burst         0.60  0.062    0.33    0.025
  slow-ramp          1.00  0.150    5.40    0.000
  negative-shift     1.00  0.286    0.00    0.000
  variance-change    1.00  0.154    0.00    0.000
  baseline-step      1.00  0.155    0.00    0.000

[robust-z (threshold=4.5)] recall 0.825  false-alarms/clean-frame 0.005  localization 0.091  mean latency 1.00
  scenario         recall local. latency FA/clean
  stable-noise          -      -       -    0.010
  localized-burst    1.00  1.000    0.00    0.012
  block-drift        1.00  0.881    0.00    0.000
  global-shock       0.00  0.000       -    0.000
  weak-burst         0.60  0.037    1.33    0.013
  slow-ramp          1.00  0.135    5.40    0.000
  negative-shift     1.00  0.270    0.00    0.000
  variance-change    1.00  0.152    0.00    0.000
  baseline-step      1.00  0.073    0.40    0.000
```

Multi-seed summary (`python3 signal_test_commons.py --seeds 1-10`):

```text
Wrote challenge_report.json
challenge 2.0.0  seeds 10 (1..10)  mean [min, max]

[absolute-threshold]
  event_detection_recall           0.957 [0.900, 0.975]
  false_alarms_per_clean_frame     0.004 [0.000, 0.009]
  channel_localization_accuracy    0.656 [0.650, 0.661]
  mean_detection_latency_frames    0.889 [0.667, 1.103]
  scenario           recall mean [min, max]  localization mean
  stable-noise              - [-, -]           -
  localized-burst        1.00 [1.00, 1.00]       0.994
  block-drift            1.00 [1.00, 1.00]       0.905
  global-shock           1.00 [1.00, 1.00]       0.934
  weak-burst             0.66 [0.20, 0.80]       0.066
  slow-ramp              1.00 [1.00, 1.00]       0.149
  negative-shift         1.00 [1.00, 1.00]       0.303
  variance-change        1.00 [1.00, 1.00]       0.138
  baseline-step          1.00 [1.00, 1.00]       0.155

[robust-z]
  event_detection_recall           0.832 [0.800, 0.875]
  false_alarms_per_clean_frame     0.006 [0.000, 0.011]
  channel_localization_accuracy    0.091 [0.088, 0.095]
  mean_detection_latency_frames    1.080 [0.879, 1.273]
  scenario           recall mean [min, max]  localization mean
  stable-noise              - [-, -]           -
  localized-burst        1.00 [1.00, 1.00]       1.000
  block-drift            1.00 [1.00, 1.00]       0.876
  global-shock           0.00 [0.00, 0.00]       0.000
  weak-burst             0.66 [0.40, 1.00]       0.062
  slow-ramp              1.00 [1.00, 1.00]       0.135
  negative-shift         1.00 [1.00, 1.00]       0.276
  variance-change        1.00 [1.00, 1.00]       0.130
  baseline-step          1.00 [1.00, 1.00]       0.079
```

What this shows: both detectors find the strong localized events, but the robust-z detector misses every `global-shock` episode because a shift shared by all channels moves the median with it. The absolute threshold misses about a third of the `weak-burst` episodes, detects `slow-ramp` only near the end of the ramp (about 5–6 frames of latency), and localizes the broad or partial changes (`slow-ramp`, `variance-change`, `baseline-step`) poorly. At a threshold of 4.5, both detectors raise very few false alarms on clean frames.

## Plug in another detector

A local detector can consume `challenge.frames`; each frame's 512 values are available as `frame.values`. Return one `Detection(frame_index, alarm, channels)` for **every** frame, where `channels` is an iterable of predicted channel IDs. `alarm=True` with no channels is allowed (an alarm without localization); `alarm=False` with channels is rejected with a `ValueError`. Pass the predictions to `evaluate(challenge, predictions)` or `build_report(challenge, predictions, detector_name="your-detector")`; `build_report` also accepts a mapping of detector names to predictions. The evaluator checks prediction coverage, frame IDs, alarm types, and channel-ID bounds and duplicates. This is a Python API for local experiments, not a live service or network interface.

## Scope

All data and scores are synthetic and deterministic for a given seed and challenge version. They are useful for exercising replay, event scoring, latency and spatial localization logic; **synthetic results do not establish or predict performance on physical instruments**. The scenarios, thresholds and amplitudes are illustrative choices, not calibrated settings. No real instrument data, network access, or external packages are used.

## License

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).

Organizations that want to use it in proprietary products or services without AGPL obligations can contact the author about a commercial license via https://sparkainlpx.xyz.
