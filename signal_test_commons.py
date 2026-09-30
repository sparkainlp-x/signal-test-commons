#!/usr/bin/env python3
# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP.
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline, deterministic, versioned synthetic challenges for 512-channel signal monitors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHALLENGE_VERSION = "2.0.0"
GRID_ROWS = 16
GRID_COLUMNS = 32
CHANNEL_COUNT = GRID_ROWS * GRID_COLUMNS
FRAMES_PER_EPISODE = 20
EPISODES_PER_SCENARIO = 5
NOISE_SIGMA = 1.0
BASELINE_THRESHOLD = 4.5
ROBUST_Z_THRESHOLD = 4.5
MAD_TO_SIGMA = 1.4826
SYNTHETIC_NOTICE = "Synthetic challenge results do not establish performance on physical instruments."


@dataclass(frozen=True)
class ScenarioSpec:
    """Single source of truth for one scenario; the generator reads nothing else.

    Offsets and lengths are in frames relative to the episode start. The injected region is
    either nothing, a ``block_rows`` x ``block_columns`` rectangle of the 16 x 32 grid (placed at
    a seeded location per episode; 1 x 1 means a single channel), or all 512 channels.

    ``additive`` injections add ``amplitude + slope_per_frame * k`` to every channel in the
    region on event frame ``k`` (k = 0 at the event start). ``variance`` injections multiply the
    noise of every channel in the region by ``sigma_factor`` and leave its mean at zero.
    """

    name: str
    description: str
    injection: str = "none"  # none | additive | variance
    region: str = "none"  # none | block | all
    block_rows: int = 0
    block_columns: int = 0
    event_offset_frames: int | None = None
    event_length_frames: int | None = None
    amplitude: float = 0.0
    slope_per_frame: float = 0.0
    sigma_factor: float = 1.0

    @property
    def has_event(self) -> bool:
        return self.injection != "none"

    def shift_at(self, event_frame: int) -> float:
        return self.amplitude + self.slope_per_frame * event_frame

    def parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "noise_sigma": NOISE_SIGMA,
            "episode_length_frames": FRAMES_PER_EPISODE,
            "injection": self.injection,
            "region": self.region,
        }
        if self.has_event:
            params["event_offset_frames"] = self.event_offset_frames
            params["event_length_frames"] = self.event_length_frames
        if self.region == "block":
            params["block_rows"] = self.block_rows
            params["block_columns"] = self.block_columns
        if self.injection == "additive":
            params["amplitude"] = self.amplitude
            params["slope_per_frame"] = self.slope_per_frame
        elif self.injection == "variance":
            params["sigma_factor"] = self.sigma_factor
        return params


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("stable-noise", "Noise only; clean reference frames."),
    ScenarioSpec(
        "localized-burst", "One channel receives +8.0 for 3 frames.",
        injection="additive", region="block", block_rows=1, block_columns=1,
        event_offset_frames=7, event_length_frames=3, amplitude=8.0,
    ),
    ScenarioSpec(
        "block-drift", "A 3x4 block receives +5.0 rising by 0.35 per event frame for 8 frames.",
        injection="additive", region="block", block_rows=3, block_columns=4,
        event_offset_frames=5, event_length_frames=8, amplitude=5.0, slope_per_frame=0.35,
    ),
    ScenarioSpec(
        "global-shock", "All 512 channels receive +6.0 for 2 frames.",
        injection="additive", region="all", event_offset_frames=8, event_length_frames=2,
        amplitude=6.0,
    ),
    ScenarioSpec(
        "weak-burst", "A 2x2 block receives +3.0 for 4 frames.",
        injection="additive", region="block", block_rows=2, block_columns=2,
        event_offset_frames=6, event_length_frames=4, amplitude=3.0,
    ),
    ScenarioSpec(
        "slow-ramp", "A 3x4 block ramps from +0.4 by +0.4 per event frame for 12 frames (to +4.8).",
        injection="additive", region="block", block_rows=3, block_columns=4,
        event_offset_frames=4, event_length_frames=12, amplitude=0.4, slope_per_frame=0.4,
    ),
    ScenarioSpec(
        "negative-shift", "A 3x4 block receives -4.0 for 4 frames.",
        injection="additive", region="block", block_rows=3, block_columns=4,
        event_offset_frames=7, event_length_frames=4, amplitude=-4.0,
    ),
    ScenarioSpec(
        "variance-change", "A 4x4 block has its noise sigma multiplied by 3 for 6 frames (mean unchanged).",
        injection="variance", region="block", block_rows=4, block_columns=4,
        event_offset_frames=6, event_length_frames=6, sigma_factor=3.0,
    ),
    ScenarioSpec(
        "baseline-step", "A 4x8 block steps to +3.5 at frame 10 and stays shifted to the episode end.",
        injection="additive", region="block", block_rows=4, block_columns=8,
        event_offset_frames=10, event_length_frames=FRAMES_PER_EPISODE - 10, amplitude=3.5,
    ),
)
SCENARIO_BY_NAME: dict[str, ScenarioSpec] = {spec.name: spec for spec in SCENARIOS}
SCENARIO_PARAMETERS: dict[str, dict[str, Any]] = {spec.name: spec.parameters() for spec in SCENARIOS}


def _check_specs(specs: Sequence[ScenarioSpec]) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("scenario names must be unique")
    for spec in specs:
        if spec.injection not in {"none", "additive", "variance"}:
            raise ValueError(f"{spec.name}: unknown injection {spec.injection!r}")
        if spec.region not in {"none", "block", "all"}:
            raise ValueError(f"{spec.name}: unknown region {spec.region!r}")
        if (spec.injection == "none") != (spec.region == "none"):
            raise ValueError(f"{spec.name}: region and injection must both be 'none' or neither")
        if spec.has_event:
            start, length = spec.event_offset_frames, spec.event_length_frames
            if start is None or length is None or start < 0 or length < 1:
                raise ValueError(f"{spec.name}: invalid event window")
            if start + length > FRAMES_PER_EPISODE:
                raise ValueError(f"{spec.name}: event window exceeds the episode")
        if spec.region == "block" and not (
            1 <= spec.block_rows <= GRID_ROWS and 1 <= spec.block_columns <= GRID_COLUMNS
        ):
            raise ValueError(f"{spec.name}: invalid block size")


_check_specs(SCENARIOS)


@dataclass(frozen=True)
class Frame:
    """One frame of 512 channel values; channel IDs are row-major grid positions (0-511).

    Frames are contiguous: episodes follow one another without gaps, so a temporal detector
    should reset its state whenever ``episode_start`` is true (``episode_offset == 0``).
    """

    index: int
    scenario: str
    episode_id: str
    episode_offset: int
    values: tuple[float, ...]
    injected_channels: tuple[int, ...] = ()
    event_id: str | None = None

    @property
    def episode_start(self) -> bool:
        return self.episode_offset == 0


@dataclass(frozen=True)
class Episode:
    """A 20-frame scenario window and, when present, its injected event interval."""

    episode_id: str
    scenario: str
    repetition: int
    start_frame: int
    end_frame: int  # inclusive
    event_start_frame: int | None
    event_end_frame: int | None  # inclusive
    injected_channels: tuple[int, ...]
    parameters: Mapping[str, Any]

    @property
    def has_event(self) -> bool:
        return self.event_start_frame is not None


@dataclass(frozen=True)
class Challenge:
    seed: int
    version: str
    frames: tuple[Frame, ...]
    episodes: tuple[Episode, ...]


@dataclass(frozen=True)
class Detection:
    """Detector output for one frame; ``channels`` holds predicted anomalous channel IDs.

    ``alarm=True`` with no channels is allowed (alarm without localization); ``alarm=False``
    with channels is rejected by the evaluator.
    """

    frame_index: int
    alarm: bool
    channels: tuple[int, ...] = ()


def _rectangle_channels(top: int, left: int, height: int, width: int) -> tuple[int, ...]:
    return tuple(
        row * GRID_COLUMNS + column
        for row in range(top, top + height)
        for column in range(left, left + width)
    )


def generate_challenge(seed: int = 7) -> Challenge:
    """Generate every scenario ``EPISODES_PER_SCENARIO`` times from a single seeded RNG.

    Episodes are ordered by repetition, then by ``SCENARIOS`` order. All region locations are
    drawn first (in episode order), then the frame noise, so the layout is part of the replay.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer (not bool)")

    rng = random.Random(seed)
    plans: list[tuple[ScenarioSpec, int, tuple[int, ...], dict[str, Any]]] = []
    for repetition in range(1, EPISODES_PER_SCENARIO + 1):
        for spec in SCENARIOS:
            parameters = spec.parameters()
            if spec.region == "block":
                top = rng.randrange(GRID_ROWS - spec.block_rows + 1)
                left = rng.randrange(GRID_COLUMNS - spec.block_columns + 1)
                channels = _rectangle_channels(top, left, spec.block_rows, spec.block_columns)
                parameters.update(top_row=top, left_column=left)
            elif spec.region == "all":
                channels = tuple(range(CHANNEL_COUNT))
            else:
                channels = ()
            plans.append((spec, repetition, channels, parameters))

    frames: list[Frame] = []
    episodes: list[Episode] = []
    for spec, repetition, channels, parameters in plans:
        episode_id = f"{spec.name}-{repetition}"
        start = len(frames)
        if spec.has_event:
            event_start: int | None = start + spec.event_offset_frames  # type: ignore[operator]
            event_end: int | None = event_start + spec.event_length_frames - 1  # type: ignore[operator]
        else:
            event_start = event_end = None
        episodes.append(
            Episode(
                episode_id=episode_id,
                scenario=spec.name,
                repetition=repetition,
                start_frame=start,
                end_frame=start + FRAMES_PER_EPISODE - 1,
                event_start_frame=event_start,
                event_end_frame=event_end,
                injected_channels=channels,
                parameters=parameters,
            )
        )
        for offset in range(FRAMES_PER_EPISODE):
            index = start + offset
            values = [rng.gauss(0.0, NOISE_SIGMA) for _ in range(CHANNEL_COUNT)]
            active = event_start is not None and event_start <= index <= event_end  # type: ignore[operator]
            if active:
                k = index - event_start  # type: ignore[operator]
                if spec.injection == "additive":
                    shift = spec.shift_at(k)
                    for channel in channels:
                        values[channel] += shift
                else:  # variance
                    for channel in channels:
                        values[channel] *= spec.sigma_factor
            frames.append(
                Frame(
                    index=index,
                    scenario=spec.name,
                    episode_id=episode_id,
                    episode_offset=offset,
                    values=tuple(values),
                    injected_channels=channels if active else (),
                    event_id=episode_id if active else None,
                )
            )

    return Challenge(seed=seed, version=CHALLENGE_VERSION, frames=tuple(frames), episodes=tuple(episodes))


def frame_values_sha256(challenge: Challenge) -> str:
    """SHA-256 of all frame values packed as little-endian IEEE-754 doubles, in frame order."""
    digest = hashlib.sha256()
    for frame in challenge.frames:
        digest.update(struct.pack(f"<{len(frame.values)}d", *frame.values))
    return digest.hexdigest()


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    value = float(threshold)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("threshold must be finite and greater than zero")
    return value


def baseline_detect(
    challenge: Challenge, threshold: float = BASELINE_THRESHOLD
) -> tuple[Detection, ...]:
    """Absolute-threshold baseline: flag channels whose ``abs(value) >= threshold``."""
    if not isinstance(challenge, Challenge):
        raise TypeError("challenge must be a Challenge")
    limit = _validate_threshold(threshold)
    detections = []
    for frame in challenge.frames:
        channels = tuple(c for c, value in enumerate(frame.values) if abs(value) >= limit)
        detections.append(Detection(frame.index, bool(channels), channels))
    return tuple(detections)


def robust_z_detect(
    challenge: Challenge, threshold: float = ROBUST_Z_THRESHOLD
) -> tuple[Detection, ...]:
    """Per-frame median/MAD robust z: flag channels with ``abs(x - median) / (1.4826 * MAD) >= threshold``.

    It is relative to the frame itself, so it cannot see a shift shared by most channels.
    """
    if not isinstance(challenge, Challenge):
        raise TypeError("challenge must be a Challenge")
    limit = _validate_threshold(threshold)
    detections = []
    for frame in challenge.frames:
        center = statistics.median(frame.values)
        mad = statistics.median(abs(value - center) for value in frame.values)
        scale = max(MAD_TO_SIGMA * mad, 1e-12)
        channels = tuple(
            c for c, value in enumerate(frame.values) if abs(value - center) / scale >= limit
        )
        detections.append(Detection(frame.index, bool(channels), channels))
    return tuple(detections)


REFERENCE_DETECTORS = {
    "absolute-threshold": baseline_detect,
    "robust-z": robust_z_detect,
}


def _validated_predictions(
    challenge: Challenge, detections: Iterable[Detection]
) -> dict[int, Detection]:
    try:
        items = tuple(detections)
    except TypeError as exc:
        raise TypeError("detections must be an iterable of Detection objects") from exc

    by_frame: dict[int, Detection] = {}
    for detection in items:
        if not isinstance(detection, Detection):
            raise TypeError("every prediction must be a Detection")
        if isinstance(detection.frame_index, bool) or not isinstance(detection.frame_index, int):
            raise TypeError("detection frame_index must be an integer")
        if detection.frame_index < 0 or detection.frame_index >= len(challenge.frames):
            raise ValueError(f"detection frame_index out of range: {detection.frame_index}")
        if detection.frame_index in by_frame:
            raise ValueError(f"duplicate prediction for frame {detection.frame_index}")
        if not isinstance(detection.alarm, bool):
            raise TypeError("detection alarm must be bool")
        try:
            channels = tuple(detection.channels)
        except TypeError as exc:
            raise TypeError("detection channels must be iterable") from exc
        if any(isinstance(c, bool) or not isinstance(c, int) for c in channels):
            raise TypeError("channel IDs must be integers")
        if len(set(channels)) != len(channels):
            raise ValueError("channel IDs must not contain duplicates")
        if any(c < 0 or c >= CHANNEL_COUNT for c in channels):
            raise ValueError(f"channel IDs must be in [0, {CHANNEL_COUNT - 1}]")
        if channels and not detection.alarm:
            raise ValueError(
                f"frame {detection.frame_index}: alarm=False must not carry predicted channels"
            )
        by_frame[detection.frame_index] = Detection(detection.frame_index, detection.alarm, channels)

    expected = set(range(len(challenge.frames)))
    if set(by_frame) != expected:
        missing = sorted(expected - set(by_frame))
        raise ValueError(f"exactly one prediction per challenge frame is required; missing: {missing[:5]}")
    return by_frame


def _ratio(numerator: float, denominator: float, empty: float | None = 0.0) -> float | None:
    return numerator / denominator if denominator else empty


def _score(
    episodes: Sequence[Episode], frames: Sequence[Frame], predictions: Mapping[int, Detection],
    empty: float | None,
) -> dict[str, Any]:
    fault = [e for e in episodes if e.has_event]
    latencies: list[int] = []
    for episode in fault:
        for index in range(episode.event_start_frame, episode.event_end_frame + 1):  # type: ignore[arg-type,operator]
            if predictions[index].alarm:
                latencies.append(index - episode.event_start_frame)  # type: ignore[operator]
                break
    clean = [f.index for f in frames if not f.injected_channels]
    false_alarms = sum(predictions[i].alarm for i in clean)
    intersection = union = injected_pairs = 0
    for frame in frames:
        if frame.injected_channels:
            truth = set(frame.injected_channels)
            predicted = set(predictions[frame.index].channels)
            intersection += len(truth & predicted)
            union += len(truth | predicted)
            injected_pairs += len(truth)
    return {
        "event_detection_recall": _ratio(len(latencies), len(fault), empty),
        "false_alarms_per_clean_frame": _ratio(false_alarms, len(clean), empty),
        "channel_localization_accuracy": _ratio(intersection, union, empty),
        "mean_detection_latency_frames": statistics.fmean(latencies) if latencies else None,
        "fault_episodes": len(fault),
        "detected_fault_episodes": len(latencies),
        "false_alarm_frames": false_alarms,
        "clean_frames": len(clean),
        "localized_channel_frame_pairs": intersection,
        "localization_union_pairs": union,
        "injected_channel_frame_pairs": injected_pairs,
    }


def evaluate(challenge: Challenge, detections: Iterable[Detection]) -> dict[str, Any]:
    """Return overall metrics, a per-scenario breakdown, and per-episode detection latency."""
    if not isinstance(challenge, Challenge):
        raise TypeError("challenge must be a Challenge")
    predictions = _validated_predictions(challenge, detections)

    metrics = _score(challenge.episodes, challenge.frames, predictions, empty=0.0)
    per_scenario: dict[str, Any] = {}
    for spec in SCENARIOS:
        episodes = [e for e in challenge.episodes if e.scenario == spec.name]
        frames = [f for f in challenge.frames if f.scenario == spec.name]
        per_scenario[spec.name] = _score(episodes, frames, predictions, empty=None)
    per_episode = []
    for episode in challenge.episodes:
        if not episode.has_event:
            continue
        latency = next(
            (
                index - episode.event_start_frame  # type: ignore[operator]
                for index in range(episode.event_start_frame, episode.event_end_frame + 1)  # type: ignore[arg-type,operator]
                if predictions[index].alarm
            ),
            None,
        )
        per_episode.append(
            {"episode_id": episode.episode_id, "detected": latency is not None, "detection_latency_frames": latency}
        )
    metrics["per_scenario"] = per_scenario
    metrics["per_episode"] = per_episode
    return metrics


METRIC_DEFINITIONS = {
    "event_detection_recall": "Fault episodes with at least one alarm in the labeled event interval / fault episodes.",
    "false_alarms_per_clean_frame": "Alarmed frames with no injected channels / frames with no injected channels.",
    "channel_localization_accuracy": "Micro Jaccard over injected frames: sum(|predicted ∩ injected channels|) / sum(|predicted ∪ injected channels|).",
    "mean_detection_latency_frames": "Mean over detected fault episodes of (first alarmed frame in the event interval - event start frame); null when nothing is detected.",
    "per_scenario": "The same metrics restricted to each scenario's episodes and frames; null where the denominator is zero.",
}


def build_report(
    challenge: Challenge,
    detections: Iterable[Detection] | Mapping[str, Iterable[Detection]],
    detector_name: str = "custom",
) -> dict[str, Any]:
    """Build a JSON-serializable report with provenance, scenario truth, and metrics.

    ``detections`` is either one detector's predictions (named by ``detector_name``) or a
    mapping of detector name to predictions.
    """
    if isinstance(detections, Mapping):
        runs = dict(detections)
    else:
        runs = {detector_name: detections}
    if not runs:
        raise ValueError("at least one detector is required")
    for name in runs:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("detector_name must be a non-empty string")
    return {
        "challenge_version": challenge.version,
        "seed": challenge.seed,
        "frame_values_sha256": frame_values_sha256(challenge),
        "dimensions": {
            "frames": len(challenge.frames),
            "episodes": len(challenge.episodes),
            "episodes_per_scenario": EPISODES_PER_SCENARIO,
            "frames_per_episode": FRAMES_PER_EPISODE,
            "channels": CHANNEL_COUNT,
            "grid_rows": GRID_ROWS,
            "grid_columns": GRID_COLUMNS,
        },
        "scenarios": {spec.name: {"description": spec.description, **spec.parameters()} for spec in SCENARIOS},
        "episodes": [
            {
                "episode_id": e.episode_id,
                "scenario": e.scenario,
                "repetition": e.repetition,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "event_start_frame": e.event_start_frame,
                "event_end_frame": e.event_end_frame,
                "injected_channels": list(e.injected_channels),
                "parameters": dict(e.parameters),
            }
            for e in challenge.episodes
        ],
        "detectors": {name: evaluate(challenge, preds) for name, preds in runs.items()},
        "metric_definitions": METRIC_DEFINITIONS,
        "synthetic_only_notice": SYNTHETIC_NOTICE,
    }


SUMMARY_METRICS = (
    "event_detection_recall",
    "false_alarms_per_clean_frame",
    "channel_localization_accuracy",
    "mean_detection_latency_frames",
)


def _stats(values: Iterable[float | None]) -> dict[str, float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "min": None, "max": None}
    return {"mean": statistics.fmean(present), "min": min(present), "max": max(present)}


def multi_seed_summary(
    seeds: Sequence[int],
    threshold: float = BASELINE_THRESHOLD,
    robust_threshold: float = ROBUST_Z_THRESHOLD,
) -> dict[str, Any]:
    """Run both reference detectors on every seed; report mean/min/max of each metric."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    thresholds = {"absolute-threshold": threshold, "robust-z": robust_threshold}
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in REFERENCE_DETECTORS}
    fingerprints = {}
    for seed in seeds:
        challenge = generate_challenge(seed)
        fingerprints[str(seed)] = frame_values_sha256(challenge)
        for name, detector in REFERENCE_DETECTORS.items():
            per_seed[name].append(evaluate(challenge, detector(challenge, thresholds[name])))
    detectors: dict[str, Any] = {}
    for name, runs in per_seed.items():
        detectors[name] = {
            "threshold": thresholds[name],
            "overall": {m: _stats(run[m] for run in runs) for m in SUMMARY_METRICS},
            "per_scenario": {
                spec.name: {m: _stats(run["per_scenario"][spec.name][m] for run in runs) for m in SUMMARY_METRICS}
                for spec in SCENARIOS
            },
        }
    return {
        "challenge_version": CHALLENGE_VERSION,
        "seeds": seeds,
        "frame_values_sha256": fingerprints,
        "detectors": detectors,
        "metric_definitions": METRIC_DEFINITIONS,
        "synthetic_only_notice": SYNTHETIC_NOTICE,
    }


def parse_seeds(text: str) -> list[int]:
    """Parse ``"1-10"``, ``"1,2,5"`` or a mix such as ``"1-3,8"`` into a list of seeds."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"invalid seed list: {text!r}")
        try:
            if "-" in part[1:]:
                split = part.index("-", 1)
                low, high = int(part[:split]), int(part[split + 1:])
                if high < low:
                    raise ValueError(f"invalid seed range: {part!r}")
                seeds.extend(range(low, high + 1))
            else:
                seeds.append(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid seed list {text!r}: {exc}") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return seeds


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a UTF-8 JSON report, creating its parent directory when needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _print_single(report: Mapping[str, Any]) -> None:
    print(f"challenge {report['challenge_version']}  seed {report['seed']}  "
          f"frames {report['dimensions']['frames']}  sha256 {report['frame_values_sha256'][:16]}…")
    for name, metrics in report["detectors"].items():
        print(f"\n[{name}] recall {_fmt(metrics['event_detection_recall'])}  "
              f"false-alarms/clean-frame {_fmt(metrics['false_alarms_per_clean_frame'])}  "
              f"localization {_fmt(metrics['channel_localization_accuracy'])}  "
              f"mean latency {_fmt(metrics['mean_detection_latency_frames'], 2)}")
        print(f"  {'scenario':<16} {'recall':>6} {'local.':>6} {'latency':>7} {'FA/clean':>8}")
        for scenario, row in metrics["per_scenario"].items():
            print(f"  {scenario:<16} {_fmt(row['event_detection_recall'], 2):>6} "
                  f"{_fmt(row['channel_localization_accuracy']):>6} "
                  f"{_fmt(row['mean_detection_latency_frames'], 2):>7} "
                  f"{_fmt(row['false_alarms_per_clean_frame']):>8}")


def _print_summary(summary: Mapping[str, Any]) -> None:
    seeds = summary["seeds"]
    print(f"challenge {summary['challenge_version']}  seeds {len(seeds)} ({seeds[0]}..{seeds[-1]})  mean [min, max]")
    for name, block in summary["detectors"].items():
        print(f"\n[{name}]")
        for metric, stats in block["overall"].items():
            print(f"  {metric:<32} {_fmt(stats['mean'])} [{_fmt(stats['min'])}, {_fmt(stats['max'])}]")
        print(f"  {'scenario':<16} {'recall mean [min, max]':>24} {'localization mean':>18}")
        for scenario, row in block["per_scenario"].items():
            recall = row["event_detection_recall"]
            print(f"  {scenario:<16} {_fmt(recall['mean'], 2):>10} [{_fmt(recall['min'], 2)}, {_fmt(recall['max'], 2)}] "
                  f"{_fmt(row['channel_localization_accuracy']['mean']):>11}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=7, help="integer replay seed (default: 7)")
    seed_group.add_argument(
        "--seeds", type=str, default=None,
        help="multi-seed summary, e.g. '1-10' or '1,2,5'; reports mean/min/max per metric",
    )
    parser.add_argument(
        "--threshold", type=float, default=BASELINE_THRESHOLD,
        help=f"absolute-threshold baseline limit (default: {BASELINE_THRESHOLD})",
    )
    parser.add_argument(
        "--robust-threshold", type=float, default=ROBUST_Z_THRESHOLD,
        help=f"median/MAD robust-z limit (default: {ROBUST_Z_THRESHOLD})",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("challenge_report.json"),
        help="output JSON report path (default: ./challenge_report.json)",
    )
    args = parser.parse_args(argv)
    try:
        threshold = _validate_threshold(args.threshold)
        robust_threshold = _validate_threshold(args.robust_threshold)
        if args.seeds is not None:
            report = multi_seed_summary(parse_seeds(args.seeds), threshold, robust_threshold)
        else:
            challenge = generate_challenge(args.seed)
            report = build_report(
                challenge,
                {
                    f"absolute-threshold (threshold={threshold:g})": baseline_detect(challenge, threshold),
                    f"robust-z (threshold={robust_threshold:g})": robust_z_detect(challenge, robust_threshold),
                },
            )
        output = write_report(report, args.report)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {output}")
    if args.seeds is not None:
        _print_summary(report)
    else:
        _print_single(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
