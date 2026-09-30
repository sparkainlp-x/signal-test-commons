# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP.
# SPDX-License-Identifier: AGPL-3.0-only
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import signal_test_commons as stc
from signal_test_commons import (
    CHANNEL_COUNT,
    CHALLENGE_VERSION,
    EPISODES_PER_SCENARIO,
    FRAMES_PER_EPISODE,
    SCENARIOS,
    Detection,
    baseline_detect,
    build_report,
    evaluate,
    frame_values_sha256,
    generate_challenge,
    main,
    multi_seed_summary,
    parse_seeds,
    robust_z_detect,
)

# Changing the generator changes this digest; bump CHALLENGE_VERSION and update both together.
SEED_7_FRAME_SHA256 = "e203ae8c6c1a92d2d56306571be689b49b99807ec9bd6782b7cb38f5d7e26f1c"
TOTAL_FRAMES = len(SCENARIOS) * EPISODES_PER_SCENARIO * FRAMES_PER_EPISODE


def _episodes(challenge, scenario):
    return [e for e in challenge.episodes if e.scenario == scenario]


class GeneratorTests(unittest.TestCase):
    def test_version_pin_on_canonical_frame_values(self):
        self.assertEqual(CHALLENGE_VERSION, "2.0.0")
        self.assertEqual(frame_values_sha256(generate_challenge(7)), SEED_7_FRAME_SHA256)

    def test_deterministic_replay_and_seed_effect(self):
        first = generate_challenge(1234)
        self.assertEqual(first, generate_challenge(1234))
        self.assertNotEqual(first, generate_challenge(1235))
        self.assertEqual(first.version, CHALLENGE_VERSION)

    def test_dimensions_scenarios_and_contiguous_episodes(self):
        challenge = generate_challenge(9)
        self.assertEqual(TOTAL_FRAMES, 900)
        self.assertEqual(len(challenge.frames), TOTAL_FRAMES)
        self.assertEqual([f.index for f in challenge.frames], list(range(TOTAL_FRAMES)))
        self.assertTrue(all(len(f.values) == CHANNEL_COUNT for f in challenge.frames))
        self.assertEqual(len(challenge.episodes), len(SCENARIOS) * EPISODES_PER_SCENARIO)
        for spec in SCENARIOS:
            self.assertEqual(len(_episodes(challenge, spec.name)), EPISODES_PER_SCENARIO)
        self.assertLessEqual(
            {"stable-noise", "localized-burst", "block-drift", "global-shock"},
            {spec.name for spec in SCENARIOS},
        )
        previous_end = -1
        for episode in challenge.episodes:
            self.assertEqual(episode.start_frame, previous_end + 1)
            self.assertEqual(episode.end_frame - episode.start_frame + 1, FRAMES_PER_EPISODE)
            previous_end = episode.end_frame
            first = challenge.frames[episode.start_frame]
            self.assertTrue(first.episode_start)
            self.assertEqual(first.episode_offset, 0)
            for offset in range(1, FRAMES_PER_EPISODE):
                frame = challenge.frames[episode.start_frame + offset]
                self.assertFalse(frame.episode_start)
                self.assertEqual(frame.episode_offset, offset)
                self.assertEqual(frame.episode_id, episode.episode_id)

    def test_generator_follows_scenario_spec(self):
        challenge = generate_challenge(42)
        for episode in challenge.episodes:
            spec = stc.SCENARIO_BY_NAME[episode.scenario]
            self.assertEqual(dict(episode.parameters), {**spec.parameters(), **{
                k: v for k, v in episode.parameters.items() if k in ("top_row", "left_column")}})
            if not spec.has_event:
                self.assertIsNone(episode.event_start_frame)
                self.assertEqual(episode.injected_channels, ())
                continue
            self.assertEqual(episode.event_start_frame - episode.start_frame, spec.event_offset_frames)
            self.assertEqual(episode.event_end_frame - episode.event_start_frame + 1, spec.event_length_frames)
            expected = {"block": spec.block_rows * spec.block_columns, "all": CHANNEL_COUNT}[spec.region]
            self.assertEqual(len(episode.injected_channels), expected)
            if spec.region == "block":
                top, left = episode.parameters["top_row"], episode.parameters["left_column"]
                self.assertEqual(
                    episode.injected_channels,
                    stc._rectangle_channels(top, left, spec.block_rows, spec.block_columns),
                )
        step = stc.SCENARIO_BY_NAME["baseline-step"]
        self.assertEqual(step.event_offset_frames + step.event_length_frames, FRAMES_PER_EPISODE)

    def test_locations_vary_across_episodes(self):
        challenge = generate_challenge(7)
        for name in ("localized-burst", "block-drift", "weak-burst"):
            locations = {e.injected_channels for e in _episodes(challenge, name)}
            self.assertGreater(len(locations), 1, name)

    def test_ground_truth_aligned_with_event_intervals(self):
        challenge = generate_challenge(42)
        episodes = {e.episode_id: e for e in challenge.episodes}
        for frame in challenge.frames:
            episode = episodes[frame.episode_id]
            active = episode.has_event and episode.event_start_frame <= frame.index <= episode.event_end_frame
            self.assertEqual(frame.event_id, episode.episode_id if active else None)
            self.assertEqual(frame.injected_channels, episode.injected_channels if active else ())

    def test_injections_match_amplitudes(self):
        # The same seed without injection is not available, so check means over the region.
        challenge = generate_challenge(11)
        shock = _episodes(challenge, "global-shock")[0]
        frame = challenge.frames[shock.event_start_frame]
        self.assertAlmostEqual(sum(frame.values) / CHANNEL_COUNT, 6.0, delta=0.3)
        negative = _episodes(challenge, "negative-shift")
        values = [challenge.frames[i].values[c] for e in negative
                  for i in range(e.event_start_frame, e.event_end_frame + 1) for c in e.injected_channels]
        self.assertAlmostEqual(sum(values) / len(values), -4.0, delta=0.3)
        variance = _episodes(challenge, "variance-change")
        values = [challenge.frames[i].values[c] for e in variance
                  for i in range(e.event_start_frame, e.event_end_frame + 1) for c in e.injected_channels]
        rms = (sum(v * v for v in values) / len(values)) ** 0.5
        self.assertAlmostEqual(rms, 3.0, delta=0.4)

    def test_invalid_seed(self):
        with self.assertRaises(TypeError):
            generate_challenge(True)
        with self.assertRaises(TypeError):
            generate_challenge(1.5)


class DetectorAndMetricTests(unittest.TestCase):
    def test_reference_detectors_discriminate(self):
        challenge = generate_challenge(7)
        baseline = evaluate(challenge, baseline_detect(challenge))
        robust = evaluate(challenge, robust_z_detect(challenge))
        self.assertEqual(baseline["per_scenario"]["global-shock"]["event_detection_recall"], 1.0)
        self.assertEqual(robust["per_scenario"]["global-shock"]["event_detection_recall"], 0.0)
        self.assertLess(baseline["event_detection_recall"], 1.0)
        self.assertLess(baseline["per_scenario"]["weak-burst"]["event_detection_recall"], 1.0)
        self.assertLess(baseline["per_scenario"]["slow-ramp"]["channel_localization_accuracy"], 0.5)
        local = _episodes(challenge, "localized-burst")[0]
        detection = baseline_detect(challenge)[local.event_start_frame]
        self.assertIn(local.injected_channels[0], detection.channels)
        self.assertLess(len(detection.channels), 32)

    def test_metrics_have_transparent_denominators(self):
        challenge = generate_challenge(3)
        target = _episodes(challenge, "localized-burst")[0]
        predictions = []
        for frame in challenge.frames:
            if frame.event_id == target.episode_id and frame.index > target.event_start_frame:
                predictions.append(Detection(frame.index, True, frame.injected_channels))
            elif frame.index == 0:
                predictions.append(Detection(frame.index, True, (5,)))  # clean-frame false alarm
            else:
                predictions.append(Detection(frame.index, False, ()))
        metrics = evaluate(challenge, predictions)
        fault = (len(SCENARIOS) - 1) * EPISODES_PER_SCENARIO
        injected_frames = sum(1 for f in challenge.frames if f.injected_channels)
        clean = TOTAL_FRAMES - injected_frames
        pairs = sum(len(f.injected_channels) for f in challenge.frames)
        self.assertEqual(metrics["fault_episodes"], fault)
        self.assertEqual(metrics["detected_fault_episodes"], 1)
        self.assertEqual(metrics["event_detection_recall"], 1 / fault)
        self.assertEqual(metrics["clean_frames"], clean)
        self.assertEqual(metrics["false_alarms_per_clean_frame"], 1 / clean)
        self.assertEqual(metrics["channel_localization_accuracy"], 2 / pairs)
        self.assertEqual(metrics["mean_detection_latency_frames"], 1.0)
        row = metrics["per_scenario"]["localized-burst"]
        self.assertEqual(row["event_detection_recall"], 1 / EPISODES_PER_SCENARIO)
        self.assertEqual(row["channel_localization_accuracy"], 2 / (3 * EPISODES_PER_SCENARIO))
        self.assertIsNone(metrics["per_scenario"]["stable-noise"]["event_detection_recall"])
        latencies = {r["episode_id"]: r["detection_latency_frames"] for r in metrics["per_episode"]}
        self.assertEqual(latencies[target.episode_id], 1)
        self.assertEqual(sum(v is not None for v in latencies.values()), 1)

    def test_invalid_thresholds(self):
        challenge = generate_challenge(0)
        for detector in (baseline_detect, robust_z_detect):
            for threshold in (0, -1, float("nan"), float("inf")):
                with self.subTest(detector=detector.__name__, threshold=threshold):
                    with self.assertRaises(ValueError):
                        detector(challenge, threshold)
            with self.assertRaises(TypeError):
                detector(challenge, True)

    def test_invalid_or_incomplete_predictions(self):
        challenge = generate_challenge(2)
        good = list(baseline_detect(challenge))
        last = len(good) - 1
        with self.assertRaises(ValueError):
            evaluate(challenge, good[:-1])
        with self.assertRaises(ValueError):
            evaluate(challenge, good + [good[0]])
        with self.assertRaises(ValueError):
            evaluate(challenge, good[:-1] + [Detection(last, True, (512,))])
        with self.assertRaises(TypeError):
            evaluate(challenge, [Detection(0, 1, ())] + good[1:])
        with self.assertRaises(ValueError):
            evaluate(challenge, [Detection(0, True, (1, 1))] + good[1:])

    def test_alarm_false_with_channels_is_rejected(self):
        challenge = generate_challenge(2)
        good = list(baseline_detect(challenge))
        with self.assertRaisesRegex(ValueError, "alarm=False"):
            evaluate(challenge, [Detection(0, False, (3,))] + good[1:])
        # An alarm without localization is allowed.
        evaluate(challenge, [Detection(0, True, ())] + good[1:])


class ReportAndCliTests(unittest.TestCase):
    def test_report_is_machine_readable(self):
        challenge = generate_challenge(101)
        report = build_report(challenge, baseline_detect(challenge), "test-baseline")
        decoded = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(decoded["seed"], 101)
        self.assertEqual(decoded["challenge_version"], CHALLENGE_VERSION)
        self.assertEqual(decoded["dimensions"]["channels"], 512)
        self.assertEqual(len(decoded["episodes"]), len(SCENARIOS) * EPISODES_PER_SCENARIO)
        self.assertIn("physical instruments", decoded["synthetic_only_notice"])
        metrics = decoded["detectors"]["test-baseline"]
        self.assertIn("channel_localization_accuracy", metrics)
        self.assertEqual(set(metrics["per_scenario"]), {spec.name for spec in SCENARIOS})
        self.assertNotIn("group", json.dumps(decoded))

    def test_invalid_detector_name(self):
        challenge = generate_challenge(4)
        with self.assertRaises(ValueError):
            build_report(challenge, baseline_detect(challenge), "  ")
        with self.assertRaises(ValueError):
            build_report(challenge, {})

    def test_parse_seeds(self):
        self.assertEqual(parse_seeds("1-10"), list(range(1, 11)))
        self.assertEqual(parse_seeds("3,1,8"), [3, 1, 8])
        self.assertEqual(parse_seeds("1-3,8"), [1, 2, 3, 8])
        for bad in ("", "a", "5-2", "1,1", "1,,2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_seeds(bad)

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue()

    def test_cli_single_seed_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "report.json"
            code, out = self._run("--seed", "7", "--report", str(path))
            self.assertEqual(code, 0)
            self.assertIn("Wrote", out)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["seed"], 7)
        self.assertEqual(report["frame_values_sha256"], SEED_7_FRAME_SHA256)
        self.assertEqual(len(report["detectors"]), 2)

    def test_cli_multi_seed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            code, out = self._run("--seeds", "1-3", "--report", str(path))
            self.assertEqual(code, 0)
            summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["seeds"], [1, 2, 3])
        self.assertEqual(set(summary["detectors"]), {"absolute-threshold", "robust-z"})
        stats = summary["detectors"]["absolute-threshold"]["overall"]["event_detection_recall"]
        self.assertLessEqual(stats["min"], stats["mean"])
        self.assertLessEqual(stats["mean"], stats["max"])
        self.assertEqual(summary, json.loads(json.dumps(multi_seed_summary([1, 2, 3]))))

    def test_cli_rejects_bad_arguments(self):
        for argv in (["--threshold", "0"], ["--seeds", "x"], ["--seed", "1", "--seeds", "1-2"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as ctx:
                    self._run(*argv, "--report", "/nonexistent-never-written.json")
                self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
