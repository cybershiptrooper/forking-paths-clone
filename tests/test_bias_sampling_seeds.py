"""CPU-only regression checks for independent rollout seed allocation.

Run: python -m unittest discover -s tests -p test_bias_sampling_seeds.py -v
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from expts.prompt_bias_circuit_discovery import collect_bias_rollouts as collector


class SamplingSeedTests(unittest.TestCase):
    def test_historical_phase_overlap_is_removed(self):
        for screen_root, screen_n, confirm_root, confirm_n, old_overlap in (
            (42, 16, 7, 64, 16), (924, 8, 925, 48, 7),
        ):
            for scheme, expected in (("legacy", old_overlap), ("disjoint", 0)):
                with self.subTest(screen_root=screen_root, scheme=scheme):
                    s = collector.sampling_parent_seeds(screen_root, 1, screen_n, scheme)[0]
                    c = collector.sampling_parent_seeds(confirm_root, 1, confirm_n, scheme)[0]
                    self.assertEqual(len(set(range(s, s + screen_n)) &
                                         set(range(c, c + confirm_n))), expected)

    def test_largest_prompt_and_rollout_blocks_do_not_touch_next_phase(self):
        seeds = collector.sampling_parent_seeds(0, 100_000, 10_000)
        self.assertEqual(len(set(seeds)), 100_000)
        self.assertTrue(all(b - a >= 10_000 for a, b in zip(seeds, seeds[1:])))
        next_phase = collector.sampling_parent_seeds(1, 1, 10_000)[0]
        self.assertLess(seeds[-1] + 9_999, next_phase)

    def test_invalid_allocations_fail_before_generation(self):
        invalid = [(-1, 2, 16), (42, -1, 16), (42, 100_001, 16),
                   (42, 2, 0), (42, 2, 10_001), (2**63, 2, 16)]
        for args in invalid:
            for scheme in ("disjoint", "legacy"):
                with self.subTest(args=args, scheme=scheme), self.assertRaises(ValueError):
                    collector.sampling_parent_seeds(*args, scheme)
        with self.assertRaises(ValueError):
            collector.sampling_parent_seeds(42, 2, 16, "unknown")

    def test_collector_preserves_global_indices_across_shards_and_records_seeds(self):
        # Alternate control/intervention prompts across two files and two shards.
        prompts = [dict(uid=f"pair{i // 2}_{'control' if i % 2 == 0 else 'intervention'}",
                        prompt_token_ids=[100 + i], question="Q", all_letters=["A", "B"],
                        all_answers=["yes", "no"], dataset_type="test", setting="test",
                        profile_id=i // 2, axis="arm", value=i % 2) for i in range(6)]
        calls = []

        class FakeLLM:
            def __init__(self, **kwargs):
                self.config = kwargs

            def generate(self, inputs, params):
                calls.append((self.config, inputs, params))
                # Reversed completion order catches code that guesses child seeds
                # from list position instead of storing vLLM's completion index.
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    index=i, token_ids=[input_["prompt_token_ids"][0], i],
                    text="reasoning</think> A", finish_reason="stop")
                    for i in reversed(range(param.n))])
                    for input_, param in zip(inputs, params)]

        vllm = ModuleType("vllm")
        vllm.LLM = FakeLLM
        vllm.SamplingParams = SimpleNamespace
        answers = ModuleType("utils.answer_utils")
        answers.parse_answer = lambda llm, rows: [
            dict(clean_answer="A", raw_answer="A") for _ in rows]

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            files = [base / "first.json", base / "second.json"]
            for file, rows in zip(files, (prompts[:3], prompts[3:])):
                file.write_text(json.dumps(rows))

            def run(shard=0, n_shards=1, scheme=None):
                output, report = base / "raw.json", base / "report.json"
                argv = ["collect_bias_rollouts", "--prompts", *map(str, files),
                        "--output", str(output), "--report", str(report),
                        "--seed", "42", "--n_rollouts", "3", "--shard", str(shard),
                        "--n_shards", str(n_shards)]
                if scheme is not None:
                    argv.extend(["--seed_scheme", scheme])
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {
                    "vllm": vllm, "utils.answer_utils": answers,
                }), contextlib.redirect_stdout(io.StringIO()):
                    collector.main()
                return json.loads(output.read_text()), json.loads(report.read_text())

            whole, report = run()
            left, _ = run(0, 2)
            right, _ = run(1, 2)
            self.assertEqual(report["args"]["seed_scheme"], "disjoint")
            self.assertEqual(sorted(left + right, key=lambda r: r["global_prompt_index"]), whole)
            self.assertEqual([r["global_prompt_index"] for r in left], [0, 2, 4])
            self.assertEqual([r["global_prompt_index"] for r in right], [1, 3, 5])
            self.assertEqual(len({r["sampling_parent_seed"] for r in whole}), 6)
            children = [c["sampling_seed"] for r in whole for c in r["rollouts"]]
            self.assertEqual(len(set(children)), 18)
            for row in whole:
                self.assertEqual(row["sampling_seed_scheme"], "disjoint")
                self.assertEqual([c["sample_index"] for c in row["rollouts"]], [2, 1, 0])
                for child in row["rollouts"]:
                    self.assertEqual(child["sampling_seed"],
                                     row["sampling_parent_seed"] + child["sample_index"])
            for config, inputs, params in calls:
                self.assertEqual(config["seed"], 42)  # LLM initialization stays unchanged.
                self.assertIsInstance(params, list)
                self.assertEqual(len(params), len(inputs))
                for input_, param in zip(inputs, params):
                    global_index = input_["prompt_token_ids"][0] - 100
                    self.assertEqual(param.seed, whole[global_index]["sampling_parent_seed"])

            legacy, legacy_report = run(scheme="legacy")
            self.assertEqual(legacy_report["args"]["seed_scheme"], "legacy")
            self.assertEqual({r["sampling_parent_seed"] for r in legacy}, {42})
            self.assertEqual({c["sampling_seed"] for r in legacy for c in r["rollouts"]},
                             {42, 43, 44})


if __name__ == "__main__":
    unittest.main()
