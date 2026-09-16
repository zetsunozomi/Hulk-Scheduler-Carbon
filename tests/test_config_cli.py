from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest

from carbon.__main__ import main
from carbon.common import ContractError, digest, json_text, timestamp
from carbon.config import Bundle
from carbon.runner import run_fixed
from carbon.trace import load_trace
from tests.helpers import ROOT, bundle


class ConfigTests(unittest.TestCase):
    def config_variant(self, change):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        c = deepcopy(bundle().raw)
        c["root"] = str(ROOT)
        change(c)
        path = Path(temp.name) / "config.json"
        path.write_text(json_text(c))
        return path

    def test_hashes_and_split_overlap_rejected(self):
        path = self.config_variant(lambda c: c["trace"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ContractError, "SHA256"):
            Bundle(path)
        path = self.config_variant(lambda c: c["splits"]["validation"].__setitem__(0, c["splits"]["train"][0]))
        with self.assertRaisesRegex(ContractError, "disjoint"):
            Bundle(path)

    def test_unknown_key_and_unresolved_template_rejected(self):
        path = self.config_variant(lambda c: c.update(typographical_error=1))
        with self.assertRaisesRegex(ContractError, "unknown keys"):
            Bundle(path)
        with self.assertRaises(ContractError):
            Bundle(ROOT / "configs/cluster.template.json")

    def test_synthetic_profiles_cannot_be_research(self):
        path = self.config_variant(lambda c: c.update(purpose="research"))
        with self.assertRaisesRegex(ContractError, "measured/published"):
            Bundle(path)

    def test_timezone_ambiguity_and_nonexistent_time(self):
        with self.assertRaisesRegex(ContractError, "Ambiguous"):
            timestamp("2024-11-03T01:30:00", "America/Chicago")
        with self.assertRaisesRegex(ContractError, "Nonexistent"):
            timestamp("2024-03-10T02:30:00", "America/Chicago")
        a = timestamp("2024-11-03T01:30:00", "America/Chicago", 0)
        b = timestamp("2024-11-03T01:30:00", "America/Chicago", 1)
        self.assertEqual(b-a, timedelta(hours=1))

    def test_trace_has_explicit_cleaning_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trace.csv"
            header = "JobID,NNodes,Submit,Start,End,TimelimitR\n"
            row = "x,4,2024-01-01T00:00:00Z,2024-01-01T00:00:00Z,2024-01-01T00:02:00Z,1\n"
            p.write_text(header+row)
            config = {"timezone":"UTC", "zero_duration_policy":"drop", "overrun_policy":"error"}
            with self.assertRaisesRegex(ContractError, "exceeds requested"):
                load_trace(p, config)
            config["overrun_policy"] = "clip"
            jobs, report = load_trace(p, config)
            self.assertEqual(jobs[0].runtime, timedelta(minutes=1))
            self.assertEqual(report["counts"]["duration_over_request_rows"], 1)
            p.write_text(header+row+row)
            with self.assertRaisesRegex(ContractError, "Duplicate"):
                load_trace(p, config)

    def test_cli_validate_and_paired_fixed_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            config = str(ROOT / "configs/synthetic.json")
            self.assertEqual(main(["validate", "--config", config]), 0)
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            for output in (a, b):
                self.assertEqual(main(["run-fixed", "--config", config, "--output", str(output)]), 0)
            self.assertEqual((a/"chunks.jsonl").read_bytes(), (b/"chunks.jsonl").read_bytes())
            self.assertEqual((a/"episodes.jsonl").read_bytes(), (b/"episodes.jsonl").read_bytes())
            episodes = [json.loads(line) for line in (a/"episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(episodes), 12)
            self.assertTrue(all(e["purpose"] == "synthetic" and e["completed_updates"] == 100 for e in episodes))
            before = (a/"episodes.jsonl").read_bytes()
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run-fixed", "--config", config, "--output", str(a)]), 2)
            self.assertEqual(before, (a/"episodes.jsonl").read_bytes())

    def test_sharding_preserves_cohort_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            results = []
            for i in range(2):
                try:
                    m = run_fixed(bundle(), Path(tmp)/str(i), [4, 8], shard_index=i, shard_count=2)
                except ContractError as exc:
                    self.assertIn("empty", str(exc))
                    continue
                results.extend(m["selected_episode_ids"])
                self.assertEqual(m["completed_episode_methods"], 2*len(m["selected_episode_ids"]))
            self.assertEqual(sorted(results), sorted(e.episode_id for e in bundle().episodes))

    def test_failure_is_recorded_and_run_cannot_appear_complete(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output=Path(tmp)/"failed"
            with patch("carbon.environment.Environment.step", side_effect=ContractError("injected failure")):
                with self.assertRaises(ContractError):
                    run_fixed(bundle(), output, [4])
            self.assertEqual(json.loads((output/"manifest.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((output/"failure.json").read_text())["episode_id"], bundle().episodes[0].episode_id)
