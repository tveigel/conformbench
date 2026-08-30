from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import summarize_diff_then_verify as summarize


def utterance(key: str, *, exact: bool, missed: int, unsupported: int) -> dict:
    return {
        "questionnaire": "form",
        "scenario_id": key,
        "state": "state_1",
        "utterance_id": "u1",
        "metric_views": {
            "whole_record_exact_match": {"exact_match": exact},
        },
        "diagnostics": {
            "counts": {
                "missed_supported_update": missed,
                "unsupported_commit": unsupported,
            }
        },
    }


class SummarizeDiffThenVerifyTests(unittest.TestCase):
    def test_paired_bootstrap_aligns_items_by_key(self):
        result = summarize.paired_exact_bootstrap(
            {"a": 1, "b": 1, "c": 0},
            {"c": 0, "b": 0, "a": 1},
            iterations=500,
            seed=7,
        )
        self.assertAlmostEqual(result["difference"], 1 / 3)
        self.assertLessEqual(result["ci_low"], result["difference"])
        self.assertGreaterEqual(result["ci_high"], result["difference"])

    def test_aggregate_metrics_uses_diagnostic_counts(self):
        summary = {
            "aggregate": {
                "metric_views": {
                    "whole_record_exact_match": {
                        "item_count": 2,
                        "exact_match_count": 1,
                        "exact_match_rate": 0.5,
                    },
                    "changed_fields": {
                        "strict": {"f1": 0.8, "precision": 0.9, "recall": 0.72}
                    },
                }
            },
            "utterances": [
                utterance("a", exact=True, missed=2, unsupported=1),
                utterance("b", exact=False, missed=3, unsupported=4),
            ],
        }
        metrics = summarize.aggregate_metrics(summary)
        self.assertEqual(metrics["exact_record_count"], 1)
        self.assertEqual(metrics["missed_supported_updates"], 5)
        self.assertEqual(metrics["unsupported_commitments"], 5)
        self.assertEqual(metrics["changed_field_f1"], 0.8)

    def test_latency_metrics_reads_solver_only_timings(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for index, values in enumerate(((1.0, 2.0, 3.0), (3.0, 4.0, 7.0))):
                path = root / f"item-{index}" / "turn_result.json"
                path.parent.mkdir(parents=True)
                payload = {
                    "agent_response": {
                        "timing": {
                            "diff_stage_seconds": values[0],
                            "verification_stage_seconds": values[1],
                            "total_generation_seconds": values[2],
                        },
                        "stages": {
                            "verify": {
                                "tool_updates": [] if index == 0 else [
                                    {
                                        "updates": {"city": "Berlin"},
                                        "result": {"status": "ok"},
                                    }
                                ]
                            }
                        },
                    }
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            metrics = summarize.latency_metrics(root)

        self.assertEqual(metrics["timed_item_count"], 2)
        self.assertEqual(metrics["median_diff_stage_seconds"], 2.0)
        self.assertEqual(metrics["median_verification_stage_seconds"], 3.0)
        self.assertEqual(metrics["median_total_generation_seconds"], 5.0)
        self.assertEqual(metrics["verifier_patch_item_count"], 1)


if __name__ == "__main__":
    unittest.main()

