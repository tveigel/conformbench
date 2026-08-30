from __future__ import annotations

import unittest

from conformbench.evaluator.metric_views import compute_turn_metric_views


class TransitionAccuracyTests(unittest.TestCase):
    def test_requires_correct_resulting_value_not_only_operation_class(self):
        gt = {
            "prior_state": {
                "set_field": None,
                "change_field": "old",
                "clear_field": "remove me",
                "preserve_field": "keep me",
            },
            "gold_resulting_state": {
                "set_field": "gold set",
                "change_field": "gold change",
                "clear_field": None,
                "preserve_field": "keep me",
            },
            "fields": {
                qid: {"expected": expected, "present_in_utterance": True}
                for qid, expected in {
                    "set_field": "gold set",
                    "change_field": "gold change",
                    "clear_field": None,
                    "preserve_field": "keep me",
                }.items()
            },
        }
        candidate_state = {
            "set_field": "wrong set",
            "change_field": "wrong change",
            "clear_field": None,
            "preserve_field": "keep me",
        }
        field_results = {
            "set_field": {"correctness": "incorrect"},
            "change_field": {"correctness": "incorrect"},
            "clear_field": {"correctness": "correct"},
            "preserve_field": {"correctness": "correct"},
        }

        views = compute_turn_metric_views(
            gt=gt,
            candidate_state=candidate_state,
            field_results=field_results,
            alignment_log=[],
            diagnostics={},
        )

        transition = views["transition_accuracy"]
        self.assertEqual(transition["correct"], 2)
        self.assertEqual(transition["correct"], views["all_fields"]["correct"])
        self.assertEqual(transition["by_transition"]["set"]["correct"], 0)
        self.assertEqual(transition["by_transition"]["change"]["correct"], 0)
        self.assertEqual(transition["by_transition"]["clear"]["correct"], 1)
        self.assertEqual(transition["by_transition"]["preserve"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
