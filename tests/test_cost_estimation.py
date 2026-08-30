from __future__ import annotations

import unittest

from scripts import run_architecture_comparison as runner


class CostEstimationTests(unittest.TestCase):
    def test_estimates_mini_cost_with_cached_input(self):
        payload = {
            "model_config": {"model": "gpt-5.4-mini"},
            "usage": {
                "prompt_tokens": 4214,
                "completion_tokens": 18,
                "total_tokens": 4232,
                "raw": {
                    "input_token_details": {"cache_read": 3712},
                },
            },
        }
        expected = (502 * 0.75 + 3712 * 0.075 + 18 * 4.50) / 1_000_000
        self.assertAlmostEqual(runner.estimate_openai_cost_usd(payload), expected)

    def test_estimates_evaluator_cost_from_full_prompt_trace(self):
        payload = {
            "model_config": {"model": "gpt-5.4", "reasoning_effort": "medium"},
            "response": {
                "response_metadata": {"model_name": "gpt-5.4-2026-03-05"},
                "usage_metadata": {
                    "input_tokens": 2258,
                    "output_tokens": 411,
                    "total_tokens": 2669,
                    "input_token_details": {"cache_read": 0},
                },
            },
        }
        expected = (2258 * 2.50 + 411 * 15.00) / 1_000_000
        self.assertAlmostEqual(runner.estimate_openai_cost_usd(payload), expected)

    def test_cost_bucket_uses_estimate_when_provider_cost_is_absent(self):
        payload = {
            "model_config": {"model": "gpt-5.4-mini"},
            "usage_metadata": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "input_token_details": {"cache_read": 200},
            },
        }
        bucket = runner.empty_cost_bucket()
        runner.add_cost_record(bucket, payload)
        self.assertEqual(bucket["estimated_cost_calls"], 1)
        self.assertEqual(bucket["provider_cost_calls"], 0)
        self.assertEqual(bucket["missing_cost_calls"], 0)
        self.assertEqual(bucket["cached_prompt_tokens"], 200)
        self.assertGreater(bucket["cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()

