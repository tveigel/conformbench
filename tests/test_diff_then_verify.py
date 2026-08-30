from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from conformbench import benchmark
from conformbench.systems import diff_then_verify


SCHEMA = {
    "questions": [
        {
            "id": "name",
            "structure_type": "regular",
            "type": "text",
            "question_text": "Name?",
            "gold_standard": "Record the stated name.",
        },
        {
            "id": "city",
            "structure_type": "regular",
            "type": "text",
            "question_text": "City?",
            "gold_standard": "Record the stated city.",
        },
    ]
}


def turn() -> SimpleNamespace:
    return SimpleNamespace(
        schema=deepcopy(SCHEMA),
        prior_state={"name": "Old", "city": None},
        visible_history=[{"role": "user", "content": "My name is Alice."}],
        current_utterance="I live in Berlin.",
        metadata={"model_id": "gpt-5.4-mini"},
    )


def diff_result() -> dict:
    return {
        "resulting_state": {"name": "Alice", "city": None},
        "agent_response": {
            "raw_response": "Updated name.",
            "tool_updates": [
                {
                    "updates": {"name": "Alice"},
                    "result": {"status": "ok", "applied_update_count": 1},
                }
            ],
            "model_calls": [{"phase": "flatagent_generation", "call_index": 1}],
        },
        "provenance": {
            "generation": {
                "agent": "FlatAgent",
                "model": {
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "requested_model": "gpt-5.4-mini",
                },
            }
        },
    }


class FakeModel:
    def __init__(self, response: AIMessage):
        self.response = response
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


def fake_model_config() -> dict:
    return {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "requested_model": "gpt-5.4-mini",
        "max_tokens": 32000,
        "timeout": 300,
        "temperature": None,
    }


class DiffThenVerifyTests(unittest.TestCase):
    def test_normalizes_repeat_row_objects_and_idempotent_deletes(self):
        shape = {
            "repeat_groups": {
                "current_medications": {"medication_name", "dose"},
                "allergies": {"reaction"},
            }
        }
        normalized, notes = diff_then_verify._normalize_corrective_updates(
            updates={
                "current_medications[0]": {
                    "medication_name": "paracetamol",
                    "dose": "1 gram",
                },
                "allergies[1]": "__DELETE_INSTANCE__",
            },
            state={
                "current_medications": [{"medication_name": None, "dose": None}],
                "allergies": [{"reaction": "hives"}],
            },
            shape=shape,
        )
        self.assertEqual(
            normalized,
            {
                "current_medications[0].medication_name": "paracetamol",
                "current_medications[0].dose": "1 gram",
            },
        )
        self.assertEqual(
            {note["action"] for note in notes},
            {"expanded_repeat_row_object", "ignored_idempotent_delete"},
        )

    def test_normalizes_repeat_group_patch_with_delete_sentinel(self):
        normalized, notes = diff_then_verify._normalize_corrective_updates(
            updates={
                "vehicles": [
                    {"role": "claimant", "speed": 0},
                    {"role": "other", "speed": 5},
                    {"role": "__DELETE_INSTANCE__"},
                ]
            },
            state={
                "vehicles": [
                    {"role": "claimant", "speed": None},
                    {"role": "other", "speed": None},
                ]
            },
            shape={"repeat_groups": {"vehicles": {"role", "speed"}}},
        )
        self.assertEqual(
            normalized,
            {
                "vehicles[0].role": "claimant",
                "vehicles[0].speed": 0,
                "vehicles[1].role": "other",
                "vehicles[1].speed": 5,
            },
        )
        self.assertIn("expanded_repeat_group_row_patches", {n["action"] for n in notes})
        self.assertIn("ignored_idempotent_delete", {n["action"] for n in notes})

    def test_ignores_unknown_verifier_paths_but_keeps_valid_updates(self):
        normalized, notes = diff_then_verify._normalize_corrective_updates(
            updates={
                ">": "__DELETE__",
                "city": "Berlin",
                "vehicles[0].speed": 30,
                "vehicles[0].unknown": "junk",
            },
            state={"city": None, "vehicles": [{"speed": None}]},
            shape={
                "bare_fields": {"city"},
                "repeat_groups": {"vehicles": {"speed"}},
            },
        )
        self.assertEqual(
            normalized,
            {"city": "Berlin", "vehicles[0].speed": 30},
        )
        self.assertEqual(
            [n["path"] for n in notes if n["action"] == "ignored_unknown_update_path"],
            [">", "vehicles[0].unknown"],
        )

    def test_verifier_applies_sparse_patch_to_materialized_candidate(self):
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_questionnaire_answers",
                    "args": {"updates": {"city": "Berlin"}},
                    "id": "verify-1",
                    "type": "tool_call",
                }
            ],
        )
        fake_model = FakeModel(response)
        stage_one = AsyncMock(return_value=diff_result())

        with (
            patch.object(diff_then_verify.flatagent, "_solve_async", stage_one),
            patch.object(
                diff_then_verify,
                "get_chat_model_with_config",
                return_value=(fake_model, fake_model_config()),
            ),
        ):
            result = asyncio.run(diff_then_verify._solve_async(turn()))

        self.assertEqual(result["resulting_state"], {"name": "Alice", "city": "Berlin"})
        self.assertEqual(
            result["agent_response"]["stages"]["diff"]["candidate_state"],
            {"name": "Alice", "city": None},
        )
        self.assertEqual(result["agent_response"]["stages"]["verify"]["model_call_count"], 1)
        self.assertEqual(result["agent_response"]["stages"]["verify"]["emitted_tool_call_count"], 1)
        self.assertEqual(len(result["agent_response"]["model_calls"]), 2)
        self.assertTrue(result["provenance"]["generation"]["same_generator_both_stages"])
        self.assertGreaterEqual(
            result["agent_response"]["timing"]["total_generation_seconds"],
            result["agent_response"]["timing"]["verification_stage_seconds"],
        )

        prompt = fake_model.messages[1].content
        self.assertIn("=== ORIGINAL PRIOR RECORD STATE ===", prompt)
        self.assertIn('"name": "Old"', prompt)
        self.assertIn("=== CANDIDATE POST-TURN RECORD TO VERIFY ===", prompt)
        self.assertIn('"name": "Alice"', prompt)
        self.assertNotIn("gold_resulting_state", prompt)

    def test_no_verifier_tool_call_preserves_diff_candidate(self):
        fake_model = FakeModel(AIMessage(content="Candidate is supported."))
        with (
            patch.object(
                diff_then_verify.flatagent,
                "_solve_async",
                AsyncMock(return_value=diff_result()),
            ),
            patch.object(
                diff_then_verify,
                "get_chat_model_with_config",
                return_value=(fake_model, fake_model_config()),
            ),
        ):
            result = asyncio.run(diff_then_verify._solve_async(turn()))

        self.assertEqual(result["resulting_state"], {"name": "Alice", "city": None})
        self.assertEqual(result["agent_response"]["stages"]["verify"]["emitted_tool_call_count"], 0)
        verify_updates = [
            update
            for update in result["agent_response"]["tool_updates"]
            if update.get("stage") == "verify"
        ]
        self.assertEqual(verify_updates, [])

    def test_public_runner_writes_auditable_two_stage_artifact(self):
        fake_model = FakeModel(AIMessage(content="Candidate is supported."))
        raw_item = {
            "item_id": "integration-1",
            "questionnaire_id": "form",
            "prior_state": {"name": "Old", "city": None},
            "visible_history": [{"role": "user", "content": "My name is Alice."}],
            "current_utterance": "I live in Berlin.",
            "gold_resulting_state": {"name": "Alice", "city": None},
            "model_id": "gpt-5.4-mini",
        }

        with (
            TemporaryDirectory() as temp,
            patch.object(
                diff_then_verify.flatagent,
                "_solve_async",
                AsyncMock(return_value=diff_result()),
            ),
            patch.object(
                diff_then_verify,
                "get_chat_model_with_config",
                return_value=(fake_model, fake_model_config()),
            ),
            patch.object(benchmark, "load_questionnaire", return_value=deepcopy(SCHEMA)),
        ):
            result = benchmark.run(
                items=[raw_item],
                solver=diff_then_verify.solve,
                output_dir=temp,
                run_id="integration",
                score=False,
            )
            artifact_path = next(Path(temp).rglob("turn_result.json"))
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(result["item_count"], 1)
        self.assertEqual(artifact["answers_after"], {"name": "Alice", "city": None})
        self.assertIn("timing", artifact["agent_response"])
        self.assertEqual(
            artifact["provenance"]["generation"]["architecture"],
            "sparse_diff_then_single_call_verification",
        )


if __name__ == "__main__":
    unittest.main()
