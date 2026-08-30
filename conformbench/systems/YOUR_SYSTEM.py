"""Implement your ConFormBench system here.

Run after implementing:

    python -m conformbench run --items public --solver conformbench.systems.YOUR_SYSTEM:solve

`solve(turn)` receives only public benchmark input:

    turn.item_id
    turn.questionnaire_id
    turn.schema
    turn.prior_state
    turn.visible_history
    turn.current_utterance
    turn.metadata

Return one dict: the full post-turn record state.

The runner checks the shape only:
- every top-level schema field and repeat-group id is present exactly once;
- repeat groups and table fields are lists, with any row count allowed;
- every repeat row contains exactly that repeat group's child field ids;
- every table row contains exactly that table's column ids.

The runner does not repair, fill, coerce, sort, or apply operations to your
output. The state you return is the state that gets evaluated.
"""

from __future__ import annotations

from typing import Any


def solve(turn: Any) -> dict[str, Any]:
    """Return the complete resulting state for one benchmark turn."""

    raise NotImplementedError("Return your system's complete resulting state here.")
