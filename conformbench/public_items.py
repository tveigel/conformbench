"""Load public development items from ``data/items/public``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .items import PUBLIC_ITEMS_DIR, normalize_item


def _ground_truth_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("ground_truth.json"))


def load_public_items() -> list[dict[str, Any]]:
    """Load the packaged public development item packets."""

    return [normalize_item(packet) for packet in load_public_item_packets()]


def load_public_item_packets() -> list[dict[str, Any]]:
    """Load raw packaged public development item packets."""

    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _ground_truth_files(PUBLIC_ITEMS_DIR)
    ]


def load_public_item_packet(item_id: str) -> dict[str, Any]:
    """Load one raw public development item packet by id."""

    for packet in load_public_item_packets():
        if normalize_item(packet)["item_id"] == item_id:
            return packet
    raise KeyError(f"Unknown public item: {item_id!r}")


def list_public_item_ids() -> list[str]:
    """Return public development item ids."""

    return [item["item_id"] for item in load_public_items()]
