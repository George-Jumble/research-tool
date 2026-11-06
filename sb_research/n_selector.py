"""
Sample-size (n) selector policy.

No live action wires a sample-size parameter: loader actions fetch fixed targets and every
analysis action is scored by the Superblock residual engine on the full cached dataset.
"""

from __future__ import annotations


def should_offer_n_selector(module_tag: str, action: str) -> bool:
    """Central switch for when to show the interactive n selector tool-wide."""
    _ = (module_tag, action)
    return False


# =============================================================================
# BLOCK: Entry-form helpers after n is chosen in the menu loop
# =============================================================================


def strip_auto_filled_fields(fields: list[dict], params: dict[str, str]) -> list[dict]:
    """Drop form fields already set by the automatic n selector."""
    skip: set[str] = set()
    if params.get("n_points"):
        skip.add("n_points")
    if params.get("batch_limit"):
        skip.update({"batch_limit", "force_rescan"})
    if not skip:
        return fields
    return [field for field in fields if field.get("key") not in skip]


def seed_field_defaults_from_params(
    fields: list[dict],
    params: dict[str, str],
) -> list[dict]:
    """Let the entry form reflect values already chosen by the n selector."""
    seeded: list[dict] = []
    for field in fields:
        key = field.get("key")
        if key and key in params:
            seeded.append({**field, "default": params[key]})
        else:
            seeded.append(field)
    return seeded
