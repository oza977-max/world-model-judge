"""WD-3 dt-alignment checking function (worlds spec ADR-W1).

In plain words: the ground truth and every model must agree on how
big one "step" is, or every chart in the project would be silently
comparing apples to oranges while looking plausible. This is the one
reusable place that check lives, so the negative case (TC-WD3-02) can
demonstrate it genuinely fails on a mismatch, not just narrate that it
would.

Test-support infrastructure, not a product deliverable.
"""

from __future__ import annotations


class DtMismatchError(ValueError):
    """Raised when a model's step size doesn't match the world's."""


def assert_dt_alignment(world_dt: float, model_dt: float) -> None:
    """Raise DtMismatchError unless world_dt == model_dt exactly.

    Worlds spec ADR-W1 (WD-3): the harness's model-rollout loop must
    advance a model exactly once per ground-truth step — this is the
    dt half of that guarantee.
    """
    if world_dt != model_dt:
        raise DtMismatchError(
            f"dt mismatch: world dt={world_dt!r}, model dt={model_dt!r} "
            f"(worlds spec ADR-W1, WD-3)"
        )
