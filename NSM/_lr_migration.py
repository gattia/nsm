"""
One-time migration help for the Aug 2026 ``LearningRateSchedule`` ``Target`` key.

Nothing here is permanent API. It exists only to explain the change to someone holding a
config written before it, and to hand them a corrected copy of their own entries.

DELETE THIS FILE once no config still in use predates the ``Target`` key. The only caller
is ``resolve_schedule_targets`` in ``NSM/utils.py``, which imports it lazily and needs a
plain one-line ValueError in its place.

Background: ``docs/KNOWN_ISSUES_HISTORY.md`` section 1.
"""

import json

from NSM.utils import LR_TARGET_KEY, LR_TARGET_LATENT, LR_TARGET_MODEL

# Which entry drove which group before Aug 2026, by optimizer family. Adam/AdamW went
# through adjust_learning_rate(), which mapped positionally against get_optimizer()'s
# [latent, model...] group order -- so entry 0 drove the latents. schedule_free_* skipped
# adjust_learning_rate() and kept get_optimizer()'s own assignment, where entry 0 drove
# the model. The two families therefore migrate to OPPOSITE annotations.
_HISTORICAL_TARGETS_ADAM = (LR_TARGET_LATENT, LR_TARGET_MODEL)
_HISTORICAL_TARGETS_SCHEDULE_FREE = (LR_TARGET_MODEL, LR_TARGET_LATENT)

_MESSAGE = """
Every 'LearningRateSchedule' entry must declare '{key}' ("{model}" or "{latent}").

{problem}

Entry order used to decide this, and from May 2023 to Aug 2026 a bug applied the two
entries swapped on every Adam/AdamW run. Configs written before and after the fix are
byte-identical while meaning opposite things, so the intent has to be stated.

To reproduce this run as it originally trained -- its optimizer is '{optimizer}', whose
historical mapping was entry 0 -> {hist_0}, entry 1 -> {hist_1}:

{annotated}
{caution}
For a new run, set '{key}' to whatever each entry should drive; order is ignored.
Full background: docs/KNOWN_ISSUES_HISTORY.md section 1.
""".strip()

_SCHEDULE_FREE_CAUTION = """
CAUTION: schedule_free_* kept get_optimizer()'s assignment -- the OPPOSITE of what an
Adam/AdamW run of the same file did -- and nothing ever decayed it. If these values came
from an Adam/AdamW config, this run applied the latent's rate to the model and vice versa,
held constant throughout, which is a plausible reason for it to have trained badly.
Reproducing it faithfully may not be what you want.
"""


def migration_error(schedule_specs, optimizer, problem):
    """Build the error shown when a config predates the ``Target`` key."""
    schedule_free = "schedule_free" in str(optimizer)
    hist = _HISTORICAL_TARGETS_SCHEDULE_FREE if schedule_free else _HISTORICAL_TARGETS_ADAM

    annotated = [
        {LR_TARGET_KEY: target, **{k: v for k, v in spec.items() if k != LR_TARGET_KEY}}
        for spec, target in zip(schedule_specs, hist)
    ]
    body = json.dumps({"LearningRateSchedule": annotated}, indent=4)

    return ValueError(
        _MESSAGE.format(
            key=LR_TARGET_KEY,
            model=LR_TARGET_MODEL,
            latent=LR_TARGET_LATENT,
            problem=problem,
            optimizer=optimizer,
            hist_0=hist[0],
            hist_1=hist[1],
            annotated="\n".join("    " + line for line in body.splitlines()),
            caution=_SCHEDULE_FREE_CAUTION if schedule_free else "",
        )
    )
