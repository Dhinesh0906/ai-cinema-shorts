"""Job and asset state machines. Transitions not listed here raise, so state can't drift silently."""
from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    PENDING = "PENDING"
    PREFLIGHT = "PREFLIGHT"
    GENERATING_SCRIPT = "GENERATING_SCRIPT"
    VALIDATING_SCRIPT = "VALIDATING_SCRIPT"
    GENERATING_IMAGES = "GENERATING_IMAGES"
    VALIDATING_IMAGES = "VALIDATING_IMAGES"
    GENERATING_VIDEO = "GENERATING_VIDEO"
    VALIDATING_VIDEO = "VALIDATING_VIDEO"
    GENERATING_AUDIO = "GENERATING_AUDIO"
    VALIDATING_AUDIO = "VALIDATING_AUDIO"
    EDITING = "EDITING"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class AssetState(StrEnum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    PASSED = "PASSED"
    RETRY_PENDING = "RETRY_PENDING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


TERMINAL_JOB = {JobState.COMPLETE, JobState.BLOCKED, JobState.FAILED}

# Stage order. Any working state may go back to an earlier one (dependency invalidation after a
# final-validation failure), or to BLOCKED / FAILED.
JOB_ORDER = [
    JobState.PENDING, JobState.PREFLIGHT,
    JobState.GENERATING_SCRIPT, JobState.VALIDATING_SCRIPT,
    JobState.GENERATING_IMAGES, JobState.VALIDATING_IMAGES,
    JobState.GENERATING_VIDEO, JobState.VALIDATING_VIDEO,
    JobState.GENERATING_AUDIO, JobState.VALIDATING_AUDIO,
    JobState.EDITING, JobState.FINAL_VALIDATION, JobState.COMPLETE,
]


def can_transition_job(src: JobState, dst: JobState) -> bool:
    if src == dst:
        return True
    if dst == JobState.PAUSED:
        return src not in TERMINAL_JOB
    if src == JobState.PAUSED:
        return dst in (JobState.PREFLIGHT, JobState.BLOCKED, JobState.FAILED)
    if src in TERMINAL_JOB:
        # resume of a BLOCKED job (e.g. after memory recovered) restarts preflight; COMPLETE/FAILED are final.
        return src == JobState.BLOCKED and dst == JobState.PREFLIGHT
    if dst in (JobState.BLOCKED, JobState.FAILED):
        return True
    if src not in JOB_ORDER or dst not in JOB_ORDER:
        return False
    i, j = JOB_ORDER.index(src), JOB_ORDER.index(dst)
    if dst == JobState.COMPLETE:
        return src == JobState.FINAL_VALIDATION
    # forward (stages with nothing left to do are skipped) or backward (repair after final validation)
    return j != i


ASSET_TRANSITIONS: dict[AssetState, set[AssetState]] = {
    AssetState.PENDING: {AssetState.GENERATING, AssetState.BLOCKED, AssetState.FAILED},
    AssetState.GENERATING: {AssetState.VALIDATING, AssetState.RETRY_PENDING, AssetState.FAILED,
                            AssetState.PENDING, AssetState.BLOCKED},
    AssetState.VALIDATING: {AssetState.PASSED, AssetState.RETRY_PENDING, AssetState.FAILED,
                            AssetState.PENDING, AssetState.BLOCKED},
    AssetState.RETRY_PENDING: {AssetState.GENERATING, AssetState.FAILED, AssetState.BLOCKED, AssetState.PENDING},
    AssetState.PASSED: {AssetState.PENDING},          # invalidated by an upstream change
    AssetState.BLOCKED: {AssetState.PENDING},         # resume
    AssetState.FAILED: {AssetState.PENDING},          # explicit invalidation only
}


def can_transition_asset(src: AssetState, dst: AssetState) -> bool:
    return src == dst or dst in ASSET_TRANSITIONS[src]


class TransitionError(RuntimeError):
    pass
