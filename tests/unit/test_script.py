from cinema.fixtures import broken, good
from cinema.pipeline.script_stage import normalize
from cinema.schemas import Screenplay
from cinema.story.generator import normalize_genre, shot_budget
from cinema.story.validator import deterministic_checks, mandatory_failures


class _Ctx:
    def __init__(self, cfg):
        self.cfg = cfg


def test_known_good_passes_deterministic(cfg):
    assert mandatory_failures(deterministic_checks(cfg, good())) == []


def test_structural_breaks_are_caught_deterministically(cfg):
    for name, (sp, detector, _) in broken().items():
        fails = mandatory_failures(deterministic_checks(cfg, sp))
        if detector == "deterministic":
            assert fails, f"{name} should be caught deterministically"


def test_normalize_fixes_duration_total_within_bounds(cfg):
    sp = good()
    sp.shots[0].duration = 9.0
    sp.shots[1].duration = 1.0
    out = normalize(_Ctx(cfg), sp)
    assert abs(sum(s.duration for s in out.shots) - 30) < 0.06
    assert all(cfg.shot_min_seconds - 0.05 <= s.duration <= cfg.shot_max_seconds + 0.05 for s in out.shots)
    assert [s.id for s in out.shots] == [f"shot_{i:02d}" for i in range(1, 7)]


def test_screenplay_roundtrip_and_starts():
    sp = good()
    again = Screenplay.model_validate_json(sp.model_dump_json())
    assert again.starts()["shot_03"] == 10.0


def test_shot_budget_and_genre(cfg):
    lo, rec, hi = shot_budget(cfg, 45)
    assert lo == 9 and lo <= rec <= hi
    import random
    assert normalize_genre("sci-fi", random.Random(1)) == "science fiction"
    assert normalize_genre("random", random.Random(1)) in {
        "horror", "psychological thriller", "science fiction", "mystery", "supernatural", "fantasy",
        "emotional drama", "suspense", "adventure"}


def test_on_screen_dialogue_rejected_by_schema():
    import pytest
    from pydantic import ValidationError
    data = good().model_dump()
    data["shots"][0]["dialogue"] = [{"speaker_id": "char_elias", "text": "hi", "delivery": "on_screen"}]
    with pytest.raises(ValidationError):
        Screenplay.model_validate(data)
