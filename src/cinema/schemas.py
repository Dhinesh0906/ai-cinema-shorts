"""Structured data contracts shared by every stage.

LLM-facing models (StoryDraft, ShotPlanDraft, CriticReport, EditPlan) are passed to Ollama as JSON
schemas; everything the LLM returns is re-validated here and by story.validator deterministic checks.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

KOKORO_VOICES = {
    # American / British English Kokoro v1.0 voices.
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_river", "af_jessica",
    "af_alloy", "af_aoede", "af_kore", "am_adam", "am_michael", "am_fenrir", "am_puck", "am_echo",
    "am_eric", "am_liam", "am_onyx", "am_santa", "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
}


# ---------------------------------------------------------------- story / screenplay

class Character(BaseModel):
    id: str = Field(description="snake_case id, e.g. char_mara")
    name: str
    role: str = Field(description="role in the story")
    age: int = Field(ge=18, le=100, description="adults only")
    gender_presentation: str
    ethnicity_and_skin: str
    face: str = Field(description="fixed facial features: face shape, eyes, nose, distinguishing marks")
    hair: str = Field(description="exact hairstyle and color")
    body: str = Field(description="build and height")
    costume: str = Field(description="exact outfit with colors and materials, fixed for the whole story")
    accessories: str
    personality: str
    voice_id: str = Field(description="Kokoro voice id")


class Location(BaseModel):
    id: str = Field(description="snake_case id, e.g. loc_lighthouse")
    name: str
    description: str
    lighting: str
    color_palette: str
    key_props: list[str]
    atmosphere: str
    ambience_sound: str = Field(description="continuous background sound of this place")


class Relationship(BaseModel):
    a: str
    b: str
    relation: str


class StoryDraft(BaseModel):
    title: str
    genre: str
    logline: str
    theme: str
    hook: str = Field(description="what happens in the first 2 seconds to grab attention")
    beginning: str
    conflict: str
    escalation: str
    climax: str
    ending: str
    twist: str
    characters: list[Character] = Field(min_length=1, max_length=4)
    relationships: list[Relationship]
    locations: list[Location] = Field(min_length=1, max_length=4)
    visual_style: str
    color_palette: str
    music_mood: str = Field(description="instrumental score description, no artist names")
    narrator_voice_id: str


class DialogueLine(BaseModel):
    speaker_id: str
    text: str
    delivery: Literal["voice_over", "off_screen"] = Field(
        description="no on-screen lip sync is available: voice_over or off_screen only")


class Shot(BaseModel):
    id: str = Field(description="shot_01, shot_02, ...")
    beat: Literal["hook", "beginning", "conflict", "escalation", "climax", "ending"]
    duration: float = Field(description="seconds")
    location_id: str
    character_ids: list[str]
    subject_count: int = Field(ge=0, le=4, description="number of people visible")
    action: str = Field(description="what physically happens, start to end")
    expected_action_sequence: list[str] = Field(min_length=1, max_length=4,
                                                description="ordered visible events")
    framing: str = Field(description="e.g. close-up, medium shot, wide establishing")
    camera_motion: str = Field(description="e.g. slow push-in, static, slow pan left")
    lighting: str
    environment_motion: str = Field(description="rain, smoke, flicker, wind ...")
    props: list[str]
    costume_notes: str
    visual_effects: str
    narration: str = Field(description="narrator line for this shot, may be empty")
    dialogue: list[DialogueLine]
    sfx: list[str] = Field(description="short sound effect descriptions")
    image_prompt: str = Field(description="composition of the first frame")
    motion_prompt: str = Field(description="motion that happens during the shot")
    must_show: list[str] = Field(min_length=1, description="visual facts a validator must confirm")


class ShotPlanDraft(BaseModel):
    shots: list[Shot] = Field(min_length=2, max_length=14)


class Screenplay(BaseModel):
    story_id: str
    target_seconds: int
    story: StoryDraft
    shots: list[Shot]

    def character(self, cid: str) -> Character:
        return next(c for c in self.story.characters if c.id == cid)

    def location(self, lid: str) -> Location:
        return next(loc for loc in self.story.locations if loc.id == lid)

    def shot(self, sid: str) -> Shot:
        return next(s for s in self.shots if s.id == sid)

    def starts(self) -> dict[str, float]:
        t, out = 0.0, {}
        for s in self.shots:
            out[s.id] = round(t, 3)
            t += s.duration
        return out


# ---------------------------------------------------------------- validation

Severity = Literal["critical", "major", "minor"]


class Finding(BaseModel):
    criterion: str
    result: Literal["pass", "fail"]
    severity: Severity
    evidence: str
    recommended_correction: str
    confidence: float = Field(ge=0, le=1)
    correction_required: bool
    affected: list[str] = Field(description="shot ids, character ids, or 'story'")


class CriticReport(BaseModel):
    findings: list[Finding]
    overall_pass: bool
    summary: str


class CheckItem(BaseModel):
    item: str
    verdict: Literal["yes", "no", "unclear"]
    evidence: str


class VisualVerdict(BaseModel):
    """What the VLM returns for one image or one shot."""
    description: str
    checks: list[CheckItem]
    defects: list[str]
    correction: str


class Decision(BaseModel):
    """One logged validation decision (also written to SQLite)."""
    asset_id: str
    stage: str
    criterion: str
    evaluator: str
    passed: bool
    mandatory: bool = True
    score: float | None = None
    evidence: str = ""
    action: str = ""
    attempt: int = 1


# ---------------------------------------------------------------- editing

Transition = Literal["cut", "crossfade", "fade_from_black", "fade_to_black", "dip_to_black"]
Grade = Literal["neutral", "teal_orange", "cold_desaturated", "warm_vintage", "noir_contrast", "moonlit_blue"]


class EditShot(BaseModel):
    shot_id: str
    in_point: float = Field(ge=0, description="seconds into the approved clip")
    out_point: float = Field(gt=0)
    transition_in: Transition
    transition_seconds: float = Field(ge=0, le=1.0)


class EditPlan(BaseModel):
    """LLM editing director output. Timing of audio is derived from shot placement."""
    shots: list[EditShot]
    grade: Grade
    music_gain_db: float = Field(ge=-30, le=-6)
    ambience_gain_db: float = Field(ge=-30, le=-6)
    sfx_gain_db: float = Field(ge=-20, le=0)
    narration_lead_seconds: float = Field(ge=0, le=1.0, description="delay of narration after shot start")
    end_with_fade: bool
    rationale: str


class AudioEvent(BaseModel):
    kind: Literal["narration", "dialogue", "sfx", "ambience", "music"]
    asset_id: str
    path: str
    start: float
    duration: float
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0


class CaptionWord(BaseModel):
    word: str
    start: float
    end: float


class Caption(BaseModel):
    start: float
    end: float
    text: str
    words: list[CaptionWord] = []   # absolute timeline times, for word-by-word highlight


class EDL(BaseModel):
    """Fully resolved, machine-renderable edit decision list."""
    job_id: str
    fps: int = 24
    width: int = 1080
    height: int = 1920
    grade: Grade
    video: list[dict]          # {shot_id, path, in, out, timeline_start, transition_in, transition_seconds}
    audio: list[AudioEvent]
    captions: list[Caption]
    duration: float
    fade_out: float = 0.0
    source: Literal["ai_director", "deterministic_fallback"]
    rationale: str = ""

    @field_validator("video")
    @classmethod
    def _nonempty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("EDL has no video")
        return v
