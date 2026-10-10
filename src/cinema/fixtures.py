"""Hand-written known-good screenplay and deliberately broken variants for validator benchmarks."""
from __future__ import annotations

import copy

from .schemas import Screenplay

KEEPER = {
    "id": "char_elias", "name": "Elias Marr", "role": "lighthouse keeper", "age": 64,
    "gender_presentation": "man", "ethnicity_and_skin": "weathered fair skin, ruddy cheeks",
    "face": "deep-set grey eyes, broad nose, scar through left eyebrow", "hair": "full white beard, grey hair under cap",
    "body": "stocky, slightly stooped", "costume": "navy wool peacoat with brass buttons, charcoal knit cap",
    "accessories": "brass oil lantern", "personality": "stubborn, haunted", "voice_id": "bm_george"}
LOC = {
    "id": "loc_lamp_room", "name": "lighthouse lamp room", "description":
    "cramped iron-framed lamp room at the top of a stone lighthouse, rain-streaked windows, storm outside",
    "lighting": "warm lantern glow against cold blue storm light", "color_palette": "deep navy, amber, slate grey",
    "key_props": ["brass oil lantern", "logbook on a wooden desk", "rain-streaked windows"],
    "atmosphere": "storm, isolation", "ambience_sound": "heavy rain on glass, wind howling, distant waves"}
LOC2 = {
    "id": "loc_stairs", "name": "spiral stone stairwell", "description": "narrow spiral stone stairs inside the lighthouse",
    "lighting": "only lantern light, deep shadows", "color_palette": "black, amber", "key_props": ["iron handrail"],
    "atmosphere": "claustrophobic", "ambience_sound": "dripping water, wind echoing in stone"}


def _shot(i, beat, loc, chars, action, seq, narr, must, dur=5.0, framing="medium shot", cam="slow push-in"):
    return {"id": f"shot_{i:02d}", "beat": beat, "duration": dur, "location_id": loc, "character_ids": chars,
            "subject_count": len(chars), "action": action, "expected_action_sequence": seq, "framing": framing,
            "camera_motion": cam, "lighting": "warm lantern glow, cold blue storm light",
            "environment_motion": "rain streaming down glass, lantern flame flickering", "props": ["brass oil lantern"],
            "costume_notes": "navy peacoat, knit cap", "visual_effects": "none", "narration": narr, "dialogue": [],
            "sfx": ["thunder rumble"], "image_prompt": f"{action}, first moment", "motion_prompt": action,
            "must_show": must}


def good() -> Screenplay:
    shots = [
        _shot(1, "hook", "loc_lamp_room", ["char_elias"], "Elias raises his lantern to the storm window and freezes",
              ["Elias lifts the lantern", "he stares out, frozen"], "Forty years on this rock. Tonight, something "
              "out there looked back.", ["man in navy peacoat and knit cap", "brass lantern", "rain-streaked window"]),
        _shot(2, "beginning", "loc_lamp_room", ["char_elias"], "Elias sets the lantern down beside the open logbook",
              ["Elias lowers the lantern onto the desk"], "Every night he logged the ships. Every night, nothing.",
              ["logbook on desk", "brass lantern"], framing="medium close-up"),
        _shot(3, "conflict", "loc_lamp_room", ["char_elias"], "Elias leans toward the glass as a faint light blinks far out at sea",
              ["a distant light blinks", "Elias leans toward the glass"], "Then a light blinked back. No ship should be out there.",
              ["man at the window", "distant light in the storm"]),
        _shot(4, "escalation", "loc_stairs", ["char_elias"], "Elias hurries down the spiral stairs holding the lantern",
              ["Elias descends the stairs"], "He ran down to signal back.", ["spiral stone stairs", "lantern light"],
              cam="tracking shot following"),
        _shot(5, "climax", "loc_lamp_room", ["char_elias"], "Elias turns from the window as the lamp room door slowly swings open behind him",
              ["the door swings open", "Elias turns around"], "But the door above him opened on its own.",
              ["open door", "man in peacoat"]),
        _shot(6, "ending", "loc_lamp_room", [], "The lantern flame dies beside the open logbook; only cold storm light remains",
              ["the lantern flame dies", "darkness"], "The lamp room was empty. The last entry was in his hand, dated tomorrow.",
              ["dark lamp room", "open logbook", "storm light"], framing="close-up", cam="static"),
    ]
    return Screenplay.model_validate({
        "story_id": "fixture_good", "target_seconds": 30, "shots": shots,
        "story": {"title": "The Last Entry", "genre": "supernatural", "logline": "A lighthouse keeper answers a signal "
                  "from a ship that cannot exist.", "theme": "isolation and fate", "hook": "the sea signals back",
                  "beginning": "routine logging", "conflict": "an impossible light", "escalation": "it spells his name",
                  "climax": "someone is already in the tower", "ending": "his own handwriting dated tomorrow",
                  "twist": "the log predicts his fate", "characters": [KEEPER], "relationships": [],
                  "locations": [LOC, LOC2], "visual_style": "photoreal moody", "color_palette": "navy and amber",
                  "music_mood": "low strings, slow dread", "narrator_voice_id": "bm_lewis"}})


def broken() -> dict[str, tuple[Screenplay, str, str]]:
    """name -> (screenplay, expected detector 'deterministic'|'critic', description)."""
    out = {}
    g = good()

    s = copy.deepcopy(g); s.shots[2].duration = 9.0
    out["duration_mismatch"] = (s, "deterministic", "shot durations do not sum to target; shot too long")

    s = copy.deepcopy(g); s.shots[3].character_ids = ["char_ghost"]
    out["unknown_character"] = (s, "deterministic", "shot references an undefined character")

    s = copy.deepcopy(g); s.shots[1].narration = ("Every night he logged the ships that passed by this lonely "
                                                  "tower, writing each name carefully in faded ink by the lamp.")
    out["narration_overflow"] = (s, "deterministic", "too many words for a 5 s shot")

    s = copy.deepcopy(g); s.shots[0], s.shots[5] = s.shots[5], s.shots[0]
    out["beats_out_of_order"] = (s, "deterministic", "ending first, hook last")

    s = copy.deepcopy(g)
    s.shots[1].narration = "Elias had been dead for a year; the tower stood empty."
    s.shots[3].narration = "Alive and well, Elias ran down to greet his wife."
    out["plot_contradiction"] = (s, "critic", "character dead, then alive with no explanation; wife never established")

    s = copy.deepcopy(g)
    s.shots[5].action = "Elias sits down and drinks tea"
    s.shots[5].narration = "Then he went to bed. Nothing else happened."
    s.shots[5].expected_action_sequence = ["Elias drinks tea"]
    s.story.ending = "he goes to bed"
    out["weak_ending"] = (s, "critic", "anticlimactic ending that drops every setup")

    s = copy.deepcopy(g)
    s.shots[4].action = ("Elias fights three armed smugglers, disarms them with complex martial arts, writes a "
                         "long letter, and reads the newspaper headline aloud on screen")
    s.shots[4].narration = "He fought them all."
    out["unfilmable_shot"] = (s, "critic", "complex multi-person fight + on-screen text in a 5 s shot")

    s = copy.deepcopy(g)
    s.shots[2].costume_notes = "bright yellow raincoat, bald head, no beard"
    s.shots[2].image_prompt = "a bald clean-shaven man in a bright yellow raincoat at the window"
    s.shots[2].must_show = ["bald clean-shaven man", "bright yellow raincoat"]
    out["costume_continuity_break"] = (s, "critic", "main character's look changes mid-story")
    return out
