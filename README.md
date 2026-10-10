# AI Cinema Shorts

An autonomous factory for short vertical (9:16) films, built for Apple Silicon. One command, or one Telegram
message, takes an idea all the way to a finished, subtitled 30–60 second film:

**story → critic → characters & locations → shot keyframes → moving video → narration → sound & music →
edit → final inspection → publish**

Every step is checked by an independent AI validator before the next one starts. Anything that fails is
regenerated with the validator's feedback, so nothing is faked or skipped. It runs on a Mac with **$0 of
paid APIs**. The heavy models run locally (MLX), and the two text models use the free Ollama Cloud tier.

```sh
uv run cinema run --autonomous --duration 45 --genre random
```

---

## What it does

- **Writes an original story.** The writer (`gemma4:31b`) produces a story and shot list. A *different*
  model family (`gpt-oss:120b`) critiques it, and the writer rewrites until the critic approves.
  Finished stories are remembered, so premises don't repeat.
- **Designs the cast and world.** FLUX.2 klein 4B paints character portraits and location plates, then a
  first frame for every shot that uses those references, so faces and costumes stay consistent.
- **Animates every shot.** Wan 2.2 (TI2V-5B) turns each first frame into real moving video (image-to-video).
- **Checks everything with an independent vision model.** Qwen3-VL 32B goes through a checklist for every
  image and every clip: right characters, right costume, the action really happens, no stray limbs,
  no warping, no frozen frames. FFmpeg/OpenCV add hard technical checks (black, frozen, wrong size,
  flicker).
- **Voices it.** Kokoro narrates, Whisper transcribes the result to confirm every word is right, and
  Stable Audio 3 makes ambience, sound effects and music. CLAP checks that the sounds fit.
- **Edits it.** An AI director builds an edit decision list. FFmpeg renders 1080×1920 H.264 with
  Shorts-style captions in the lower middle of the frame, where the word being spoken lights up yellow.
- **Inspects the final film** and repairs only the part that failed (one keyframe → its clip → the edit).
- **Runs from your phone.** An always-on Telegram bot has `start`, `status` (with % and ETA), `pause`,
  `resume`, `abort`, `script`, `publish` and `queue`, and sends you the video when it's done.
- **Publishes.** YouTube Shorts uploads through the official API, with SEO title, description and hashtags
  and the AI disclosure. Instagram Reels is supported too. Films are spaced 2 h apart.

### Reliability by design
- **One heavy model at a time.** Each model runs in its own process under a global lock, with a memory
  check before launch, peak-memory measurement, and verified unload afterwards. A 48 GB Mac never swaps.
- **Resumable everywhere.** Every asset and its dependencies live in SQLite. A crash, reboot or `pause`
  continues from the last validated asset.
- **Honest outcomes.** Every job ends `COMPLETE`, `BLOCKED` (needs attention; `resume` continues it) or
  `FAILED`.
- **Bounded retries.** Each asset gets 3 attempts. If all are rejected, the best validated attempt is used
  instead of stalling.
- **Clean storage.** Scratch files (rejected attempts, candidates, sampled frames) are deleted when a
  film completes. Approved clips, keyframes, references and audio are kept for re-edits, thumbnails
  and sequels.

---

## Requirements

| | |
|---|---|
| Mac | Apple Silicon. **48 GB unified memory recommended**; the video and vision models peak at ~23 GB each. |
| Disk | ~60 GB for models, plus ~100 GB free as a safety reserve |
| Tools | [Homebrew](https://brew.sh), `ffmpeg`, [`uv`](https://docs.astral.sh/uv/), Python 3.12 (uv installs it) |
| Accounts (free) | [Ollama](https://ollama.com) API key(s) for the writer/critic. Optional: Telegram bot, Google Cloud (YouTube API). |

**Speed (M5 Pro, 48 GB):** about 7–8 min per 5-second shot at 480×832, so a 45 s film (10 shots)
takes about 2–3 h including validation and redos. It's best run overnight.

---

## Getting started

### 1. Install
```sh
brew install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh        # if you don't have uv

git clone https://github.com/Dhinesh0906/ai-cinema-shorts.git
cd ai-cinema-shorts
uv sync --extra dev
uv run cinema setup              # creates folders + config.yaml, installs mlx-gen
uv run cinema setup --models     # downloads the models (~50 GB, asks first)
```

Stable Audio 3 (sound effects and music) is a separate install:
```sh
git clone https://github.com/Stability-AI/stable-audio-3 ~/Desktop/stable-audio-3
cd ~/Desktop/stable-audio-3/optimized/mlx && ./install.sh
```
If you put it somewhere else, set `models.sa3_dir` in `config.yaml`. You can also turn audio generation
off with `generate_sfx: false` / `generate_music: false`.

### 2. Add your keys
Secrets live in `secrets/`, which is gitignored and never committed. Templates are in `examples/secrets/`.

```sh
mkdir -p secrets && chmod 700 secrets
cp examples/secrets/ollama_api_keys.json secrets/     # required: your Ollama API key(s)
```
Get a key at **ollama.com → Settings → API keys**. You can list several: when one hits its rate
limit, the next is used automatically, and the ring starts over once all are used.

Prefer fully offline? Point `models.writer`/`models.critic` at local Ollama models and set
`cloud.models: []`.

### 3. Check and test
```sh
uv run cinema doctor             # tools, models, memory, disk: everything should be green
uv run pytest -q                 # unit tests (~20 s)
uv run cinema test-video         # one real moving clip + technical checks (~8 min)
uv run cinema test-pipeline      # a 12 s, 3-shot film through every gate (renders 3 shots)
```

### 4. Make a film
```sh
uv run cinema run --autonomous --duration 45 --genre "dark fantasy"
uv run cinema run --autonomous --duration 40 --topic "a future love story told through old voicemails"
uv run cinema status <JOB_ID>
uv run cinema resume <JOB_ID>
```
The result is in `outputs/<JOB_ID>/final.mp4`.

---

## Telegram bot (optional, recommended)

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Send your bot any message, then get your chat id from
   `https://api.telegram.org/bot<TOKEN>/getUpdates` (`message.chat.id`).
3. `cp examples/secrets/telegram.json secrets/` and fill in both values.
4. Run it: `uv run cinema bot`. To keep it always on, even after a reboot, use the launchd template:
   ```sh
   sed "s|__PROJECT__|$PWD|g; s|__HOME__|$HOME|g" deploy/cinema-telegram-bot.plist \
     > ~/Library/LaunchAgents/local.cinema-telegram-bot.plist
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.cinema-telegram-bot.plist
   ```

| Message | What happens |
|---|---|
| `start` | random 30–45 s film, random genre |
| `start 40 fictional future story with future romance` | 40 s film about that topic |
| `status` | stage, progress %, ETA, any problems |
| `script` | the approved screenplay |
| `pause` / `resume` | finishes the current step, unloads the models, keeps all progress |
| `abort` | asks yes/no, then stops and deletes that film's progress |
| `publish` | previews the title, description and hashtags, then queues it on yes |
| `queue` | publishing schedule and results |

The bot only answers the chat id in `secrets/telegram.json`.

---

## Publishing (optional)

**YouTube Shorts**
1. In [Google Cloud Console](https://console.cloud.google.com): create a project, enable **YouTube Data
   API v3**, and set up the OAuth consent screen (External; add yourself as a test user).
2. Create an OAuth client of type **Desktop app**, download the JSON, and save it as
   `secrets/youtube_client.json`.
3. `uv run cinema publish-auth youtube` opens the browser once to sign in.

New API projects can only upload as *private* until Google approves an API audit. Until then, set the
video's visibility in YouTube Studio.

**Instagram Reels** needs an Instagram Professional (Creator/Business) account and a long-lived token
with `instagram_business_content_publish`. Save it as `secrets/instagram.json`, then add `instagram`
to `publish_platforms` in `config.yaml`. By default only YouTube is posted automatically, and the bot
gives you the Instagram caption to post yourself.

Set your channel name with `channel_name` in `config.yaml`. It's used in descriptions and calls to
action.

---

## All commands

| Command | Purpose |
|---|---|
| `cinema doctor` | readiness report |
| `cinema setup [--models]` | folders, config, mlx-gen, model downloads, licenses |
| `cinema models` | model registry: repo, size, license |
| `cinema run --autonomous --duration N [--genre G] [--topic T] [--preset P] [--detach]` | make a film |
| `cinema status JOB` / `cinema jobs` | progress / all jobs |
| `cinema pause JOB` / `cinema resume JOB` | pause after the current step / continue |
| `cinema cleanup [--execute]` | remove scratch files of finished jobs |
| `cinema benchmark --stage story\|image\|video\|validators\|memory` | measured benchmarks and adversarial validator tests |
| `cinema bot` | run the Telegram bot |
| `cinema publish-auth youtube` | one-time YouTube sign-in |

Run each one as `uv run cinema …`.

## Project layout
```
src/cinema/
  orchestrator.py      job state machine, resume, repair rounds, cleanup
  pipeline/            script, image, video, audio, edit and final stages (+ best-of fallback)
  quality/vlm.py       vision-model checklists and scoring
  resource_manager.py  one-heavy-model lock, memory preflight, pause, process reaping
  cloud.py             Ollama Cloud client with rotating key ring
  editing/             EDL, Shorts captions, FFmpeg renderer
  publish/             YouTube, Instagram, queue, SEO metadata
  telegram_bot.py      always-on remote control
  workers/             isolated model subprocesses (VLM, Whisper, TTS, SFX, CLAP)
config.example.yaml    every setting, with measured notes
docs/                  MODEL_SELECTION.md (why each model), OPERATIONS.md (troubleshooting)
examples/secrets/      credential templates
deploy/                launchd template for the bot
```

Per film: `projects/<JOB>/` holds the approved references, keyframes, clips, audio and validation
reports. `outputs/<JOB>/` holds `final.mp4`, the thumbnail, subtitles, the YouTube metadata, the
screenplay, and full model and decision manifests.

---

## Limitations

- Video quality is limited by what fits on a laptop. Wan 2.2 5B at 480p is good, not state of the art.
  The video backend is pluggable (`video_backend`), and larger models on free cloud GPUs are next on
  the roadmap.
- Automated checks catch most defects (stray hands, frozen shots, wrong costumes), but they don't
  guarantee good filmmaking. Watch before you publish.
- Closing the lid sleeps the Mac and pauses the job. `resume` continues it.

## Licenses

This repository is the pipeline code. Each model has its own license, which `cinema setup` downloads
into `models/licenses/` and records per film in `license_manifest.json`: FLUX.2 klein 4B, Wan 2.2
(Apache-2.0), Qwen3-VL, Kokoro, Whisper, Stable Audio 3 (Stability AI Community License), CLAP. Check
them before commercial use, and disclose AI-generated content where platforms require it. The pipeline
adds the disclosure automatically.
