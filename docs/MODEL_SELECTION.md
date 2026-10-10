# Model selection report

Machine: Apple M5 Pro, 48 GB unified memory, macOS 27.0.1, Ollama 0.30.10, mlx-gen 0.38.0,
mlx-vlm 0.7.6, FFmpeg 9.0.2 (no libass/drawtext). All numbers below were measured on this machine
unless marked "documented". Raw data: `benchmarks/results.json`, `state/cinema.sqlite3` (worker_runs).

| stage | selected | repo / revision | license | commercial use | runtime |
|---|---|---|---|---|---|
| story writer | Gemma 4 31B | `gemma4:31b` (Ollama Cloud) | Apache-2.0 | yes (model); Ollama Cloud free-tier terms | ollama.com API, rotating keys |
| script critic | GPT-OSS 120B | `gpt-oss:120b` (Ollama Cloud) | Apache-2.0 | yes (model); Ollama Cloud free-tier terms | ollama.com API, rotating keys |
| image | FLUX.2 [klein] 4B, 4-bit MLX | `AbstractFramework/flux.2-klein-4b-4bit` @4540c00 (4.6 GB) | Apache-2.0 | yes | mlx-gen |
| image/video/final validator | Qwen3-VL 32B Instruct 4-bit | `mlx-community/Qwen3-VL-32B-Instruct-4bit` @6e5644d (19.6 GB) | Apache-2.0 | yes | mlx-vlm |
| identity / layout evidence | Apple Vision (face/human rectangles, feature prints) | macOS system framework | system | n/a | pyobjc |
| video | Wan2.2 TI2V-5B, 8-bit MLX | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` @6875952 (18.2 GB) | Apache-2.0 | yes | mlx-gen |
| narration | Kokoro-82M v1.0 | kokoro-onnx model files (0.35 GB) | Apache-2.0 / MIT | yes | onnxruntime CPU |
| speech validator | Whisper large-v3-turbo | `mlx-community/whisper-large-v3-turbo` (3 GB) | MIT | yes | mlx-whisper |
| sfx / ambience / score | Stable Audio 3 small (sm-sfx, sm-music) | `stabilityai/stable-audio-3-optimized` (local MLX clone) | Stability AI Community License | yes below USD 1M annual revenue | sa3_mlx |
| sfx relevance | CLAP HTSAT unfused | `laion/clap-htsat-unfused` @8fa0f1c (0.6 GB) | Apache-2.0 | yes | transformers CPU |

Full license texts fetched into `models/licenses/` by `cinema setup`.

## Rejected / not installed

- **Qwen-Image-2.1-Turbo**: HF card license `other` = Qwen Research License (not commercial without a
  separate license); 32.5 GB; no verified MLX route for the Turbo checkpoint. Kept out of production;
  not downloaded.
- **Bernini-R 1.3B**: documented by mlx-gen for reference-guided generation; not downloaded or
  benchmarked in this build (marked experimental/unverified in the registry).
- **Wan2.2 A14B** (documented by mlx-gen as stronger at small canvases): not in the approved list and
  ~30 GB+ extra; not downloaded.
- **InsightFace**-style face embeddings: the commonly used model weights are non-commercial; Apple
  Vision feature prints were used instead (supporting evidence only).

## Video benchmarks (Wan2.2 TI2V-5B, I2V, same keyframe/prompt/seed 42)

| canvas | frames | steps | low-RAM | wall s | peak footprint | MLX peak | result |
|---|---|---|---|---|---|---|---|
| 256×448 | 17 | 8 | yes | 101 | 11.3 GiB | 10.6 GiB | **collapsed**: psychedelic corruption after frame 1 |
| 480×832 | 49 | 20 | yes | 294 | 22.5 GiB | 23.2 GiB | coherent; requested action weak |
| 480×832 | 81 | 25 | no | 390 | 34.7 GiB | 33.0 GiB | coherent, action performed; too close to 48 GB |
| 480×832 | 81 | 25 | yes | 439 | 22.0 GiB | 23.2 GiB | **selected** (bit-identical to the no-low-RAM run) |
| 704×1280 | 81 | 25 | yes | 1288 | 41.0 GiB | 50.6 GiB | best detail but **swapped 9.8 GB → rejected** |

mlx-gen's own docs say TI2V-5B canvases below 832×480 are routing checks, not quality settings,
which matches the collapse measured at 256×448. The previous prototype's presets (256×448, 320×576)
could not have produced usable footage.

## Image benchmarks (FLUX.2 klein 4B, 4 steps)

480×832 text-to-image: 5.7 s wall, 12.3 GiB peak footprint. 704×1280: 8.1 s, 15.3 GiB.
Two concurrent requests were serialized by the heavy-model lock (memory benchmark): 4.7 s + 4.2 s.

## Writer / critic selection

Operator decision (2026-10-10): writer `gemma4:31b` and critic `gpt-oss:120b` on Ollama Cloud with a
rotating free-key ring. 120B does not fit locally (~65 GB > 48 GB). Measured before the switch:

| writer (local) | time per draft (story + 10 shots) | deterministic fails | critic mandatory fails |
|---|---|---|---|
| qwen3.5:27b, thinking on | 1319 s | duplicate/garbage location ids, thin narration | 3 |
| gemma4:31b-it-qat | 346 s | voice id (trailing space), thin narration | 1 |
| gemma4:31b cloud | StoryDraft in 3.2 s | — | — |

Ollama Cloud **ignores the `format` JSON schema** (measured: free text "Elephant" for a required-integer
schema on both models), so in cloud mode the schema is embedded in the system prompt and Pydantic
validation + error-feedback retries enforce it. Local fallbacks remain installed: `gemma4:31b-it-qat`,
`qwen3.5:27b`, `gpt-oss:20b`.

## Validator benchmark (`cinema benchmark --stage validators`, small hand-built set)

| validator | bad cases caught | false accepts | false rejects on good |
|---|---|---|---|
| script: deterministic + gpt-oss:120b (cloud) | 8/8 (4 structural by deterministic checks, 4 semantic by critic) | 0 | 1 of 5 runs on the clean fixture (stochastic; costs one rewrite) |
| script: deterministic + gpt-oss:20b (local, earlier) | 7/8 | 1 | 1 |
| image: Qwen3-VL 32B checklist | 4/4 (wrong clothing, wrong person, two people, wrong location) | 0 | 1/2 (in-world handwriting flagged as "text"; rule fixed to overlays only) |
| video technical (ffmpeg/OpenCV) | frozen, black, broken file | collapsed generation passed technical | 0 |
| video semantic (VLM, 6 ordered frames) | wrong action, collapsed generation | 0 | 1 ("setting unclear"; item reworded to "same place as first frame") |
| speech: Whisper + loudness | missing words, clipping, mid-line gap | 0 | 0 |

VLM cost: ~48 s per image check, 14.5 tok/s, 21.2 GiB peak footprint (measured).

Calibration (Apple Vision feature prints): drift from keyframe for good clips 0.57-0.85 vs collapsed
0.98 — not discriminative, so the drift gate stays disabled; face distance same person 0.85 vs
different person 0.96 — supporting evidence only.

## LLM observations

- Ollama 0.30.10 enforces the JSON-schema `format` for `qwen3.5:27b` **only with thinking enabled**
  (`think=false` returned free text "Panda" for a required-integer schema). The writer therefore runs
  with thinking on (`models.writer_think: true`). GPT-OSS honors the schema either way.
- `qwen3.5:27b` decodes at ~15 tok/s; model load ~5 s.
- Ollama unload with `keep_alive=0` is asynchronous: the model was still listed in `/api/ps`
  immediately after the call, so the resource manager polls until it is gone.

## Memory isolation (cinema benchmark --stage memory)

PASS: two concurrent FLUX requests ran sequentially (no overlapping heavy runs), every worker exited
clean, Qwen3.5 was listed in `/api/ps` during its session and absent after it, no leftover worker
processes. Unit tests also prove process-group reaping of orphaned grandchildren and timeouts.
