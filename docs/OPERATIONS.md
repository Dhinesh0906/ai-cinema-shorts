# Operations

## Starting a film and walking away

```sh
cd ~/Desktop/automations/ai-cinema-shorts
uv run cinema doctor                       # must say ready_for_autonomous_run: true
uv run cinema run --autonomous --duration 45 --genre random --detach
uv run cinema status JOB_ID                # any time, from any terminal
```

`--detach` survives closing the terminal. Sleep is prevented only while the job process lives
(`caffeinate -ims -w PID`); global power settings are not changed. Closing the lid still sleeps the
Mac: the job is then interrupted and `uv run cinema resume JOB_ID` continues from the last validated
asset.

## Where time goes (balanced preset, measured on this M5 Pro)

Per 5-second shot: ~7.3 min Wan generation (81 frames, 25 steps, low-RAM), plus VLM validation.
A 45 s film is ~9-10 shots, so allow roughly 1.5-3 h including retries, image generation,
validation loads and narration. `draft` saves ~20 % per shot with lower motion fidelity.

## Statuses and what to do

- `BLOCKED` — read `outputs/JOB_ID/failure_report.json`. Typical causes: a shot that failed the
  semantic video gate 3 times (prompt asks for something Wan cannot render), memory pressure that
  never recovered, a model missing. Fix the cause, then `uv run cinema resume JOB_ID` (this grants
  one more retry budget to the blocked assets; maximum 5 resumes per job).
- `FAILED` — final validation still failed after 2 attributed repair rounds. The rejected draft is
  `outputs/JOB_ID/REJECTED_draft.mp4` with `quality_report.json`. Start a new job.
- `COMPLETE` — review `final.mp4` and `quality_report.json` (limitations are listed) before publishing.

## Memory model

One heavy model at a time (file lock `state/heavy_model.lock`). Before each launch: normal memory
pressure and model-estimate + 8 GiB available, disk ≥ 100 GiB reserve. After each worker: verified
exit of the whole process group, Ollama `/api/ps` empty, memory recovered. All measurements are in
the `worker_runs` table and `outputs/JOB_ID/worker_runs.json`. Only processes the pipeline started
are ever terminated.

## Disk

Models ≈ 95 GB (HF cache + Ollama). Each job keeps references, clips, audio and rejected
candidates (~0.5-1.5 GB). `uv run cinema cleanup --dry-run` lists removable scratch of finished
jobs; `--execute` deletes only `work/` and `temp/` of COMPLETE/FAILED jobs.

## Changing models

Edit `config.yaml` `models:`. Each model sits behind one adapter (`model_workers.py`,
`workers/*.py`, `llm.py`). Re-run `cinema benchmark --stage validators` after changing a validator.

## Ollama Cloud key ring

Keys: `secrets/ollama_api_keys.json` (gitignored, mode 600). State: `state/cloud_keyring.json`
(current index, exhausted keys, completed cycles), shared by all jobs under a file lock.
Rate limit (HTTP 429 or a "limit" error) or an invalid key -> next key. All exhausted -> reset the
ring, pause `cloud.cycle_pause_s`, start again from key 1 — forever. Network/5xx errors (not rate
limits) retry with backoff for `cloud.max_error_seconds`, then the job becomes BLOCKED (resumable).
Logs show only key index and last 4 characters.
