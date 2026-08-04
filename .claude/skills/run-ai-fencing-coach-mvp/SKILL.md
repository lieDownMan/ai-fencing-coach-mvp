---
name: run-ai-fencing-coach-mvp
description: Build, run, and drive the AI Fencing Coach Clip Analysis UI (app.py). Use when asked to start the app, run app.py, take a screenshot of the fencing coach UI, or confirm a change works by uploading a clip and running analysis.
---

This repo's main runtime mode is `app.py`, a Gradio app that takes an
uploaded video clip through YOLO pose extraction, FenceNet action
classification, and rule-based posture heuristics, then shows an
annotated video and a playbook summary. Drive it via
`.claude/skills/run-ai-fencing-coach-mvp/driver.py`, a small Playwright
script (no `chromium-cli`/Node toolchain is present on this machine —
Playwright-for-Python fills the same role and is committed as the
harness). All paths below are relative to the repo root.

The repo also has `src/realtime/realtime_app.py` (live webcam OpenCV
window) and `web_realtime.py` (browser webcam streaming) — both need a
real or simulated camera and were not exercised by this skill. `app.py`
is the one path that's fully driveable with a plain uploaded file.

## Prerequisites

macOS (verified on arm64/Apple Silicon) or Linux. No system packages
beyond Python 3 were needed — the system `python3` (3.9.6 here) worked
fine despite `requirements.txt` recommending 3.11/3.12.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r .claude/skills/run-ai-fencing-coach-mvp/requirements-run.txt
python -m playwright install chromium
```

`requirements-run.txt` is a verified-working **subset** of the repo's
own `requirements.txt` — just what `app.py`'s import chain and the
driver actually need (torch, opencv, ultralytics, pandas, gradio,
scipy, python-dotenv, playwright), with `huggingface_hub` and
`pydantic` pinned to versions that don't break Gradio (see Gotchas).
Installing the full `requirements.txt` also works but pulls in
training/quantization extras (`bitsandbytes`, `onnx*`, `tensorboard`,
`jupyter`) that `app.py` doesn't need.

No `.env` / `GEMINI_API_KEY` is required — without it, `app.py` still
produces a deterministic playbook-based summary.

## Run (agent path)

Launch `app.py` in the background on a fixed port, wait for it to
serve, then drive it with the Playwright script:

```bash
source venv/bin/activate
GRADIO_SERVER_PORT=7860 nohup python app.py > /tmp/app_py.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:7860 >/dev/null && echo up && break; sleep 1; done
```

Cold start (first import of torch/ultralytics/matplotlib) took ~28s
in verification — poll, don't fix-sleep.

```bash
python .claude/skills/run-ai-fencing-coach-mvp/driver.py sample-clip /tmp/clip.mp4
python .claude/skills/run-ai-fencing-coach-mvp/driver.py run --video /tmp/clip.mp4 --out-dir /tmp/shots
```

`sample-clip` writes a tiny synthetic mp4 (no real fencers, just for
exercising the upload → pipeline → summary round trip). `run` uploads
the given video on the Analysis tab, clicks **Run Analysis**, waits
for the "Summary will appear here" placeholder to be replaced, and
writes three screenshots to `--out-dir`:

| file | shows |
|---|---|
| `01_initial.png` | page loaded, before upload |
| `02_uploaded.png` | file attached, before clicking Run Analysis |
| `03_done.png` | annotated video + playbook summary + drill table rendered |

It also prints any browser console errors it captured (`no console
errors` on a clean run). On first run ever, ultralytics auto-downloads
`yolov8n-pose.pt` (~6.5MB from GitHub) into the repo root — needs
network access once; subsequent runs reuse the cached file.

Stop the server: `lsof -ti:7860 -sTCP:LISTEN | xargs -r kill`.

For a real clip instead of the synthetic one, pass any local video
file to `--video` — a visible person in frame will actually produce
non-empty action segments and posture-error rows.

## Run (human path)

```bash
source venv/bin/activate
python app.py
```

Open `http://127.0.0.1:7860`, upload or record a clip, click **Run
Analysis**. Ctrl-C to stop.

## Test

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

79 passed, 2 skipped, 36 failed in the verified run — all 36 failures
are in `tests/test_system.py` and pre-date this skill: they import
`FencingCoachApplication` from `app.py`, a class that no longer exists
(the README notes older branch history still references removed flows
like `web_app.py`). Not caused by the environment set up here.

## Gotchas

- **`huggingface_hub` too new breaks Gradio.** Plain `pip install
  gradio` pulls the latest `huggingface_hub` (1.x), which removed
  `HfFolder` — Gradio 4.44.1 still imports it, so `import gradio`
  crashes with `ImportError: cannot import name 'HfFolder'`. Fix:
  `pip install "huggingface_hub<0.26"` after installing gradio.
- **`pydantic` too new breaks Gradio's schema introspection.** With
  pydantic 2.13, opening the app's own `/config` route throws
  `TypeError: argument of type 'bool' is not iterable` deep in
  `gradio_client/utils.py::get_type` (a `Dict[str, Any]`-shaped schema
  serializes `additionalProperties` as a bare bool on newer pydantic).
  Gradio's own startup probe hits this route and then refuses to
  launch on `0.0.0.0`, demanding `share=True`. Fix: `pip install
  "pydantic<2.11"`.
- **`src/__init__.py` pulls in `scipy` even for the clip-analysis
  path.** `app.py` never imports training code directly, but
  `src.realtime.feedback_config` (which it does import) triggers
  `src/__init__.py`, which imports `src.training`, which needs
  `scipy.io.loadmat`. Install `scipy` even though nothing in `app.py`
  looks like it needs it.
- **`ultralytics` installs `opencv-python` alongside
  `opencv-python-headless`.** Harmless (headless still gets used), just
  don't be surprised to see both in `pip freeze`.
- **No `chromium-cli` / Node on this machine.** Used Playwright for
  Python instead (`pip install playwright && playwright install
  chromium`) — same role, committed as `driver.py`.
- **`GRADIO_SERVER_PORT` env var, not a CLI flag** — `app.py` reads it
  directly (`_pick_gradio_port` in `app.py`) to pin the port instead of
  auto-scanning from 7860, which matters when scripting a wait-for-port
  loop.

## Troubleshooting

- **`ValueError: When localhost is not accessible, a shareable link
  must be created`** on `app.launch()`: this is Gradio's own startup
  probe failing because of the pydantic bug above, not an actual
  network problem — pin `pydantic<2.11` and relaunch.
- **`ModuleNotFoundError: No module named 'torch'` / `'gradio'`**: the
  system `python3` has neither; make sure the venv is activated before
  running anything.
