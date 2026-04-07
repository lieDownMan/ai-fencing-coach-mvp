# Quick Start

This file is intentionally short. For product scope and workflow, read [mvpspec.md](mvpspec.md).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Interactive Mode

```bash
python app.py --interactive
```

## Process A Video

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device auto
```

With an athlete name:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --fencer-name "John Smith"
```

Use CUDA if available:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device cuda
```

Use BiFenceNet:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --use-bifencenet
```

Load a model checkpoint:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --model weights/fencenet/model_best.pth
```

## Help

```bash
python app.py --help
```

## Outputs To Check

- `data/fencer_profiles/` for athlete profile JSON files.
- Processed video output if the selected workflow writes annotated video.
- The OpenCV preview window for live visual debugging.

## If Something Fails

- `ModuleNotFoundError: cv2`: install dependencies with `pip install -r requirements.txt`.
- CUDA unavailable: use `--device cpu` or install CUDA-compatible PyTorch.
- Video not opening: verify the file path and video format.
- Low FPS: use CPU/GPU settings appropriate to the machine and reduce video resolution if needed.
