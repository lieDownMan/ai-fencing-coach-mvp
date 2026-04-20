# Checkpoint Format

This project can load PyTorch checkpoints for the FenceNet/BiFenceNet action-recognition stage.

Checkpoint loading is not the same as training. Training produces a checkpoint file; loading uses that file during video inference. This repository documents and validates the expected format, but it does not currently include trained fencing weights.

## Recommended Format

Use a dictionary with metadata plus a `state_dict`:

```python
{
    "format_version": 1,
    "model_type": "fencenet",  # or "bifencenet"
    "input_channels": 18,
    "num_classes": 6,
    "action_classes": ["R", "IS", "WW", "JS", "SF", "SB"],
    "state_dict": model.state_dict(),
}
```

The loader validates metadata when present:

- `format_version` must be `1`.
- `model_type` must match the selected CLI model type.
- `input_channels` must be `18` for the paper's 9-joint, 2D skeleton feature tensor.
- `num_classes` must be `6`.
- `action_classes` must be `["R", "IS", "WW", "JS", "SF", "SB"]`.

## Backward-Compatible Formats

The loader also accepts common PyTorch shapes without metadata:

```python
{"state_dict": model.state_dict()}
```

```python
{"model_state_dict": model.state_dict()}
```

```python
model.state_dict()
```

These formats can run, but they are less self-describing than the recommended format.

## CLI Usage

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --model weights/fencenet/model_best.pt --pose-backend mock
```

The CLI prints whether it is using checkpoint weights or random weights. Random weights only validate pipeline plumbing; they do not provide meaningful coaching labels.
