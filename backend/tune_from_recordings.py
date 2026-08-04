"""Offline threshold tuner for PHONE-recorded sessions.

Because the live verdict is always rendered by the phone's own pose model
and camera, thresholds tuned against phone data need no cross-device
fudging — this script replays iPhone recordings through the same window
metrics the app uses (via tuning_server's line-for-line Dart mirror) and
recommends thresholds from the good-vs-error metric distributions.

Workflow:
 1. On the phone: Debug tab → 錄製調參數據 → do ONE kind of content per
    take (all-good en garde / one deliberate error), 10–20 s each.
 2. Files app ▸ AI Fencing Coach ▸ tuning/ → AirDrop the .jsonl files over.
 3. Label by renaming: the part before "__" is the label —
        good__enGarde.jsonl
        stance_too_high__take1.jsonl
        wide_step__luther.jsonl        (any error_key from the specs)
 4. Run:
        venv/bin/python backend/tune_from_recordings.py /path/to/recordings
        venv/bin/python backend/tune_from_recordings.py dir --target-side right
        venv/bin/python backend/tune_from_recordings.py --self-test

Output: per-error distribution table (good vs error windows), a
recommended threshold on the separation band (biased toward the good side
— false positives annoy ten times more than misses), and a ready-to-paste
HeuristicsConfig snippet. Overlapping distributions are flagged instead of
recommended: that means the metric itself can't separate the error, and no
threshold will fix it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tuning_server import (  # noqa: E402  (path setup above)
    DEFAULT_CONFIG,
    SPECS,
    compute_window_metrics,
)

WINDOW = 60   # frames per evaluation window — matches the app's buffer
STRIDE = 10   # frames between evaluations — matches the app


def load_recording(path):
    """Parse one SessionRecorder JSONL file → (skeletons, fps)."""
    skels, stamps = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        skels.append({k: (v[0], v[1]) for k, v in row["joints"].items()})
        stamps.append(row["t"])
    fps = 30.0
    if len(stamps) >= 2 and stamps[-1] > stamps[0]:
        fps = (len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1000.0)
    return skels, fps


def window_metrics(skels, fps, target_side, config):
    """Slide the app-sized window over a recording, yield metric dicts."""
    if len(skels) < WINDOW:
        # Short take: evaluate what we have rather than dropping the file.
        yield compute_window_metrics(skels, fps, target_side, config)
        return
    for start in range(0, len(skels) - WINDOW + 1, STRIDE):
        yield compute_window_metrics(
            skels[start:start + WINDOW], fps, target_side, config)


def pct(values, p):
    """Percentile with linear interpolation (matches numpy default)."""
    vals = sorted(values)
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def recommend(spec, good_vals, err_vals):
    """Threshold on the separation band, biased toward the good side.

    Returns (value, note). value is None when the distributions overlap —
    the metric can't separate this error and no threshold will fix it.
    """
    if spec["direction"] == "above":
        good_edge = pct(good_vals, 95)
        err_edge = pct(err_vals, 5)
        if err_edge <= good_edge:
            return None, "分佈重疊 — 這個指標分不開，調閾值救不了"
        return good_edge + 0.5 * (err_edge - good_edge), ""
    good_edge = pct(good_vals, 5)
    err_edge = pct(err_vals, 95)
    if err_edge >= good_edge:
        return None, "分佈重疊 — 這個指標分不開，調閾值救不了"
    return good_edge - 0.5 * (good_edge - err_edge), ""


def fmt(v, spec):
    return "--" if v is None else f"{v:.{spec['decimals']}f}"


def dist_row(label, vals, spec):
    return (f"    {label:<12} n={len(vals):<4} "
            f"p05={fmt(pct(vals, 5), spec)}  "
            f"p50={fmt(pct(vals, 50), spec)}  "
            f"p95={fmt(pct(vals, 95), spec)}")


def run(directory, target_side):
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        print(f"沒有 .jsonl 檔：{directory}")
        return 1

    # label → metric name → [values across all that label's windows]
    by_label = {}
    for path in files:
        label = path.name.split("__")[0]
        skels, fps = load_recording(path)
        if not skels:
            print(f"  (空檔跳過: {path.name})")
            continue
        bucket = by_label.setdefault(label, {})
        n_windows = 0
        for m in window_metrics(skels, fps, target_side, DEFAULT_CONFIG):
            n_windows += 1
            for k, v in m.items():
                bucket.setdefault(k, []).append(v)
        print(f"  {path.name}: {len(skels)} 幀 @ {fps:.1f}fps → "
              f"{n_windows} 個窗口 (標籤 {label})")

    good = by_label.get("good", {})
    if not good:
        print("\n⚠ 沒有 good__*.jsonl — 無法建立健康分佈，只列錯誤檔的數字。")

    tuned = {}
    print("\n" + "=" * 72)
    for spec in SPECS:
        key, metric, param = spec["error_key"], spec["metric"], spec["param"]
        err_vals = by_label.get(key, {}).get(metric, [])
        good_vals = good.get(metric, [])
        if not err_vals and not good_vals:
            continue
        print(f"\n{spec['label']}  ({metric} ↔ {param}, "
              f"{'大於' if spec['direction'] == 'above' else '小於'}觸發, "
              f"目前 {fmt(DEFAULT_CONFIG[param], spec)}{spec['unit']})")
        if good_vals:
            print(dist_row("good", good_vals, spec))
        if err_vals:
            print(dist_row(key, err_vals, spec))
        if good_vals and err_vals:
            value, note = recommend(spec, good_vals, err_vals)
            if value is None:
                print(f"    → {note}")
            else:
                tuned[param] = round(value, spec["decimals"])
                print(f"    → 建議閾值 {fmt(value, spec)}{spec['unit']}")
        elif err_vals:
            print("    → 缺 good 分佈，僅供參考")

    if tuned:
        print("\n" + "=" * 72)
        print("貼回 HeuristicsConfig（frontend/lib/heuristics/"
              "heuristics_engine.dart）的建議值：\n")
        for k, v in tuned.items():
            print(f"    this.{k} = {v},")
        print("\n（tuning_server.py 的 DEFAULT_CONFIG 也要同步同樣的值）")
    return 0


def self_test():
    """Synthetic good + stance_too_high recordings must separate cleanly."""
    import tempfile
    import time as _time

    def frame(t, straight):
        knee = [0.56, 0.65] if straight else [0.62, 0.65]
        return json.dumps({
            "t": t, "action": "SF", "conf": 0.9,
            "joints": {
                "left_hip": [0.48, 0.5], "right_hip": [0.52, 0.5],
                "right_knee": knee, "right_ankle": [0.60, 0.80],
                "left_ankle": [0.40, 0.80],
                "front_shoulder": [0.54, 0.35], "left_shoulder": [0.5, 0.35],
                "right_shoulder": [0.54, 0.35],
                "front_elbow": [0.64, 0.40], "front_wrist": [0.74, 0.45],
            },
        })

    t0 = int(_time.time() * 1000)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "good__synthetic.jsonl").write_text(
            "\n".join(frame(t0 + i * 33, False) for i in range(90)) + "\n")
        (td / "stance_too_high__synthetic.jsonl").write_text(
            "\n".join(frame(t0 + i * 33, True) for i in range(90)) + "\n")
        assert run(td, "left") == 0

    # bent knee ≈138.7°, straight ≈180° → must separate and recommend between
    good_v, err_v = [138.715], [180.0]
    spec = next(s for s in SPECS if s["error_key"] == "stance_too_high")
    value, note = recommend(spec, good_v, err_v)
    assert value is not None and 138.715 < value < 180.0, (value, note)
    print("\nself-test OK")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?",
                        help="放手機錄製 .jsonl 檔的資料夾")
    parser.add_argument("--target-side", default="left",
                        choices=["left", "right"])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.directory:
        parser.error("請給資料夾路徑，或用 --self-test")
    sys.exit(run(args.directory, args.target_side))


if __name__ == "__main__":
    main()
