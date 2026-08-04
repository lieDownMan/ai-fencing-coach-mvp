#!/usr/bin/env python3
"""Playwright driver for the AI Fencing Coach Clip Analysis UI (app.py).

app.py is a Gradio app with no native automation hooks, so this script
drives it the same way a human would: open a page, upload a video file
into the Gradio file-drop input, click "Run Analysis", wait for the
result panel to populate, and screenshot each step.

Usage (run app.py separately first, see SKILL.md):

    python driver.py sample-clip out.mp4
    python driver.py run --url http://127.0.0.1:7860 --video out.mp4 --out-dir ./shots
    python driver.py screenshot --url http://127.0.0.1:7860 --out ./shots/page.png
"""
import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def make_sample_clip(path: str, size=(320, 240), frame_count: int = 45, fps: float = 15.0) -> None:
    """Write a tiny synthetic mp4 (no real fencers) for smoke-testing the upload path."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), (index * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    print(f"wrote sample clip: {path}")


def screenshot(url: str, out: str, wait_ms: int = 1500) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_timeout(wait_ms)
        page.screenshot(path=str(out_path), full_page=True)
        browser.close()
    print(f"saved {out_path}")


def run_analysis(url: str, video: str, out_dir: str, timeout_ms: int = 90000) -> None:
    """Upload `video` on the Analysis tab, click Run Analysis, wait for the result, screenshot."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        page.goto(url)
        page.wait_for_selector("text=Run Analysis", timeout=timeout_ms)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(out / "01_initial.png"), full_page=True)

        page.locator("input[type=file]").first.set_input_files(video)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "02_uploaded.png"), full_page=True)

        page.get_by_role("button", name="Run Analysis").click()
        # Placeholder text is removed once the summary panel renders real content.
        page.wait_for_selector("text=Summary will appear here", state="detached", timeout=timeout_ms)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(out / "03_done.png"), full_page=True)

        browser.close()

    print(f"screenshots written to {out}")
    if errors:
        print("console errors:")
        for line in errors:
            print(f"  {line}")
    else:
        print("no console errors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample-clip", help="write a tiny synthetic mp4 for upload testing")
    p_sample.add_argument("path")

    p_shot = sub.add_parser("screenshot", help="load a URL and screenshot it")
    p_shot.add_argument("--url", default="http://127.0.0.1:7860")
    p_shot.add_argument("--out", required=True)

    p_run = sub.add_parser("run", help="upload a clip, run analysis, screenshot each step")
    p_run.add_argument("--url", default="http://127.0.0.1:7860")
    p_run.add_argument("--video", required=True)
    p_run.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    if args.command == "sample-clip":
        make_sample_clip(args.path)
    elif args.command == "screenshot":
        screenshot(args.url, args.out)
    elif args.command == "run":
        run_analysis(args.url, args.video, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
