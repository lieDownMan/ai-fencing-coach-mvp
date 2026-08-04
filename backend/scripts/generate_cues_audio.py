#!/usr/bin/env python3
"""
Generate offline cue audio files (.m4a) from coach_playbook.json.
Uses macOS built-in 'say' command with the 'Mei-Jia' voice.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    playbook_path = project_root / "backend" / "coach_playbook.json"
    output_dir = project_root / "docs" / "cues_audio"

    if not playbook_path.exists():
        print(f"Error: Playbook not found at {playbook_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading cues from {playbook_path}...")
    with open(playbook_path, "r", encoding="utf-8") as f:
        playbook = json.load(f)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Generate each cue
    generated_count = 0
    for key, info in playbook.items():
        short_cue = info.get("short_cue")
        if not short_cue:
            print(f"Skipping key '{key}': 'short_cue' not found or empty.")
            continue

        error_name = info.get("error_name", key)
        output_file = output_dir / f"{key}.m4a"

        print(f"Generating audio for '{error_name}' ({key}):")
        print(f"  Text: \"{short_cue}\"")
        print(f"  File: {output_file.relative_to(project_root)}")

        try:
            # -v Mei-Jia: voice name
            # -o output_file: output destination path
            cmd = ["say", "-v", "Mei-Jia", "-o", str(output_file), short_cue]
            subprocess.run(cmd, check=True)
            print("  Status: Success\n")
            generated_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  Status: Failed to run say command: {e}\n", file=sys.stderr)
        except FileNotFoundError:
            print("  Status: Failed ('say' command not found. Are you on macOS?)\n", file=sys.stderr)

    print(f"Done! Successfully generated {generated_count} audio files.")

if __name__ == "__main__":
    main()
