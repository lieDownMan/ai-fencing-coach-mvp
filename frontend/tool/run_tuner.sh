#!/usr/bin/env bash
# Launch the macOS video threshold-tuner (lib/tuner_main.dart).
# Prefills the cue-video folder (repo docs/) and the YOLO pose mlpackage path;
# both can be changed later in the app's 路徑設定 dialog.
set -euo pipefail

cd "$(dirname "$0")/.."          # frontend/
REPO="$(cd .. && pwd)"

exec flutter run -d macos -t lib/tuner_main.dart \
  --dart-define=TUNE_VIDEO_DIR="$REPO/docs" \
  --dart-define=TUNE_MODEL_PATH="$PWD/ios/Runner/yolov8n_pose.mlpackage"
