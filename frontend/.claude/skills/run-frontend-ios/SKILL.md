---
name: run-frontend-ios
description: Build, run, and screenshot the AI Fencing Coach iPhone app (Flutter) on the iOS Simulator. Use when asked to run the iOS app, start the Flutter frontend, build for iOS, boot the simulator, or screenshot the fencing coach mobile UI.
---

This is the on-device iPhone port of the AI Fencing Coach: a Flutter app (`frontend/`) with native
Swift/CoreML bridges (`ios/Runner/FenceNetBridge.swift`, `YoloPoseBridge.swift`) for on-device pose
estimation and action classification. Drive it via
`.claude/skills/run-frontend-ios/driver.sh`, a thin wrapper around `flutter` + `xcrun simctl` that
boots a simulator, builds, launches, and screenshots the app. All paths below are relative to
`frontend/` (the app root), except the driver path itself which is relative to `frontend/`.

## Prerequisites

macOS with Xcode (a **full** Xcode install, not just Command Line Tools — `xcodebuild -version` must
work, not just `xcode-select -p`). Verified on macOS 26.2, Apple Silicon, Xcode 26.6.

```bash
brew install --cask flutter   # Flutter SDK (verified: 3.44.7)
brew install cocoapods        # verified: 1.17.0
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadPlatform iOS   # downloads the iOS Simulator runtime (verified: iOS 26.5)
```

If Xcode.app itself isn't installed yet, that part needs a human: it requires Apple ID sign-in
(and likely 2FA), which can't be scripted. Install via the App Store app, then run the two `sudo`
lines above (also can't be scripted — they need a password prompt on a real TTY).

## Setup

```bash
cd frontend
flutter pub get
```

No API keys are required to build and launch. `GEMINI_API_KEY` (optional, for `GeminiAgent` postgame
summaries) would go in `frontend/.env` if used — not needed to verify the app runs.

## Build

```bash
cd frontend
flutter build ios --no-codesign --debug
```

This must succeed before `flutter run` will work. If it fails on `coremlc`/`.mlpackage` errors, see
Gotchas below — that was a real, pre-existing repo issue, not a flake.

## Run (agent path)

```bash
cd frontend
UDID=$(.claude/skills/run-frontend-ios/driver.sh boot)
.claude/skills/run-frontend-ios/driver.sh build
.claude/skills/run-frontend-ios/driver.sh run "$UDID"
.claude/skills/run-frontend-ios/driver.sh screenshot "$UDID" /tmp/shots/live.png
```

| driver.sh command | what it does |
|---|---|
| `boot` | finds or creates a simulator named "AI Fencing Coach Test" (iPhone 16 / iOS 26.5), boots it, prints its UDID |
| `build` | `flutter pub get` + `flutter build ios --no-codesign --debug` |
| `run <udid>` | backgrounds `flutter run -d <udid> --debug`, polls the log (`/tmp/flutter_run_ios.log`) until the Dart VM Service line appears (cold start took ~60-90s in verification, mostly Xcode build time) |
| `screenshot <udid> <path>` | `xcrun simctl io <udid> screenshot <path>` |
| `record <udid> <path.mp4> [seconds]` | records the simulator screen (h264, default 10s) while the app runs — `simctl io recordVideo` in the background, stopped with SIGINT |
| `shutdown <udid>` | `xcrun simctl shutdown <udid>` |

Stop the running app: `pkill -f "flutter run -d <udid>"` then `driver.sh shutdown <udid>`.

Verified real screenshot output: the Live tab correctly shows "Initializing Camera" / IDLE — this is
**expected**, not a bug: iOS Simulators have no real camera, so `availableCameras()` (or the
subsequent `CameraController.initialize()`) never completes. Everything else (tab bar, app chrome,
model-loaded badges) renders normally.

## Deeper interaction (tapping through tabs) — use `integration_test`, not OS-level clicks

I tried driving taps via `cliclick` and AppleScript's `System Events... click at {x,y}` to navigate
to the Postgame tab (no camera needed there — it's an `image_picker` + a mocked "Run Analysis" delay,
see `lib/screens/postgame_screen.dart`). Both failed, but not because of the app: in this specific
hosting environment, an overlay window belonging to the IDE/browser session itself
(a "Dia" process with no enumerable position/size — consistent with a full-screen or high
window-level overlay) intercepted every synthetic click regardless of which screen coordinates were
targeted, even after granting both Accessibility and Input Monitoring permissions and relocating the
Simulator window. This looks specific to automating clicks *from inside* an agent session hosted by
that same IDE, not a property of the Flutter app.

**Don't chase OS-level click automation for this app.** Flutter's own `integration_test` package
drives the widget tree directly through the Dart VM Service — no synthetic mouse/keyboard events, no
window-manager/overlay interference, and it's the standard tool for this exact job. It isn't set up
in this repo yet; adding it (a `integration_test/` dir + `flutter test integration_test/app_test.dart
-d <device>`) is the right next step for anyone who needs to script taps through Postgame/Settings,
rather than more OS permission grants.

## Run (human path)

```bash
cd frontend
open -a Simulator
flutter run
```
Pick the booted simulator if prompted. `r` hot-reloads, `q` quits.

## Test

```bash
cd frontend
flutter analyze   # 0 errors (pre-existing lint warnings/info only)
flutter test      # 1 passed — checks FencingCoachApp renders its tab bar
```

## Gotchas

- **`ffmpeg_kit_flutter` is a dead dependency.** It was declared in `pubspec.yaml` but never imported
  anywhere in `lib/`, and its pinned GitHub release (`arthenica/ffmpeg-kit` v6.0 iOS xcframework) 404s
  — upstream took down release binaries. This broke `pod install` outright. Already removed on this
  branch; if it reappears (e.g. from a merge), remove it again rather than trying to fix the URL.
- **`.gitignore` had a case-insensitive collision that silently dropped every CoreML model's weight
  data.** A bare `data/` rule (meant for a root-level training-data folder) also matched
  `*.mlpackage/Data/` on macOS's case-insensitive filesystem, so only `Manifest.json` ever got
  committed for any `.mlpackage` — the actual weight binaries were never in git. `coremlc` fails at
  Xcode build time without them (`Failed to read model package... Item does not exist for identifier`).
  Fixed by anchoring the rule to `/data/`. If you ever see that build error again, check
  `git check-ignore -v <path>/Data/...` before assuming the export script is broken.
- **Missing CoreML models can be regenerated from source** — don't hand-roll a converter.
  `backend/scripts/export_coreml.py` (needs `backend/weights/fencenet/best_model.pth`, already in
  git) and `backend/scripts/export_yolo_coreml.py` (needs `yolov8n-pose.pt` at repo root — gitignored,
  auto-downloads via `ultralytics` if missing) regenerate
  `frontend/ios/Runner/{fencenet_v2,yolov8n_pose}.mlpackage` and
  `frontend/assets/models/fencenet_v2.mlpackage` exactly. Needs `coremltools` and `ultralytics` in a
  Python venv (see the repo-root Python skill, `.claude/skills/run-ai-fencing-coach-mvp/`, for that
  venv setup) — this is a Python step, not a Flutter one.
- **`pubspec.yaml` was missing `uuid`, `intl`, `path`** even though `lib/main.dart`,
  `lib/screens/history_screen.dart`, and `lib/database/app_database.dart` import them. `flutter pub
  get` succeeded anyway (transitive resolution), but `flutter analyze`/`flutter test` failed with
  `uri_does_not_exist`. Fixed via `flutter pub add uuid intl path`.
- **`test/widget_test.dart` was unmodified `flutter create` boilerplate** referencing a `MyApp` class
  that was never part of this app (the real entry widget is `FencingCoachApp`). Rewrote it to check
  the app actually renders its tab bar.
- **The iOS Simulator has no camera.** `availableCameras()` / `CameraController.initialize()` just
  never completes, so the Live tab sits on "Initializing Camera" forever. This is correct behavior for
  a simulator, not something to fix — it's exactly why a real iPhone is needed to validate the Live
  coaching feature (see below).

## Real-device testing note

For anything about the *feel* of live coaching — camera framerate/distance, CoreML/Neural Engine
inference latency, TTS through the earpiece with the silent switch handling
(`setIosAudioCategory(.playback)` in `main.dart`) — use a real iPhone, not the Simulator. The
Simulator has no Neural Engine (CoreML falls back to CPU/GPU only) and Simulator's own webcam
passthrough (Xcode's I/O > Camera menu) doesn't replicate real fencing-distance framing. Use the
Simulator for fast UI/logic iteration; validate the actual coaching experience on-device.

## Troubleshooting

- **`Error (Xcode): Failed to read model package at file://.../yolov8n_pose.mlpackage/. Error: Item
  does not exist for identifier: ...`**: the `.mlpackage`'s `Data/` folder is missing or was silently
  gitignored (see Gotchas). Regenerate via the export scripts above, and verify with
  `git check-ignore -v frontend/ios/Runner/yolov8n_pose.mlpackage/Data/com.apple.CoreML/weights/weight.bin`
  (should exit 1 / print nothing).
- **`pod install` fails with `curl: (56) ... 404` on `ffmpeg-kit-https-*-ios-xcframework.zip`**: dead
  dependency, see Gotchas — remove `ffmpeg_kit_flutter` from `pubspec.yaml` and `flutter pub get`
  again (also delete `ios/Pods`, `ios/Podfile.lock`, `ios/.symlinks` first for a clean retry).
- **`xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory is
  '/Library/Developer/CommandLineTools'`**: full Xcode isn't installed or isn't selected — install it
  from the App Store, then run the two `sudo xcode-select`/`xcodebuild -runFirstLaunch` commands above.
