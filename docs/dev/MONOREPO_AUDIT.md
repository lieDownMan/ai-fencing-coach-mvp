# Monorepo Audit

Ponytail audit of the current repo plus `feature/flutter-homepage-ui`.

## Ranked findings

- `delete:` tracked IDE and machine-local Android files. Replacement: nothing. `android/.idea/`, `android/app/.idea/`, `android/.kotlin/`, `android/app/local.properties`
- `delete:` tracked macOS archive junk. Replacement: nothing. `__MACOSX/`
- `delete:` local cache and generated folders from the repo root. Replacement: keep ignored only. `.pytest_cache/`, `.venv/`, `web_outputs/`, `Ultralytics/`, `__pycache__/`
- `shrink:` move the Python web app tree under one folder instead of leaving runtime files at repo root. Replacement: `apps/web/`. `app.py`, `database.py`, `llm_agent.py`, `inference/`, `src/`, `scripts/`, `weights/`
- `delete:` stale Android-side Python/Flutter bridge leftovers that do not belong in the native Android app anymore. Replacement: native Android runtime only. `android/app_main.py`, `android/flutter_main.dart`, `android/llm_agent.py`, `android/pose_estimator.py`, `android/coach_playbook.json`, `android/coach_playbook_en.json`
- `delete:` scratch file with no caller in the current tree. Replacement: nothing. `scratch_posebackend.kt`
- `yagni:` a second mobile app tree on `feature/flutter-homepage-ui` is not a standalone iOS app; it is one Flutter app with platform shells. Replacement: keep it together as one app until shared code is extracted. `frontend/pubspec.yaml`, `frontend/lib/`, `frontend/ios/`, `frontend/web/`
- `delete:` extra Flutter desktop targets if the product scope is only Android, iOS, and web. Replacement: nothing. `frontend/linux/`, `frontend/macos/`, `frontend/windows/`
- `delete:` generated Flutter build artifact committed inside source. Replacement: nothing. `frontend/lib/build/ios/SourcePackages/workspace-state.json`
- `shrink:` do not merge `feature/flutter-homepage-ui` wholesale into `main`. Replacement: cherry-pick the useful renames and app code because that branch predates the current paper docs and some Android updates.

net: big tree simplification possible, 0 new deps needed.

## What the branches tell us

- `main` has the newest Android work and the current paper/docs.
- `feature/android-parity` is effectively `main` plus docs, not a repo-shape branch.
- `feature/flutter-homepage-ui` is the only branch that already applies the useful backend rename from repo root into `backend/`.
- `feature/flutter-homepage-ui` also adds a full Flutter app under `frontend/`; the `ios/` folder there is only one platform wrapper around shared Flutter code in `frontend/lib/`.

## Smallest safe target

This is the lazy version that actually works:

```text
ai-fencing-coach-mvp/
  apps/
    web/
    flutter/
  android/
  docs/
  README.md
```

Why this shape:

- Keeping `android/` in place avoids touching the current strongest app during this pass.
- `apps/web` keeps the existing Gradio/browser app and Python inference stack together.
- The Flutter branch is one app, not separate `web` and `ios` apps yet.

## If you insist on `apps/web` and `apps/ios`

That is a second-phase refactor, not a cleanup pass.

You would first need to extract Flutter shared code into something like:

```text
packages/
  shared/
    flutter_app/
```

Then create thin wrappers for:

```text
apps/
  ios/
  web/
```

Without that extraction, moving `frontend/ios` by itself will break because it depends on `frontend/pubspec.yaml`, `frontend/lib/`, and shared assets/models.

## Recommended order

1. Clean tracked junk from `main`: Android IDE files, Kotlin error logs, `__MACOSX`, scratch files.
2. Rename the root Python stack into `apps/web/` on `main`.
3. Bring over the Flutter app from `feature/flutter-homepage-ui` as one unit under `apps/flutter/`.
4. Only after that, decide whether `apps/flutter/` is good enough or whether we should pay the cost to split it into `apps/ios` and `apps/web` plus `packages/shared/`.
