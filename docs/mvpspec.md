# MVP Specification and Research Workflow

This is the canonical workflow and positioning spec for the AI Fencing Coach MVP. It consolidates the useful parts of the previous README, quickstart, detailed structure, and generated project summary.

## 1. Product Positioning

The MVP is a desktop coaching support tool for beginner and intermediate fencers. It analyzes side-view fencing video and provides feedback about distance, stance, and footwork patterns.

It should be framed as a coaching aid, not a referee replacement. The HCI contribution depends on whether the system changes what fencers and coaches can understand or do during practice and review.

## 2. Target Users

Primary users:

- Beginner and intermediate fencers who need feedback between coach interactions.
- Fencing coaches who want faster review of distance management and footwork habits.

Secondary users:

- HCI researchers studying AI-mediated sports feedback.
- ML/CV builders prototyping fencing motion-analysis pipelines.

Current evidence gap: the repo documents plausible users, but the next research step is direct observation or interviews with real fencers and coaches.

## 3. MVP Scope

Implemented in the current debugged code path:

- Native desktop workflow using Python and OpenCV.
- Imported video input through the CLI/app pipeline.
- Pose/keypoint extraction with explicit `mock`, `ultralytics`, and `auto` backend behavior.
- Opt-in real Ultralytics pose smoke coverage for `video/fencing_match.mp4` when `ultralytics` and a local YOLO pose model are available.
- Side-based two-fencer candidate tracking for visualization. The pipeline keeps the two largest pose candidates per frame and labels them `fencer_L`/`fencer_R` by horizontal center.
- Prototype distance feedback marks global `too_close` frames when front-ankle x-distance is less than `1.0x` average tracked fencer bounding-box height, then the annotated-video HUD also shows per-fencer distance status against each fencer's own detected height.
- Annotated MP4 output draws fencer boxes, skeleton keypoints, engagement-distance lines, dual left/right HUD panels, speed/movement cues, the current global action label, optional left/right height calibration, optional web-friendly downscaling plus H.264 transcoding, and the too-close warning banner.
- A local no-dependency browser demo lets reviewers process the sample/server-side video, watch the annotated MP4, and inspect summary metrics without composing terminal processing commands.
- Single selected fencer skeleton per frame remains the classifier input. For Ultralytics results, the largest detected person is selected.
- Spatial normalization into the paper's fixed 9-joint, 18-channel model feature tensor, with `nose` and `front_ankle` retained as normalization references.
- Six-class FenceNet/BiFenceNet footwork recognition using the CVPRW 2022 TCN architecture as the model-based path.
- Sliding-window inference over long videos instead of collapsing every clip to 28 frames.
- Pattern analysis for action frequencies, offensive/defensive ratio, JS/SF ratio, repeated patterns, and average confidence.
- LLM coaching interface with deterministic analytical fallback when no real LLM is loaded.
- Visual debugging UI rendering and headless-safe UI tests.
- Athlete profile storage for longitudinal analysis.
- Metadata-aware FenceNet/BiFenceNet checkpoint loading with explicit status for checkpoint vs random model weights.
- JSON report output for processed videos, including frame count, classification windows, action frequencies, confidence, runtime metadata, and feedback.

Planned or research-facing:

- Webcam/live mode beyond the current menu shell.
- Robust fencer identity persistence through crossings, occlusions, and exchange resets.
- Coach-validated distance thresholds, stance-width checks, and recovery/timing heuristics for lightweight coaching.
- Trained checkpoints for meaningful action recognition. The expected loading format is now documented, but trained weights are not included yet.
- Real LLM loading or API-backed generation.

Out of scope:

- Official refereeing or scoring.
- Blade contact detection.
- Full tactical understanding of all weapon actions.
- Multi-camera 3D reconstruction.
- Hosted production web deployment or mobile app distribution.
- Safety, medical, or certification claims.

## 4. Canonical Workflow

```text
1. Input
   Current path: imported bout video through the CLI/app pipeline.
   Planned path: webcam stream or imported bout video.

2. Pose estimation and tracking
   Current path: detect people, keep the two largest fencer candidates, and assign `fencer_L`/`fencer_R` by center X position while preserving one largest skeleton for classification.
   Planned path: add identity persistence through crossings, occlusion, and exchange resets.

3. Feature extraction
   Current path: export a fixed 9-joint skeleton tensor, keep `nose` and `front_ankle` as normalization references, and store fencer bounding boxes plus engagement distance for visualization. Optional left/right height inputs calibrate the annotated-video HUD only.
   Planned path: also extract stance width, limb/reach calibration, and stronger scale-normalized motion metrics.

4. Motion understanding
   Current path: classify normalized skeleton windows with FenceNet/BiFenceNet, show the current global action label in the annotated HUD, and flag prototype `too_close` frames from engagement distance.
   Planned path: add per-fencer action classification, coach-validated distance thresholds, stance-width checks, limb/reach-aware thresholds, and recovery/timing rules.

5. Pattern analysis
   Aggregate action frequencies, offensive/defensive ratios, JS/SF ratio, and repeated patterns.

6. Coaching feedback
   Current path: build prompts but use analytical fallback unless a real LLM backend is loaded.
   Planned path: generate concise live feedback, break strategy, or post-bout summary using the best available coaching backend.

7. Interface and persistence
   Current path: return CLI summaries, JSON reports with two-fencer tracking frames, write annotated MP4 review videos, expose a local browser demo for post-bout review, render the OpenCV dashboard when requested, and save athlete profile updates.
   Planned path: add upload support, progress updates, and stronger coach-facing review controls.
```

## 5. Core Data and State

| Item | Purpose |
| --- | --- |
| `fencer_L`, `fencer_R` | Tracked fencer detections and keypoints. |
| `avg_height` | Dynamic global scale reference based on bounding box height. |
| `left_height_cm`, `right_height_cm` | Optional annotated-video calibration inputs for per-fencer HUD distance display. |
| `engagement_dist` | X-axis distance between the opposing front ankles. |
| `stance_width` | Within-fencer front/back foot distance for form feedback. |
| `audio_cooldown` | Prevents repeated audio prompts in live mode. |
| athlete profile | Stores bout history and longitudinal metrics. |

Current implementation note:

- The current model input uses `front_wrist`, `front_elbow`, `front_shoulder`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, and `right_ankle`.
- `nose` and `front_ankle` are still required by normalization, but they are not exported as model feature channels.
- This keeps inference at `9 joints * 2 coordinates = 18 channels`.

## 6. Target Footwork Classes

| Code | Action | Category |
| --- | --- | --- |
| `R` | Rapid lunge | Offensive |
| `IS` | Incremental speed lunge | Offensive |
| `WW` | With waiting lunge | Offensive |
| `JS` | Jumping sliding lunge | Offensive |
| `SF` | Step forward | Neutral |
| `SB` | Step backward | Defensive |

## 7. Feedback Design

Use feedback only when it can support the user without overwhelming them.

Live feedback:

- Very short.
- Rate-limited.
- Focused on immediately actionable corrections such as distance or footwork form.

Break strategy:

- Summarizes observed tendencies.
- Suggests one tactical adjustment.
- Should be understandable without exposing model internals.

Post-bout summary:

- Connects this bout to prior profile data.
- Identifies one strength, one risk, and one next practice target.

## 8. HCI Research Framing

Working problem statement:

Beginner and intermediate fencers often get limited real-time feedback about distance and footwork when a coach is not watching every exchange. Existing workarounds include self-reviewing video, relying on occasional coach comments, or practicing without immediate feedback. The MVP explores whether computer vision and AI-generated coaching can make practice feedback more timely, understandable, and actionable.

Likely contribution type:

- Artifact contribution if the work centers on the interactive coaching system.
- Empirical contribution if the main result is a study of how fencers and coaches use or trust AI-generated practice feedback.

Gap hypothesis:

- Experience gap: fencers can already get feedback from coaches or video review, but the feedback is delayed, sparse, or hard to connect to specific movement moments.
- Possible cost gap: the system may offer some coaching feedback using commodity webcams instead of specialized sensors.

Danger-zone check:

- This project is not strong if the only claim is "we used AI for fencing." The stronger claim must be about the coaching experience: faster feedback, better reflection, changed practice decisions, or improved coach-student discussion.

## 9. Killer Scenarios

Scenario 1: Live distance correction

A beginner fencer practices footwork with a webcam pointed at the strip. The system highlights when the fencers collapse distance too early and gives a short prompt such as "recover before stepping in again." The coach can keep teaching while the system catches recurring distance-management lapses.

Scenario 2: Post-bout pattern review

After a recorded bout, the fencer loads the video and sees a summary showing that they overuse forward steps before attacking and rarely recover after lunges. The coach uses this to assign a specific drill for the next practice.

## 10. Evidence Needed Next

Before claiming this is a strong HCI contribution, collect:

- Two observations of beginner/intermediate fencers reviewing or practicing footwork.
- One coach interview about what feedback they wish students could receive when they are not watching.
- Three current alternatives or workarounds, such as coach review, video self-analysis, generic sports-analysis apps, or fencing-specific training tools.
- One low-fidelity feedback test: show annotated clips or Wizard-of-Oz prompts and ask whether the feedback is understandable and actionable.

## 11. Engineering Validation Status

Recent debug milestones:

- Pose backend behavior is explicit; mock mode is deterministic and real Ultralytics mode fails clearly when unavailable.
- Preprocessing and inference now agree on the paper's `18` model input channels.
- Long videos are preserved for sliding-window inference.
- Pattern statistics reset between videos and profile result accounting no longer treats `completed` as a loss.
- Coaching uses the pipeline's actual pattern statistics and falls back cleanly when no LLM is loaded.
- CLI/config handling and UI rendering are testable without opening windows.
- Checkpoint loading now accepts common PyTorch state-dict formats, validates optional metadata, and reports when the app is still using random weights.
- JSON report output can be written explicitly with `--report` or through `output.save_reports` and `output.reports_dir` in config.
- Two-fencer candidate tracking now records `fencer_L`/`fencer_R` side labels, per-frame centers/bounding boxes/keypoints, coverage, average front-ankle x-distance, and prototype `too_close` distance cues for reports and CLI summaries.
- Annotated video output can write a processed MP4 with fencer overlays, dual left/right HUD panels, per-fencer height-relative distance status, speed/movement cues, current global action label, optional `--annotated-max-width` downscaling plus H.264 transcoding, and a red `TOO CLOSE` banner when the distance heuristic triggers.
- Local browser demo output is available through `python web_app.py`; it reuses the same pipeline/report/annotator code and writes generated assets under `web_outputs/`.

Current local sample-video smoke:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id smoke_fencer --device cpu --pose-backend mock --profiles-dir /tmp/ai-fencing-coach-smoke-profiles
```

Observed result in the current environment:

- `video/fencing_match.mp4` opens with 776 frames at 30 FPS.
- Mock pose mode processes all 776 frames and records two side-based fencer candidates on each frame.
- Annotated-video mode writes `video/fencing_match_processed.mp4` or another requested output path with fencer overlays, dual HUDs, optional `--left-height-cm` / `--right-height-cm` calibration, optional `--annotated-max-width 1280` downscaling plus H.264 transcoding, and distance cues.
- Sliding-window inference emits 54 classifications.
- The post-bout feedback path completes using analytical fallback.
- The action labels are not semantically meaningful until trained model weights are provided.
- Real Ultralytics processing is available when `ultralytics` and a local YOLO pose model are installed; default tests keep this smoke path opt-in so the suite remains deterministic.

## 12. Further Work

Further work should be split into two tracks so the prototype can keep improving without overstating what the actual coaching system already does.

### 12.1 Current Prototype

These items improve the reliability, testability, and clarity of the current repo:

- Keep expanding the opt-in real `ultralytics` smoke path with pose-quality assertions and documented model/version combinations.
- Add or link a trained FenceNet/BiFenceNet checkpoint using the documented checkpoint format.
- Build a small labeled clip set so tests can check semantic action correctness, not only runtime shape and plumbing.
- Add pose-quality handling for low-confidence, missing, or intermittent skeleton frames.
- Add a lightweight CI profile that runs all deterministic tests while skipping local ignored media files when absent.
- Improve CLI output so users can tell whether they are running mock pose or real YOLO pose.
- Add a short developer note explaining that `mock` pose validates the system pipeline but does not validate pose accuracy.

### 12.2 Actual Coaching System

These items move beyond the current prototype toward a useful deployed or study-ready system:

- Upgrade side-based `fencer_L`/`fencer_R` candidate tracking into robust identity persistence across exchanges, crossings, and occlusions.
- Replace the prototype `too_close` ratio with coach-validated engagement-distance thresholds, then add limb/reach calibration, stance-width, recovery, and timing feedback grounded in fencing coaching concepts.
- Train or fine-tune action-recognition models on fencing-specific labeled data and evaluate them against held-out real bouts.
- Extend the local browser demo with video upload, progress/status updates, and coach-facing review UI that links feedback to specific moments in the video rather than only giving aggregate summaries.
- Add a real LLM backend or carefully designed prompt/API layer with coach-reviewed fallback templates and guardrails.
- Study feedback trust and usefulness with beginner/intermediate fencers and at least one coach before making strong HCI claims.
- Compare against realistic alternatives such as coach review, self-review video, and generic sports-analysis tools.
- Decide the deployment target: local desktop research artifact, coaching-club tool, or broader product prototype.

## 13. Document Ownership

- Keep this file as the canonical workflow/spec.
- Keep [README.md](README.md) as the entry point and navigation document.
- Keep [QUICKSTART.md](QUICKSTART.md) as commands only.
- Treat [detailedstructure.md](detailedstructure.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) as historical pointers unless the team decides to delete them later.
