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

In scope:

- Native desktop workflow using Python and OpenCV.
- Webcam or imported video input.
- Two-fencer tracking in a side-view fencing scene.
- Left/right fencer assignment by horizontal position.
- Pose/keypoint extraction.
- Dynamic distance and stance heuristics for lightweight coaching.
- Six-class FenceNet/BiFenceNet footwork recognition as the model-based path.
- Visual debugging UI with skeletons, boxes, labels, metrics, and feedback.
- Athlete profile storage for longitudinal analysis.

Out of scope:

- Official refereeing or scoring.
- Blade contact detection.
- Full tactical understanding of all weapon actions.
- Multi-camera 3D reconstruction.
- Web/mobile deployment.
- Safety, medical, or certification claims.

## 4. Canonical Workflow

```text
1. Input
   Webcam stream or imported bout video.

2. Pose estimation and tracking
   Detect people, keep the two largest fencer candidates, and assign Fencer L/R by center X position.

3. Feature extraction
   Extract keypoints, bounding boxes, front ankles, stance width, and engagement distance.

4. Motion understanding
   MVP path: apply dynamic rules such as "too close" and stance-width checks.
   Model path: normalize skeleton sequences and classify footwork with FenceNet/BiFenceNet.

5. Pattern analysis
   Aggregate action frequencies, offensive/defensive ratios, JS/SF ratio, and repeated patterns.

6. Coaching feedback
   Generate concise live feedback, break strategy, or post-bout summary.

7. Interface and persistence
   Show annotated video, feedback, and metrics. Save athlete profile updates where relevant.
```

## 5. Core Data and State

| Item | Purpose |
| --- | --- |
| `fencer_L`, `fencer_R` | Tracked fencer detections and keypoints. |
| `avg_height` | Dynamic scale reference based on bounding box height. |
| `engagement_dist` | X-axis distance between the opposing front ankles. |
| `stance_width` | Within-fencer front/back foot distance for form feedback. |
| `audio_cooldown` | Prevents repeated audio prompts in live mode. |
| athlete profile | Stores bout history and longitudinal metrics. |

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

## 11. Document Ownership

- Keep this file as the canonical workflow/spec.
- Keep [README.md](README.md) as the entry point and navigation document.
- Keep [QUICKSTART.md](QUICKSTART.md) as commands only.
- Treat [detailedstructure.md](detailedstructure.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) as historical pointers unless the team decides to delete them later.
