# Related Work Memo: AI-Assisted Fencing Coaching

Status: Working memo
Last updated: 2026-04-20
Purpose: Preserve the current related-work scan, comparison framing, and positioning notes for later paper/proposal writing.

## 1. Project Filter: Define "Us" First

Use this 5-line brief as the screening filter for related work:

- Target users: fencers, coaches, and learners.
- Problem: AI-assisted fencing coaching for skill learning and technique refinement.
- Modality: commodity video + pose estimation + action recognition + interactive feedback.
- Contribution: fencing-specific coaching support from ordinary camera video rather than expensive lab setups.
- HCI angle: actionable, interpretable feedback that helps users improve, not just recognize actions.

Working screening rules:

- Papers that only do recognition, without coaching or feedback, are relevant but should not dominate the final framing.
- Strong HCI coaching papers from other sports are still important because they help explain how AI becomes usable feedback.
- Commercial products and academic literature should be discussed separately so workflow tools are not treated as scientific evidence.

## 2. Candidate Pool

This round produced 22 high-relevance candidates. The pool is smaller than a 40-60 paper sweep, but it is strong enough for a first high-quality screening pass.

### Bucket A: Fencing-Specific Candidates

- Real-Time Action Detection and Analysis in Fencing Footwork
- Recognition of Action Dynamics in Fencing Using Multimodal Cues
- Visualization of Technical and Tactical Characteristics in Fencing
- FenceNet: Fine-Grained Footwork Recognition in Fencing
- Temporal Segmentation of Actions in Fencing Footwork Training
- A fencing action recognition algorithm based on keypoint detection
- VirtualFencer
- Automatic Assessment of Skill and Performance in Fencing Footwork

### Bucket B: Sports Coaching / HCI / Motor Learning Candidates

- AI Coach: Deep Human Pose Estimation and Analysis for Personalized Athletic Training Assistance
- Pose Estimation for Facilitating Movement Learning from Online Videos
- Visualizing Instructions for Physical Training: Exploring Visual Cues to Support Movement Learning from Instructional Videos
- Bridging Coaching Knowledge and AI Feedback to Enhance Motor Learning in Basketball Shooting Mechanics Through a Knowledge-Based SOP Framework
- Train Me: Exploring Mobile Sports Capture and Replay for Immersive Self-Training
- Intercorporeal Biofeedback for Movement Learning
- Designing a Technique-Oriented Sport Training Game for Runners

### Bucket C: Adjacent Sports / Similar Methods Candidates

- Exploration of Applying Pose Estimation Techniques in Table Tennis
- Player Performance Analysis in Table Tennis Through Human Action Recognition
- Table tennis coaching system based on a multimodal large language model with a table tennis knowledge base
- Generating Tennis Action Instruction Based on a Large Language Model
- Enhancing Remote Dance Education Through Motion Capture and Web-Based Feedback Using Expert Reference Sequences
- SwingVision validity / camera-position studies

### Bucket D: Method / Survey Backdrop

- A Systematic Review of the Application of Camera-Based Human Pose Estimation in the Field of Sport and Physical Exercise
- Deep learning-based human body pose estimation in providing feedback for physical movement: a review

Notes:

- These survey papers are useful for the Introduction or methods background.
- They are less central as direct comparison targets in the final Related Work section.

## 3. Seed Papers for Citation Chasing

First-round seed set:

- FenceNet
- Temporal Segmentation of Actions in Fencing Footwork Training
- Visualization of Technical and Tactical Characteristics in Fencing
- Pose Estimation for Facilitating Movement Learning from Online Videos
- Bridging Coaching Knowledge and AI Feedback...
- Table tennis coaching system based on a multimodal LLM...

Why these seeds:

- They cover the three-way intersection of the project: fencing-specific analysis, HCI / movement learning, and AI-generated coaching feedback.
- They align directly with the current gap hypothesis:
  - fencing work leans toward recognition or analysis
  - HCI coaching work often targets general sports or exercise
  - LLM-based coaching work is emerging in table tennis, basketball, and tennis, not fencing

Citation-chasing directions:

- From FenceNet and Temporal Segmentation, follow the shared authors, dataset lineage, and fencing-footwork pipeline literature.
- From Pose Estimation for Facilitating Movement Learning and Visualizing Instructions, follow the HCI literature on feedback representation and motor-learning support.
- From Bridging..., TAGS, and the table tennis multimodal coaching system, follow 2024-2026 work on LLM-in-sports-coaching.

## 4. Screening Logic

Criteria used to keep papers in the stronger set:

- directly handles fencing action, tactics, training analysis, or skill assessment
- explicitly supports coaching, feedback, motor learning, or athlete support
- uses methods close to the current system, such as video, pose, HAR, multimodal sensing, or LLM feedback
- represents an important cluster rather than being a weaker duplicate

Papers kept in Zotero but deprioritized for the top 10:

- A Systematic Review of Camera-Based HPE in Sport
- SwingVision validity papers
- Exploration of Applying Pose Estimation Techniques in Table Tennis
- Player Performance Analysis in Table Tennis Through HAR
- Train Me
- Enhancing Remote Dance Education Through Motion Capture and Web-Based Feedback

Reasons for deprioritization:

- useful as backdrop, not the strongest comparison target
- method-adjacent but weak on coaching
- strong HCI but farther from the core AI-assisted fencing coaching contribution

## 5. Final Top-10 Shortlist

Final shortlist shape:

- 4 fencing-specific papers
- 4 sports/HCI/coaching papers
- 2 adjacent AI-feedback papers

Final top 10:

1. FenceNet
2. Recognition of Action Dynamics in Fencing Using Multimodal Cues
3. Temporal Segmentation of Actions in Fencing Footwork Training
4. Visualization of Technical and Tactical Characteristics in Fencing
5. AI Coach
6. Pose Estimation for Facilitating Movement Learning from Online Videos
7. Visualizing Instructions for Physical Training
8. Bridging Coaching Knowledge and AI Feedback...
9. Table tennis coaching system based on a multimodal LLM...
10. Generating Tennis Action Instruction Based on a Large Language Model (TAGS)

## 6. Ten-Paper Comparison Table

| Paper | Year | Domain / Sport | Research Goal | Method + Sensing | Supports Coaching / Feedback / Assessment / Tactics? | HCI / User Study? | Why Relevant to AI-Assisted Fencing Coaching | Key Limitation | Category | Difference From Us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FenceNet: Fine-Grained Footwork Recognition in Fencing | 2022 | Fencing | Recognize fine-grained fencing footwork from ordinary videos | Skeleton-based action recognition with TCNs over 2D pose | Assessment / motion analysis | No user study; technical evaluation | Very close to the current commodity-video + pose pipeline; strong baseline for fencing technique recognition | Recognition only; no interactive coaching layer | Different problem, similar approach | Similar input pipeline, but it stops at footwork classification instead of actionable coaching feedback. |
| Recognition of Action Dynamics in Fencing Using Multimodal Cues | 2018 | Fencing | Distinguish subtly different fencing actions by modeling motion dynamics | Multimodal features and dynamic descriptors | Assessment / technique recognition | No HCI study | Strong evidence that fencing coaching systems should model temporal dynamics, not just static pose | Offline recognition rather than a coaching interface | Different problem, similar approach | Similar recognition backbone idea, but not designed for interpretable learner-facing feedback. |
| Temporal Segmentation of Actions in Fencing Footwork Training | 2023 | Fencing | Segment continuous footwork training into actions for usable training analysis | Temporal segmentation + per-frame classification | Assessment / training support | No formal HCI study | Relevant because real coaching needs continuous-session parsing, not just isolated clips | Strong on segmentation, weaker on pedagogical explanation | Same/similar problem, different approach | Closer to training support, but still mainly infrastructure for analysis rather than an interactive AI coach. |
| Visualization of Technical and Tactical Characteristics in Fencing | 2019 | Fencing | Help experts analyze fencing tactics and technical patterns | Interactive visualization system with coordinated views | Tactic analysis / analyst support | Yes; expert feedback and case studies | Strong precedent for explainable fencing interfaces and analysis-to-insight workflows | Geared more to analysts than learners; not automated coaching | Same/similar problem, different approach | Fencing-specific and explainable, but focused on tactical visualization rather than camera-based personal coaching. |
| AI Coach: Deep Human Pose Estimation and Analysis for Personalized Athletic Training Assistance | 2019 | General athletic training | Deliver personalized athletic training assistance from pose analysis | Deep HPE + movement analysis from video | Coaching / feedback / assessment | Yes; reported user studies | Early direct example of pose estimation leading to personalized coaching | Not fencing-specific; likely broader and less interpretable than the target system | Same/similar problem, different approach | Very close in spirit, but not built around fencing-specific technique vocabulary or explainable feedback for coaches and fencers. |
| Pose Estimation for Facilitating Movement Learning from Online Videos | 2020 | General movement / exercise | Improve movement learning from tutorial videos | Trainer video + webcam + pose overlays | Coaching / feedback / skill acquisition | Yes; user study | Highly relevant for feedback presentation design in the interface | Not sport-specific; feedback is visual rather than domain-semantic | Same/similar problem, different approach | Similar goal of helping learners self-correct, but in generic movement-learning settings rather than fencing coaching. |
| Visualizing Instructions for Physical Training: Exploring Visual Cues to Support Movement Learning from Instructional Videos | 2022 | Physical training / workout | Study how visual cues support movement imitation and learning | Augmented video with directional cues, highlights, and metaphorical visualizations | Coaching / feedback / skill acquisition | Yes | Strong HCI evidence that feedback representation matters, not just recognition accuracy | Does not solve fencing recognition or tactic understanding | Different problem, similar approach | Similar concern with movement learning and feedback design, but targets generic physical training rather than fencing-specific coaching. |
| Bridging Coaching Knowledge and AI Feedback to Enhance Motor Learning in Basketball Shooting Mechanics Through a Knowledge-Based SOP Framework | 2025 | Basketball | Turn coaching knowledge into actionable AI feedback for motor learning | Coach-informed knowledge framework with AI feedback on mechanics | Coaching / feedback / motor learning | Very likely yes; CHI-style system framing | Probably the closest HCI precedent for AI + coach knowledge + actionable feedback | Not fencing-specific and not centered on continuous bout/tactic context | Same/similar problem, different approach | The strongest HCI comparison, but it addresses basketball shooting rather than fencing actions and bout-specific coaching. |
| Table tennis coaching system based on a multimodal large language model with a table tennis knowledge base | 2025 | Table tennis | Build a low-cost AI coach for beginners | Smartphone video, ball trajectory recognition, OpenPose, multimodal LLM, table-tennis knowledge base | Coaching / feedback / assessment / tactics | Yes; expert evaluation | Strong adjacent-sport evidence for knowledge-grounded multimodal AI coaching | Performance on footwork is weaker; beginner-focused; table-tennis-specific knowledge base | Same/similar problem, different approach | A strong analogue for multimodal AI + domain knowledge + coaching output, but in racket sport rather than fencing. |
| Generating Tennis Action Instruction Based on a Large Language Model (TAGS) | 2025 | Tennis | Generate expert-like action instructions from tennis video | 3D skeletal keypoints + pose-language representation + prompt-based LLM | Coaching / feedback / assessment | No clear in-situ HCI study | Excellent adjacent method for explainable, queryable coaching feedback from motion | Workshop paper; tennis-specific; limited evidence on real coaching use | Different problem, similar approach | Similar explainable instruction-generation direction, but not fencing-specific and likely more model-centric than interface-centric. |

## 7. Backup Papers to Keep Nearby

- Train Me
- Exploration of Applying Pose Estimation Techniques in Table Tennis
- Player Performance Analysis in Table Tennis Through Human Action Recognition
- Enhancing Remote Dance Education Through Motion Capture and Web-Based Feedback Using Expert Reference Sequences

Why keep them:

- useful if the framing needs more HCI self-training or remote-feedback evidence
- useful as method adjacency if reviewers ask for broader pose/HAR context

## 8. Objective "Difference From Us" Statements

- FenceNet is close in using commodity video and pose, but it focuses on footwork recognition rather than interactive coaching or interpretable learner feedback.
- Recognition of Action Dynamics in Fencing Using Multimodal Cues shows why temporal dynamics matter in fencing, but it remains an offline recognition pipeline rather than a coaching system.
- Temporal Segmentation of Actions in Fencing Footwork Training is more training-oriented, yet it mainly provides segmentation and classification infrastructure rather than user-facing coaching feedback.
- FencingVis is explainable and fencing-specific, but it supports analysts' tactical interpretation more than novice/intermediate skill learning from camera video.
- AI Coach is similar in spirit as pose-based personalized training, but it is not fencing-specific and does not appear to encode fencing coaching knowledge.
- Pose Estimation for Facilitating Movement Learning from Online Videos is strong evidence for effective visual feedback design, but it targets generic online-video learning rather than fencing technique coaching.
- Visualizing Instructions for Physical Training informs how to present cues, but it does not solve fencing action understanding or tactic-aware instruction.
- Bridging Coaching Knowledge and AI Feedback... is probably the closest HCI comparison because it converts coach knowledge into AI feedback, but it is built around basketball shooting rather than fencing actions and bouts.
- The table-tennis multimodal LLM coach is methodologically very close, but it is optimized for beginner racket-sport errors and still struggles with footwork.
- TAGS is close on explainable instruction generation from motion, but it is a tennis workshop paper with less evidence so far on in-situ coaching use.

## 9. Commercial Landscape

| Product | Domain | Hardware Burden | Video / Pose / AI Automation | Feedback Type | Coach-Facing or Athlete-Facing? | Fencing-Specific? |
| --- | --- | --- | --- | --- | --- | --- |
| Onform | Multi-sport video coaching | Low: phone/tablet/computer | Video review, annotations, some AI-powered / 3D analysis features | Slow motion, drawing, sharing, remote review | Both, especially coach workflow | No |
| SwingVision | Tennis / pickleball | Low: single camera/mobile | Automated scoring, stats, highlights, line calling, form evaluation | Match analytics + AI coaching | Both | No |
| HomeCourt | Basketball | Low: iPhone / iPad | AI shot tracking, interactive drills, analytics | Guided workouts, gamified practice, analytics | More athlete-facing, still coach-usable | No |
| CoachNow | Multi-sport coaching platform | Low: phone video capture/upload | Video/image analysis, async coaching, AI analysis marketing, skeleton tracking | Annotated clips, voice-over, communication, progress tracking | Both | No |
| Dartfish | Multi-sport performance analysis | Low-to-medium depending on setup | Video and data analysis for motion/game analysis | Technical and tactical review, instant feedback, replay workflows | Coach / analyst oriented | No |
| Sportsbox AI | Golf / sports-fitness | Low: single mobile slow-motion video | 2D-to-3D motion analysis, metrics, corrective feedback | 3D analysis + remote coaching notes | Both, coach-heavy | No |

Commercial summary:

- No mature product was found in this scan that clearly markets fencing-specific AI coaching.
- Existing products cluster around multi-sport video review, single-sport analytics in other domains, or general remote-coaching workflows.
- This supports the positioning that the current project is not just another generic video-review app, but a potential combination of fencing specificity and actionable coaching.
- This conclusion should be treated as "not found in this round of scanning," not as an absolute claim that no such product exists anywhere.

## 10. Competitive Positioning Quadrant

Recommended axes:

- X-axis: domain specificity, from general sports analysis to fencing-specific coaching
- Y-axis: feedback actionability, from descriptive analysis to interactive / prescriptive coaching

High actionability
^
|                         Our system
|               Bridging..., Table-tennis LLM coach, TAGS
|        HomeCourt, SwingVision, CoachNow, Onform
|
|                 FencingVis
|    Pose-learning / visual-cue HCI papers
|
|  FenceNet, Fencing action dynamics, Temporal segmentation
+------------------------------------------------------------> High fencing specificity

Interpretation:

- High specificity, low-to-medium actionability:
  - FenceNet
  - Recognition of Action Dynamics...
  - Temporal Segmentation...
  These are fencing-specific, but mainly recognition or segmentation systems.
- High specificity, medium actionability:
  - FencingVis
  It helps experts interpret tactics, but it is not a direct AI coach for novice/intermediate skill learning.
- Low specificity, medium-to-high actionability:
  - Onform
  - CoachNow
  - Dartfish
  - HomeCourt
  - SwingVision
  These are strong on workflow, analytics, or guided practice, but not fencing-specific.
- Medium specificity, high actionability:
  - Bridging...
  - table-tennis multimodal LLM coach
  - TAGS
  These are the closest to the desired "AI becomes coach-like feedback" framing, but they are not about fencing.

## 11. Final Related Work Structure

### A. Same / Similar Problem, Different Approach

#### A1. Fencing-Specific Systems and Analysis

- FenceNet
- Recognition of Action Dynamics in Fencing Using Multimodal Cues
- Temporal Segmentation of Actions in Fencing Footwork Training
- Visualization of Technical and Tactical Characteristics in Fencing

#### A2. Sports Coaching and HCI Systems

- AI Coach
- Pose Estimation for Facilitating Movement Learning from Online Videos
- Visualizing Instructions for Physical Training
- Bridging Coaching Knowledge and AI Feedback...

### B. Different Problem, Similar Approach

#### B1. Similar AI Approaches in Adjacent Domains

- Table tennis coaching system based on a multimodal LLM...
- Generating Tennis Action Instruction Based on a Large Language Model
- Backup: table-tennis pose / HAR papers, remote dance feedback systems

#### B2. Commercial Products and Competitive Landscape

- Onform
- SwingVision
- HomeCourt
- CoachNow
- Dartfish
- Sportsbox AI

## 12. Concise Related Work Draft

Prior work relevant to AI-assisted fencing coaching falls into two broad groups. The first group addresses the same or a nearby problem but with a different emphasis: fencing-specific papers such as FenceNet, Recognition of Action Dynamics in Fencing Using Multimodal Cues, and Temporal Segmentation of Actions in Fencing Footwork Training focus on recognition, dynamics modeling, or segmentation, while FencingVis supports expert tactical interpretation through interactive visualization. Together, these papers show that fencing actions can be recognized and tactically analyzed, but they stop short of delivering actionable learner-facing coaching from ordinary video. The second group addresses a different sport but a similar approach: HCI and sports-coaching systems such as AI Coach, Pose Estimation for Facilitating Movement Learning from Online Videos, Visualizing Instructions for Physical Training, and Bridging Coaching Knowledge and AI Feedback... demonstrate how pose estimation, feedback presentation, and coach knowledge can improve motor learning. Recent adjacent-sport systems in table tennis and tennis further show that multimodal LLMs can turn motion data into targeted technical advice. However, these systems are not fencing-specific, and commercial products such as Onform, SwingVision, HomeCourt, CoachNow, Dartfish, and Sportsbox AI still emphasize general video analysis, sport-specific analytics in other domains, or generic remote-coaching workflows rather than fencing-specific AI coaching. Our work therefore sits at the intersection: it aims to combine fencing specificity, commodity-camera accessibility, and actionable, interpretable coaching feedback in one system.

## 13. One-Paragraph Gap Statement

The clearest gap is that fencing-specific work already exists, but it is concentrated on recognition, segmentation, and tactical visualization rather than learner-facing AI coaching; meanwhile, sports-HCI and adjacent-sport AI papers already show how pose, multimodal sensing, and LLMs can support coaching and feedback, but they do not target fencing. Commercial tools prove that coaches value video review, asynchronous feedback, and low-hardware workflows, yet they still do not provide fencing-specific AI guidance. That leaves a credible opening for a system that uses commodity video to understand fencing movement and then returns interpretable, coach-like feedback for skill acquisition.

## 14. Usage Notes

Before promoting this memo into a formal paper section:

- manually verify metadata, venues, and phrasing against the full papers
- read the Abstract and Introduction carefully without AI paraphrasing
- keep academic work and commercial products in separate argumentative roles
- avoid overclaiming that no fencing-specific competitor exists unless the market scan is broadened again
- sharpen "actionable, interpretable coaching feedback" into concrete user-facing behaviors and study questions
