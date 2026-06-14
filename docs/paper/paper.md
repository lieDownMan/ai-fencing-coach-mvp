# Supporting Solo Fencing Practice with Interpretable AI Feedback

## Abstract

Fencers practicing without continuous coaching may struggle to identify and
correct subtle posture and footwork errors. Existing fencing-computing systems
emphasize action recognition or expert analysis, leaving open how
fencing-specific feedback should support learners during and after solo
practice. We present AI Fencing Coach, an Android prototype whose on-device
pipeline combines commodity-camera pose estimation, fencing action recognition,
biomechanical heuristics, and feedback prioritization. It delivers visual and
spoken cues during practice plus post-session video and error review, while
retaining coaches as final authorities. We conducted a preliminary mixed-method
study with four participants using baseline self-review and AI-assisted
practice, collecting ratings from all four and detailed post-task interviews
from two. Ratings were highest for error awareness, post-review usefulness, and
perceived advantage over self-review (all M=4.75), but lower for timing accuracy
and correction understandability (both M=3.25). Participants could act on step
and center-of-mass cues; however, delayed timing and action misclassification
obscured which movement the feedback referred to. These findings identify
temporal grounding, layered feedback, and explicit handling of recognition
errors as requirements for interpretable AI support in solo movement practice.

## Hero Image

<!-- Insert hero image here.
Figure 1. Intended use of AI Fencing Coach during solo footwork practice: a
commodity camera captures the fencer, the system identifies a fencing-specific
movement issue, and the interface presents one prioritized correction while
preserving the coach as the final authority. -->

## 1. Introduction

### 1.1 Motivation and Human Problem

Learning fencing requires repeated practice together with feedback about how a
movement was performed. During solo drills or lightly supervised training,
however, a coach cannot observe every advance, retreat, lunge, or recovery.
Learners must execute the movement while also judging stance height, balance,
step width, arm extension, and hand-foot coordination. Some errors are visible
only briefly, and others are difficult to interpret without fencing-specific
knowledge. When feedback is unavailable at the relevant moment, a learner may
continue repeating a movement without knowing which part should change.

Our preliminary observations illustrate that this difficulty is not uniform.
One participant could identify conspicuous vertical movement and a narrow
stance through self-review, but needed external explanation for less familiar
hand and defensive-action issues. Another participant initially could not name
a specific problem in the same baseline review and only recognized a
center-of-mass issue after it was pointed out. The challenge is therefore not
simply to expose learners to more data. It is to help people notice a relevant
error, connect it to the movement that produced it, understand a correction,
and decide whether to act on it.

This is an HCI problem as much as a computer-vision problem. A technically
correct label can still be ineffective if it arrives too late, refers
ambiguously to a completed movement, interrupts the next repetition, or uses a
concept the learner cannot translate into action. Conversely, a concise cue
can support immediate adjustment but omit the evidence and explanation needed
for reflection. AI-assisted fencing practice must therefore coordinate
detection, timing, presentation, and appropriate reliance, rather than treat
recognition accuracy as the endpoint.

### 1.2 Current Practice and Workarounds

Recorded-video self-review is a lightweight way to revisit practice without
specialized sensing. It preserves the performed movement and can reveal gross
posture or footwork patterns, but it also shifts the burden of diagnosis to the
learner. In our baseline condition, the two interviewed participants differed
substantially in what they could infer from video alone. Self-review helped one
participant articulate a visible balance problem, while the other remained
unsure what to change. Video can show that something happened without
necessarily explaining what the movement means in fencing terms or what the
next repetition should look like.

The preliminary study suggests value in combining immediate and reflective
support. Across four participants, the highest mean ratings were for error
awareness, usefulness of post-session review, and advantage over self-review
(`M=4.75` for each). Actionable correction and solo-practice support were also
assigned means of `M=4.50`. In the interviews, both participants
reported that they could respond to step-size and center-of-mass cues, and one
described post-session analysis as adding detail that short spoken feedback
could not provide during movement.

The same evidence also exposes the limits of automated feedback. Timing
accuracy and correction understandability received lower ratings
(`M=3.25` for each). Both interviewees were sometimes unsure whether a cue
referred to the current action or an earlier one, and both encountered cases in
which ordinary arm motion was interpreted as a defensive action. These
breakdowns matter because they weaken the connection between evidence and
correction. They motivate a layered feedback design: brief, prioritized cues
for the next action; richer, timestamped review after practice; and explicit
room for learners and coaches to question the system's interpretation.

### 1.3 Comparison with Key Prior Work

Fencing-computing research has established that fine-grained movement can be
modeled from video and pose data. [FenceNet](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Zhu_FenceNet_Fine-Grained_Footwork_Recognition_in_Fencing_CVPRW_2022_paper.html)
uses temporal convolution over 2D skeleton sequences to recognize six fencing
footwork actions without requiring wearable sensors. Work on
[continuous footwork segmentation](https://doi.org/10.24132/CSRN.3301.28)
extends recognition to longer training sequences, while
[FencingVis](https://doi.org/10.1007/s12650-018-0521-3) demonstrates how
interactive visualization can help experts inspect technical and tactical
patterns. Together, these systems establish important infrastructure for
fencing analysis, but their primary outputs are action labels, segments, or
expert-facing analytical views rather than moment-to-moment learner
corrections.

In parallel, HCI research on movement learning shows that feedback
representation shapes what learners can perceive and change. Systems using
[pose estimation alongside instructional video](https://doi.org/10.1145/3399715.3399835)
and studies of
[visual cues for physical training](https://doi.org/10.1145/3491102.3517735)
demonstrate how overlays and targeted visualizations can support comparison
with a desired movement. An ethnographic-informed study of
[feedback in rhythmic gymnastics](https://doi.org/10.1145/3613904.3642434)
further shows that useful feedback varies with timing, form, quantity, movement
complexity, and the athlete's exercise-specific skill. This work foregrounds
the feedback experience, but it does not resolve how to ground that experience
in fencing-specific actions, terminology, and error priorities using a
commodity phone.

The two research traditions therefore stop short in complementary ways:
fencing-specific systems are strong on recognition and analysis, whereas
learner-facing movement systems are stronger on interaction and feedback but
are not organized around fencing practice. The remaining opening is not merely
to combine their technical components. It is to determine how fencing-specific
AI feedback can remain understandable and actionable when cues must be brief,
several errors may co-occur, and the underlying recognition can be delayed or
wrong.

### 1.4 Research Gap and Questions

We investigate this opening through **AI Fencing Coach**, a native Android
prototype for solo practice. Its core pipeline runs on the phone and combines
camera-based pose estimation, target tracking, fencing-action recognition,
rule-based biomechanical checks, and a scheduler that ranks detected issues.
During practice, the interface presents a small number of visual issues and
speaks one prioritized correction at a time. After a live session or selected
video, it provides action counts, repeated-error summaries, a cue timeline,
practice suggestions, and persistent session history. Optional language-model
summaries can rephrase this evidence, but deterministic playbook feedback
remains available and the detected movement data constrains what the summary
may claim.

We ask:

1. **RQ1:** How does AI-assisted feedback support fencers' ability to notice and
   interpret posture and footwork errors compared with video self-review?
2. **RQ2:** Which properties of live feedback make a correction actionable
   without unnecessarily interrupting practice?
3. **RQ3:** How do cue timing, action-recognition errors, and post-session
   evidence shape appropriate trust in an AI fencing coach?

To obtain initial evidence, we conducted a preliminary mixed-method study with
four participants who completed baseline self-review and AI-assisted practice.
We collected post-use ratings from all four and analyzed detailed post-task
interviews from two. The results suggest that participants valued the system
most for making errors visible and supporting later review. They could act on
several concrete cues, especially those concerning step size and body balance,
without reporting substantial interruption. At the same time, delayed cues and
misclassified actions made some feedback difficult to ground in a specific
movement. We use these results to characterize both the promise and the design
requirements of interpretable feedback, rather than as evidence of durable
skill improvement.

### 1.5 Contributions and Scope

This work contributes:

1. **An empirical account of early AI-assisted fencing practice.** Descriptive
   ratings and interviews show how learners used live and post-session
   feedback, where they perceived advantages over unaided self-review, and how
   timing ambiguity and recognition errors undermined interpretation.
2. **A phone-based fencing coaching artifact.** The Android prototype
   integrates on-device pose and action analysis with fencing-specific
   heuristics, prioritized visual and spoken cues, video review, and
   longitudinal session history.
3. **Design implications for interpretable movement feedback.** The findings
   motivate temporal grounding between cue and action, layered detail across
   live and reflective feedback, prioritization when errors co-occur, and
   interaction mechanisms that allow uncertain or incorrect analysis to be
   inspected rather than silently treated as authoritative.

Our scope is deliberately bounded. The four-participant study supports
feasibility observations and design hypotheses, not population estimates,
long-term learning effects, or claims that the system improves fencing skill.
The prototype analyzes a limited set of visible movement features from a
single camera and cannot evaluate every technical, tactical, or safety-relevant
aspect of fencing. AI Fencing Coach is therefore positioned as a practice and
reflection aid for moments when expert attention is unavailable, not as a
replacement for a coach or as an independent authority on technique.

## 2. Related Work Analysis

### 2.1 Computational Analysis of Fencing

### 2.2 Interactive Systems for Sports and Movement Learning

### 2.3 AI-Generated Feedback, Explainability, and Trust

### 2.4 Positioning the Present Work

## 3. Formative Prototype Study and Analysis

Before stabilizing the interaction design, we used an early working prototype
to reduce four consequential uncertainties: (1) what learners could diagnose
through video self-review alone, (2) which corrections could be understood and
acted on during movement, (3) how immediate and post-session feedback should
divide explanatory work, and (4) how timing and recognition errors would
affect interpretation and trust. The study was therefore formative in a
design-oriented sense: it examined current self-review alongside reactions to
prototype feedback and was used to refine requirements. It was not a
pre-design ethnography, a technical validation, or a test of skill learning.

The same sessions provide the preliminary feasibility evidence reported in
Sections 7 and 8. This section focuses on what the sessions changed or
constrained about the design; the later sections report the evaluative
questions, descriptive ratings, and observed interaction outcomes. Those
later results should not be read as an independent validation of requirements
derived from the same sessions.

### 3.1 Participants and Recruitment

Four participants contributed post-use questionnaire ratings. Detailed
Chinese-language interview records were available for two of them, referred
to as P1 and P2. The records cover unaided video self-review, real-time
earphone feedback, a feedback-guided attempt, and post-session video analysis.
For the other two participants, only their contribution to the aggregate
questionnaire means is preserved.

The supplied artifacts do not record recruitment source, inclusion criteria,
age, gender, fencing experience, compensation, study location, or the
relationship between participants and researchers. We therefore cannot
reconstruct the sampling rationale or claim that the participants represent
beginner, intermediate, or expert fencers. Coaches were not represented in the
available corpus, which is especially important because the study concerns
feedback authority and technical correctness. Table 1 reports only evidence
that can be verified from the retained files.

**Table 1. Retained evidence by participant.**

| Available formative evidence | P1 | P2 | Two additional participants |
| --- | --- | --- | --- |
| Post-use questionnaire | Included in aggregate | Included in aggregate | Included in aggregate |
| Unaided self-review account | Available | Available, with interviewer prompting | Not available |
| Real-time feedback account | Available | Available | Not available |
| Feedback-guided attempt account | Available | Partial; final verbal response missing | Not available |
| Post-session analysis account | Available | Available | Not available |

The artifacts also do not preserve the applicable ethics-review status,
consent procedure, video-retention policy, or withdrawal process. These details
must be recovered from the original study records before publication. In this
manuscript, we use pseudonymous identifiers and report only
movement- and interaction-relevant observations.

### 3.2 Procedure and Materials

The study used a fixed sequence centered on a short fencing movement set that
could include en garde, advance, retreat, lunge, and advance-lunge actions.
Participants first performed without automated feedback and reviewed the
recorded video. They described what appeared correct, what appeared
problematic, and what they would change on another attempt. This phase served
as the current-practice baseline because recorded self-review is accessible
without a continuously present coach.

Participants were then introduced to an early prototype with two feedback
channels. During another performance, the real-time mode analyzed camera input
and delivered short spoken cues through earphones, including cues about stance
height, center-of-mass position, and step width. Participants subsequently
attempted to adjust their movement. After practice, they inspected a
video-analysis view that associated detected problems with moments in the
recording and summarized recurring issues. A semi-structured interview asked
about error awareness, cue meaning, timing, immediate adjustment,
interruption, incorrect detections, and the relative value of live and
post-session feedback. Participants also completed a 15-item post-use
questionnaire.

The retained corpus consists of ordered question-and-answer records for P1 and
P2 and a CSV containing the mean of each questionnaire item across four
participants. The interview files are reorganized summaries, not complete
verbatim transcripts, and the questionnaire file does not include item
wording, response anchors, participant-level values, or dispersion. The
prototype build, phone model, camera placement, model version, session
duration, and exact movement repetitions are also absent. These omissions
limit reproducibility and must be filled from contemporaneous records where
possible.

The order was not counterbalanced: self-review always preceded AI-assisted
practice. Repetition, growing task familiarity, and interviewer explanation
could therefore influence later responses. Two deviations are visible in the
records. The interviewer proposed a forward center-of-mass problem before P2
agreed with it, so P2's baseline was not fully independent. Developer
explanations of possible error types and system behavior were also interleaved
with P1's session, potentially shaping the language and interpretations used
later in that interview.

### 3.3 Qualitative Analysis

We used a focused hybrid analysis organized around a practice-feedback cycle:
**notice**, **interpret**, **act**, and **verify**. These deductive categories
captured whether a participant detected a problem, understood its referent,
selected a correction, and obtained evidence about the result. We then tracked
inductively observed breakdowns involving cue timing, interruption,
action-recognition errors, disagreement with the system, and differences
between live and post-session feedback.

We organized the ordered answers in a participant-by-finding matrix and
compared P1 and P2 within each phase. We retained negative and divergent cases,
including a participant who could already diagnose major errors, a
learner-relevant issue the system apparently did not cue, delayed but still
actionable feedback, and defensive-action labels that participants questioned.
Because the records summarize speech, we do not treat their phrasing as a
verbatim corpus except where an answer explicitly preserves participant words.

Questionnaire means were used only as descriptive triangulation. We did not
infer individual responses, variability, confidence intervals, or statistical
significance from the aggregate file. The analysis also does not claim
thematic saturation: only two detailed cases were available, one baseline was
interviewer-prompted, and the development team's presence may have encouraged
participants to accommodate the prototype's framing.

### 3.4 Findings

#### Unaided self-review exposed different diagnostic starting points

P1 independently identified vertical center-of-mass movement and narrow steps
and could state a corresponding adjustment. P2 initially could not name a
specific problem or correction and agreed with a center-of-mass diagnosis only
after interviewer prompting. The formative need is therefore not simply to
show every learner an error they cannot see. Some learners need help detecting
and naming a problem, while others need finer evidence, prioritization, or
confirmation of an existing judgment. This variation argues for establishing
baseline diagnostic ability rather than treating all users as equally unable
to self-review. The high aggregate ratings for error awareness and advantage
over self-review (both `M=4.75`, `N=4`) are promising, but they cannot erase
the difference between P1 and P2 or establish how common either case is.

#### Brief directional cues supported action, while reflection required detail

Both interviewees reported attempting adjustments to step size and
center-of-mass position after spoken feedback. P1 found step cues especially
direct because "too far" or "too near" implied a concrete change on the next
attempt. Neither participant identified the presence of earphone audio as a
major interruption, consistent with the comparatively high low-interruption
rating (`M=4.25`). However, the interviews do not establish that these
adjustments were biomechanically correct; no blinded coach rating or kinematic
pre/post measure was retained.

More technical or unfamiliar issues, particularly those concerning the weapon
arm or a defensive action, were harder to resolve from a short spoken cue. P1
described the post-session analysis as adding detail that could not reasonably
be delivered during movement. This creates a division of labor between
feedback channels: live feedback should support one immediate next action,
whereas review should preserve evidence, terminology, repeated patterns, and
practice guidance. The lower correction-understandability rating (`M=3.25`)
reinforces that hearing a concise cue is not the same as understanding a full
correction.

#### Interpretability depended on temporal grounding

Both detailed cases had difficulty determining which movement a cue described.
P1 was unsure whether the system was responding to the current action or the
preceding repetition. P2 reported that some cues arrived well after an action
had ended. Participants could understand the words and still lack an
actionable referent. This distinction helps explain why timing accuracy was
among the lowest-rated items (`M=3.25`) even though participants reported
acting on several cues.

The design problem is consequently broader than reducing average latency. A
cue must identify, through timing or representation, the movement episode to
which it belongs. When immediate delivery cannot be guaranteed, the interface
should defer or mark the feedback rather than present a delayed diagnosis as
if it described the action currently underway.

#### Recognition errors required contestability, not unconditional trust

Both interviews contained cases in which ordinary hand movement was associated
with a defensive-action diagnosis. P1 also questioned an apparent
lunge-related classification. The cases produced different responses: P1
could reinterpret one hand-motion description as plausible after discussion,
whereas P2 maintained that no defensive action had occurred. This variation is
more informative than a single trust score. It shows that users may accept,
reinterpret, or reject an output depending on their own memory and the
available evidence.

An incorrect action label changes the meaning of otherwise sensible coaching:
advice about how to perform a parry is not useful if no parry occurred. A
detailed post-session report can amplify this problem by making a false
classification appear more authoritative. The formative implication is that
confidence, abstention, inspection, and correction are part of the feedback
interaction rather than purely backend concerns. The moderate aggregate trust
rating (`M=3.75`) should therefore not be interpreted as blanket acceptance.

### 3.5 Design Requirements

We translated the findings into five design requirements. Table 2 also records
how the current Android repository responds to each requirement. These
implementation links establish traceability, not effectiveness: a feature's
presence does not demonstrate that the requirement has been satisfied in use.

**Table 2. Formative findings, design requirements, current implementation,
and unresolved evaluation questions.**

| Requirement | Evidence-backed rationale | Current Android response | Remaining question |
| --- | --- | --- | --- |
| **DR1: Reduce diagnostic burden without assuming zero self-awareness.** | P1 could diagnose gross posture and footwork errors; P2 struggled to name a problem before external prompting. | Live visual labels and speech name detected issues; post-session breakdowns and timelines provide additional evidence. | Which feedback layer adds information beyond each learner's unaided self-review? |
| **DR2: Keep live feedback brief and prioritized; move explanation to review.** | Directional step and balance cues supported reported adjustment, while unfamiliar arm and defensive-action issues needed more context. | The scheduler speaks one weighted cue at a time, limits repeated speech with cooldowns, shows up to three visual cues, and stores richer diagnosis and practice text for review. | Does prioritization improve next-attempt correction without increasing interruption or hiding learner-relevant errors? |
| **DR3: Ground every cue in a movement episode.** | Both participants were unsure whether delayed feedback referred to the current or a previous action. | Cue history stores frame-based timestamps, and review views expose a timeline; video analysis can preserve the relevant moment. Real-time speech does not yet identify a repetition or action referent. | What end-to-end latency and representation let users identify the intended movement reliably? |
| **DR4: Make recognition uncertainty inspectable and contestable.** | Hand motion was interpreted as a defensive action, and an incorrect action label made the attached correction misleading. | The live interface displays action confidence, and generated summaries are constrained to detected counts and playbook evidence. Cue-level confidence, user correction, dismissal, and abstention are not yet implemented. | How often does false feedback occur, and do uncertainty and correction controls support appropriate reliance? |
| **DR5: Preserve learner and coach authority.** | Participants did not accept every diagnosis, and the corpus contains no coach validation of technical correctness. | Users can disable speech, pause analysis, focus or mute error categories, and retain deterministic playbook feedback when an optional language-model summary is unavailable. A coach-review workflow is not yet present. | Do learner controls and coach review prevent automation from overriding situated expertise? |

The study narrowed three initial assumptions. More simultaneous feedback was
not necessarily more useful; semantic clarity alone could not repair a cue
whose movement referent was ambiguous; and a detailed report was not
inherently trustworthy when its underlying action classification was wrong.
These constraints motivated the current layered, prioritized, and
history-preserving design while identifying temporal grounding and
contestability as incomplete work.

The resulting requirements remain design hypotheses. The small and partially
retained corpus, fixed condition order, interviewer influence, missing
participant characteristics, absence of coaches, and lack of objective
movement measures prevent claims about prevalence, learning, or technical
correctness. A later study should independently establish baseline
self-diagnosis, measure cue latency and false-feedback frequency, include coach
judgment, and test whether the implemented responses improve correction and
appropriate reliance.

<!-- Insert design-traceability figure here.
Figure 2. Traceability from formative evidence through DR1-DR5 to current
Android features and unresolved evaluation questions. -->

## 4. System Design

### 4.1 Design Goals

### 4.2 Interaction Flow

### 4.3 Feedback Timing and Prioritization

### 4.4 Safety, Uncertainty, and Coach Authority

## 5. Implementation

### 5.1 System Architecture

### 5.2 Pose and Action Analysis

### 5.3 Biomechanical Heuristics

### 5.4 Visual, Spoken, and Post-Session Feedback

<!-- Insert system overview figure here.
Figure 3. AI Fencing Coach pipeline from commodity-camera video through pose
extraction, target tracking, action recognition, fencing-specific heuristic
analysis, feedback prioritization, and visual, spoken, or post-session output. -->

<!-- Insert feedback-rules table here.
Table 3. Implemented feedback rules, triggering conditions, required pose
signals, confidence or abstention behavior, presentation timing, and intended
fencing correction. -->

## 6. Technical Evaluation

### 6.1 Evaluation Dataset and Conditions

### 6.2 Accuracy, Latency, and Calibration Measures

### 6.3 Failure Cases and Abstention Behavior

<!-- Insert technical-evaluation table here.
Table 4. Technical performance by error type and recording condition, including
detection accuracy, false-feedback frequency, latency, confidence calibration,
and observed failure cases. -->

## 7. User Study Design and Analysis

We conducted a preliminary mixed-method feedback study to examine whether the
prototype's real-time and post-session feedback was understandable and
actionable during a short fencing practice workflow. The study was designed as
a feasibility and design-diagnostic study, not as a test of skill acquisition
or long-term training effectiveness. The analyzed corpus contains aggregate
ratings from four participants and detailed ordered interview records for two
of those participants.

### 7.1 Research Questions

Following the questions introduced in Section 1.4, the study asked:

- **RQ1:** How does AI-assisted feedback support fencers' ability to notice and
  interpret posture and footwork errors compared with video self-review?
- **RQ2:** Which properties of live feedback make a correction actionable
  without unnecessarily interrupting practice?
- **RQ3:** How do cue timing, action-recognition errors, and post-session
  evidence shape appropriate trust in an AI fencing coach?

### 7.2 Participants and Ethics

Four participants contributed questionnaire ratings. Detailed Chinese-language
interview records were available for two participants, referred to as P1 and
P2. The supplied study artifacts do not preserve recruitment source, age,
gender, fencing experience, compensation, session duration, or the applicable
ethics-review and consent procedure. We therefore do not infer these
characteristics or claim that the sample represents the wider fencing
population. These details, together with video-data retention and withdrawal
procedures, must be added from the original study records before publication.

The present analysis retains participant identifiers rather than names and
reports only movement-relevant observations. Because only two detailed
interviews were available, qualitative findings describe cases and tensions
observed in this preliminary sample; they do not establish thematic saturation
or population prevalence.

### 7.3 Conditions and Baseline

The study used a fixed sequence built around the learner's current self-review
practice and the prototype's two feedback modes:

1. **Video self-review baseline.** Participants performed a short sequence that
   could include en garde, advance, retreat, and lunge movements. They reviewed
   the recording and described what had gone well, what appeared incorrect,
   and what they would change.
2. **Real-time voice feedback.** Participants repeated the movements while the
   camera-based system analyzed their pose and action sequence. Short cues were
   delivered through earphones, including feedback about stance height,
   center-of-mass position, and step width.
3. **Feedback-guided attempt.** Participants performed another attempt while
   trying to act on the preceding voice feedback.
4. **Post-session video analysis.** Participants reviewed timestamped detected
   issues and a summary of recurring problems, then discussed whether this
   information added value beyond unaided video review.

This was not a counterbalanced comparison. The baseline always preceded
AI-assisted practice, so order, repetition, interviewer guidance, and growing
familiarity with the task are alternative explanations for any perceived
improvement.

### 7.4 Procedure

The researcher first explained the two prototype modes: camera-based real-time
analysis with earphone cues and post-session video analysis. Participants then
completed the baseline, real-time, and feedback-guided phases described above.
After the movement tasks, they inspected the post-session analysis and
completed feedback ratings. Detailed semi-structured interview records were
retained for P1 and P2. Interview prompts covered self-detected errors, cue
comprehension, timing, immediate adjustment, interruption, perceived
usefulness, and incorrect feedback.

Two protocol deviations are visible in the available records. During P2's
baseline review, the interviewer suggested that the participant's center of
mass appeared to move forward before P2 agreed with that diagnosis. P2's
baseline therefore cannot be treated as a fully independent self-diagnosis.
The records also include developer explanations of available error types and
system behavior during P1's session. These explanations may have shaped later
interpretations of the feedback. In addition, P2's final feedback-guided
attempt lacks a complete post-attempt verbal response.

### 7.5 Measures

Table 5 maps each research question to the evidence available in the study
corpus. The questionnaire summary contains 15 item means, each based on four
participants. Item wording, response anchors, participant-level values, and
dispersion statistics were not included in the supplied file.

| Research question | Construct | Operational evidence | Data source |
| --- | --- | --- | --- |
| RQ1 | Unaided error awareness | Problems verbalized during baseline video self-review | Ordered interview records |
| RQ1 | Added diagnostic value | Problems noticed only after voice or video analysis | Ordered interview records; error-awareness and self-review ratings |
| RQ2 | Comprehension | Participant explanation of what a cue referred to | Ordered interview records; clarity and understandability ratings |
| RQ2 | Actionability | Specific adjustment selected or attempted on the next performance | Ordered interview records; actionable-correction and next-attempt ratings |
| RQ3 | Temporal grounding | Reports of delay or uncertainty about which movement a cue described | Ordered interview records; timing-accuracy rating |
| RQ3 | Interruption | Whether voice cues disrupted movement rhythm | Ordered interview records; low-interruption rating |
| RQ3 | Appropriate trust | Acceptance, questioning, or rejection of system diagnoses | Ordered interview records; feedback-trust rating |
| RQ3 | Training fit | Perceived usefulness for solo practice, post-review, coach supplementation, progress tracking, and future use | Aggregate questionnaire ratings |

### 7.6 Quantitative and Qualitative Analysis

We analyzed questionnaire data descriptively. Because only aggregate means and
the participant count were available, we did not reconstruct individual
responses, calculate standard deviations or confidence intervals, or conduct
null-hypothesis significance tests. We calculated category-level means by
averaging the five supplied item means within each category.

For the qualitative analysis, we used a focused hybrid framework organized
around the practice-feedback cycle: **notice**, **interpret**, **act**, and
**verify**. We additionally coded timing, interruption, recognition error,
trust, and the relationship between real-time and post-session feedback. We
constructed a participant-by-finding matrix, compared P1 and P2, and retained
negative and contradictory cases. The source files are ordered interview
records containing summarized answers rather than complete verbatim
transcripts; quotations are therefore used only where the record preserves the
participant's words. Chinese quotations were translated into English for
reporting.

<!-- Insert user-study procedure figure here.
Figure 4. Fixed-order preliminary study procedure: unaided video self-review,
real-time earphone feedback, a feedback-guided attempt, post-session analysis,
questionnaire, and semi-structured interview. -->

## 8. Preliminary Study Results (4-6 Users)

### 8.1 Participant Overview

All four participants contributed to each aggregate questionnaire item. P1 and
P2 additionally contributed detailed interview evidence spanning baseline
self-review, real-time voice feedback, and post-session video analysis. No
interview or participant-level questionnaire records were available for the
other two participants. Participant demographics and fencing-experience levels
were also unavailable, preventing subgroup comparison.

| Available evidence | P1 | P2 | Two additional participants |
| --- | --- | --- | --- |
| Questionnaire contribution | Included in aggregate | Included in aggregate | Included in aggregate |
| Baseline self-review record | Available | Available, with interviewer prompt | Not available |
| Real-time feedback interview | Available | Available | Not available |
| Feedback-guided attempt account | Available | Partial; no complete final response | Not available |
| Post-session analysis interview | Available | Available | Not available |

### 8.2 Behavioral and Performance Results

The two interviews showed different starting points for unaided self-review.
P1 independently identified vertical center-of-mass movement and narrow steps
and could already state a correction: remain level and adjust step width. P2
did not independently identify a concrete correction and said, in translation,
"I am not really sure"; P2 agreed with a center-of-mass diagnosis only after
the interviewer introduced it. The prototype therefore did not provide wholly
new awareness for every participant. Instead, it made feedback more specific
for a participant with limited self-diagnosis while adding detail for a
participant who could already identify gross errors.

Both interviewees reported attempting immediate adjustments after voice cues.
P1 described changing step size in response to "too far" or "too near" cues
and using subsequent feedback to judge whether the adjustment had changed the
system response. P2 reported adjusting center-of-mass position and step size.
These accounts demonstrate perceived actionability, but the available corpus
does not contain blinded technique ratings, coded kinematics, or
participant-level pre/post performance measures. We therefore cannot determine
whether the attempted corrections were technically correct or improved
fencing performance.

### 8.3 Experience, Trust, and Actionability

Across the 15 questionnaire items, the unweighted overall mean was 4.20. The
five learning-support items had the highest category mean (4.55), followed by
training value (4.45) and feedback quality (3.60). The contrast is important:
participants rated the workflow as useful for noticing and reviewing errors,
but were less positive about whether individual cues arrived at the right time
and clearly explained how to correct the movement.

| Category | Measure | Mean | N |
| --- | --- | ---: | ---: |
| Feedback quality | Feedback clarity | 3.50 | 4 |
| Feedback quality | Timing accuracy | 3.25 | 4 |
| Feedback quality | Correction understandability | 3.25 | 4 |
| Feedback quality | Low interruption | 4.25 | 4 |
| Feedback quality | Feedback trust | 3.75 | 4 |
| Learning support | Error awareness | 4.75 | 4 |
| Learning support | Actionable correction | 4.50 | 4 |
| Learning support | Improved next attempt | 4.25 | 4 |
| Learning support | Reflection support | 4.50 | 4 |
| Learning support | Better than self-review | 4.75 | 4 |
| Training value | Solo practice support | 4.50 | 4 |
| Training value | Post-review usefulness | 4.75 | 4 |
| Training value | Progress tracking | 4.25 | 4 |
| Training value | Coach supplement | 4.50 | 4 |
| Training value | Willingness to use | 4.25 | 4 |

The highest item means were error awareness, better than self-review, and
post-review usefulness (all M=4.75). Timing accuracy and correction
understandability were lowest (both M=3.25). Low interruption was comparatively
high (M=4.25), matching the interviews: neither P1 nor P2 identified voice
feedback itself as a major disruption. Their difficulty was temporal
reference, not merely the presence of audio.

### 8.4 Qualitative Findings

#### Feedback made uneven self-review more concrete

The baseline cases show that self-review ability varied. P1 could identify
large posture and footwork problems without assistance, whereas P2 struggled
to name either a problem or a next action. After using the system, both
participants discussed concrete categories such as stance height,
center-of-mass position, and step width. For P2, the primary value was external
diagnosis: the system named issues that were difficult to articulate through
video review alone. For P1, the value was refinement rather than first
detection, especially when post-session analysis supplied more detailed or
technical descriptions.

#### Short cues supported action, while review supported explanation

Both participants could act most readily on concise, directional feedback
about step size and center of mass. P1 described step-width cues as especially
clear because they directly indicated whether to make the next step shorter or
wider. However, detailed issues involving the weapon arm or a defensive action
were harder to resolve through audio alone. P1 explicitly treated the two
modalities as complementary: voice was useful for immediate adjustment,
whereas video analysis and the end-of-session summary provided the detail
needed for reflection. This finding supports a layered design rather than
attempting to deliver a full biomechanical explanation during movement.

#### A cue without a clear movement referent was difficult to use

Temporal grounding was the most consistent breakdown. P1 was unsure whether a
cue described the current movement or the preceding repetition. P2 reported
that some feedback arrived well after the movement had ended, making it
difficult to know which action should be corrected. This pattern explains why
timing accuracy and correction understandability received the lowest mean
ratings even though participants generally heard and understood the words.
Interpretability in this setting therefore depends on linking advice to a
specific movement episode, not only on simplifying the cue text.

#### Recognition errors changed the meaning of otherwise plausible advice

Both interviews contain examples in which hand movement was interpreted as a
defensive action even though the participant did not believe they had
performed one. P1 also questioned an apparent lunge-related classification.
These were not cosmetic errors: once the action label was wrong, the associated
coaching advice no longer had a trustworthy referent. P2 still considered most
other feedback reasonable, showing calibrated rather than total rejection.
The finding nevertheless indicates that action confidence, uncertainty, and a
way to inspect or dismiss questionable detections are necessary parts of the
feedback interaction.

### 8.5 Unexpected Findings and Negative Cases

Several cases narrow the contribution. First, P1 already recognized the main
center-of-mass and footwork problems through self-review, so the system's value
cannot be framed as universal discovery of otherwise invisible errors. Second,
P1 did not recall hearing an explicit cue for the vertical center-of-mass
movement that they considered important, showing that the system could miss or
fail to prioritize a learner-relevant issue. Third, P2 acted on feedback while
also reporting substantial delay; usefulness and timing quality should
therefore be evaluated as separate constructs. Fourth, the defensive-action
false positives show that a detailed post-session report can amplify, rather
than repair, a recognition error if it presents the diagnosis confidently.

The study also exposed protocol limitations. The fixed order confounds system
use with repetition and learning, P2's baseline was interviewer-prompted,
developer explanations were interleaved with P1's interview, and P2's final
post-attempt account was incomplete. Only aggregate questionnaire means were
retained, and only two of four interviews were available. These constraints
support a claim that the study identified feasibility and design issues; they
do not support claims of improved skill, superiority to self-review, or
generalizable trust.

The resulting design requirements are:

| Finding | Design requirement | Main-study measure |
| --- | --- | --- |
| Participants could act on short step and center-of-mass cues | Keep real-time cues brief and directional; move explanation to post-session review | Cue comprehension and next-attempt correction |
| Participants could not reliably connect delayed cues to a movement | Attach feedback to a repetition or timestamp and report end-to-end cue latency | Referent-identification accuracy and latency |
| Action misclassification produced misleading coaching | Expose confidence, support dismissal or correction, and abstain when action evidence is weak | False-feedback rate and appropriate reliance |
| Self-review ability differed between P1 and P2 | Measure baseline diagnostic ability and report who benefits from which feedback layer | Added error detection beyond unaided review |
| Voice and video served different roles | Preserve layered immediate, visual, and post-session feedback | Interruption, review usefulness, and correction quality |

The current Android implementation already reflects part of this agenda: its
scheduler speaks one prioritized cue while retaining up to three visual issues,
and post-session review presents a timestamped cue history. It also displays
action-classification confidence. However, real-time advice is not explicitly
bound to a repetition identifier, and the interface does not yet expose
cue-specific uncertainty or let a learner dismiss or correct a questionable
detection. These remain implementation and evaluation priorities rather than
resolved outcomes of the preliminary study.

## 9. Discussion

### 9.1 How AI Feedback Fits Solo Fencing Practice

### 9.2 Design Implications for AI-Assisted Sports Coaching

### 9.3 Relationship to Coaches and Existing Practice

## 10. Limitations and Future Work

## 11. Conclusion

## Acknowledgments

## References
