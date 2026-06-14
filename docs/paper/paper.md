# Supporting Solo Fencing Practice with Interpretable AI Feedback

## Abstract

Fencers practicing without continuous coaching may struggle to identify and
correct subtle posture and footwork errors. Much existing fencing-computing
work emphasizes action recognition or expert analysis, leaving open how
fencing-specific feedback should support learners during and after solo
practice. We present AI Fencing Coach, an Android prototype whose on-device
pipeline combines commodity-camera pose estimation, fencing action recognition,
biomechanical heuristics, and feedback prioritization. It delivers visual and
spoken cues during practice plus post-session video and error review, while
retaining coaches as final authorities. We conducted a preliminary mixed-method
study with four participants using baseline self-review and AI-assisted
practice, collecting ratings from all four and detailed post-task interviews
from two. Descriptive ratings were highest for error awareness, post-review
usefulness, and perceived advantage over self-review (M=4.75 for each), and
lowest for timing accuracy and correction understandability (M=3.25 for each).
Participants could act on step and center-of-mass cues; however, delayed timing
and action misclassification obscured which movement the feedback referred to.
These findings identify temporal grounding, layered feedback, and explicit
handling of recognition errors as requirements for interpretable AI support in
solo movement practice.

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
support. Among the 15 supplied aggregate questionnaire items, the highest means
were for error awareness, usefulness of post-session review, and advantage over
self-review (`M=4.75` for each). Actionable correction and solo-practice support
were assigned means of `M=4.50`. In the interviews, both participants reported
that they could respond to step-size and center-of-mass cues, and one described
post-session analysis as adding detail that short spoken feedback could not
provide during movement.

The same evidence also exposes the limits of automated feedback. Timing
accuracy and correction understandability received lower ratings
(`M=3.25` for each). Both interviewees were sometimes unsure whether a cue
referred to the current action or an earlier one, and both encountered cases in
which ordinary arm motion was interpreted as a defensive action. These
breakdowns matter because they weaken the connection between evidence and
correction. They motivate a layered feedback design: brief, prioritized cues
for the next action; a richer, timestamped cue history after practice; and
explicit room for learners and coaches to question the system's interpretation.

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

Fencing-specific systems have also begun to investigate feedback through
specialized interaction settings. An
[IoT fencing feedback system](https://doi.org/10.3390/s23249801) uses a
body-mounted gyroscope with visual or smartwatch-based haptic feedback, while
an
[instructional design map for immersive VR fencing](https://doi.org/10.22364/htqe.2022.13)
organizes expert-informed requirements for simulated practice. These efforts
show that fencing technology can move beyond classification, but they center
on wearable sensing or immersive environments rather than the timing and
interpretation problems of feedback generated from an ordinary phone camera.

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

The opening is therefore not the complete absence of learner-facing fencing
feedback. Rather, recognition and visualization work leaves the feedback
interaction unresolved, existing fencing feedback centers on specialized
sensing or immersive practice, and commodity-camera movement systems are not
organized around fencing technique and error priorities. We still lack
evidence about how a phone-based fencing feedback loop should connect imperfect
recognition to sparse live cues and richer review when several errors may
co-occur and a delayed or incorrect inference can misdirect the learner.

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
   evidence shape appropriate reliance on AI-generated fencing feedback?

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
   persistent session history.
3. **Design implications for interpretable movement feedback.** The findings
   motivate temporal grounding between cue and action, layered detail across
   live and reflective feedback, prioritization when errors co-occur, and
   requirements for inspecting and contesting uncertain or incorrect analysis
   rather than silently treating it as authoritative.

Our scope is deliberately bounded. The four-participant study supports
feasibility observations and design hypotheses, not population estimates,
long-term learning effects, or claims that the system improves fencing skill.
The prototype analyzes a limited set of visible movement features from a
single camera and cannot evaluate every technical, tactical, or safety-relevant
aspect of fencing. AI Fencing Coach is therefore positioned as a practice and
reflection aid for moments when expert attention is unavailable, not as a
replacement for a coach or as an independent authority on technique.

## 2. Related Work Analysis

Prior work relevant to AI Fencing Coach spans three connected conversations:
computational analysis of fencing, interactive systems for movement learning,
and human-AI feedback that remains understandable when model outputs are
uncertain. We organize this section by what each conversation establishes for
the present research. The central distinction is between recognizing a
movement and designing a feedback interaction through which a learner can
identify, interpret, act on, and, when necessary, question that recognition.

### 2.1 Computational Analysis of Fencing

Fencing-computing research has established increasingly accessible methods for
representing footwork. Malawski and Kwolek introduced the Fencing Footwork
Dataset (FFD) while studying how the dynamics of six visually similar actions
could be distinguished using synchronized skeleton, depth, and inertial
signals in
[Recognition of Action Dynamics in Fencing Using Multimodal Cues](https://doi.org/10.1016/j.imavis.2018.04.005).
Their results showed the value of temporal and multimodal information for
fine-grained recognition, but the sensing configuration included Kinect depth
data and body-worn inertial measurement units. [FenceNet](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Zhu_FenceNet_Fine-Grained_Footwork_Recognition_in_Fencing_CVPRW_2022_paper.html)
subsequently demonstrated that temporal convolution over 2D skeleton sequences
could classify the same six action classes without depth or wearable inputs.
More recent work on
[temporal segmentation of fencing footwork](https://doi.org/10.24132/CSRN.3301.28)
extends this line from isolated actions toward continuous training sequences.
Together, these studies make ordinary RGB video a plausible basis for
fencing-action analysis, while primarily evaluating labels and segments rather
than how a learner understands or uses a correction.

Other systems broaden the object and audience of fencing analysis. [FencingVis](https://doi.org/10.1007/s12650-018-0521-3)
uses coordinated visualizations to help experts inspect technical and tactical
patterns across competition data. Sawahata et al. address the different
computer-vision problem of a thin, rapidly moving weapon in
[markerless sword-tip tracking](https://doi.org/10.1109/SII58957.2024.10417603).
Their monocular system recognizes the sword through instance segmentation and
uses historical and wrist information to estimate the tip when it is not
visible. These systems make actions, tactics, and trajectories more
inspectable, but they are oriented toward analytical interpretation or
trajectory visualization. Weapon tracking also cannot by itself explain the
fencer's stance, balance, or coordination.

This literature therefore supplies necessary sensing and recognition
infrastructure, not a complete coaching interaction. A classifier may produce
the correct action label while leaving unresolved which error matters, when a
cue should be delivered, what the learner should change next, and how a false
label should be exposed. Our work treats those unresolved questions as HCI
problems rather than assuming that recognition accuracy directly implies
useful coaching.

### 2.2 Interactive Systems for Sports and Movement Learning

Research on movement-learning interfaces shows that feedback representation
changes what learners can compare and correct. Systems using
[pose estimation with online instructional videos](https://doi.org/10.1145/3399715.3399835)
support visual comparison between a learner and a demonstrated movement, while
work on
[visual cues for physical training](https://doi.org/10.1145/3491102.3517735)
examines how overlays, highlights, and directional representations communicate
movement instructions. These approaches move beyond classification by making
differences perceptible. However, visual similarity to a demonstration does
not determine which fencing-specific issue should be prioritized, especially
when several posture and footwork problems co-occur during a rapid sequence.

Feedback is also situated within a coaching practice rather than defined only
by its modality. An ethnographic-informed study of
[feedback in rhythmic gymnastics](https://doi.org/10.1145/3613904.3642434)
found that coaches vary feedback form, timing, format, quantity, and type in
relation to the athlete and exercise. This account cautions against treating
"real-time feedback" as a single design property: information that is useful
before a movement, during a slower exercise, or after a complex sequence may
not be interchangeable. Related work on
[knowledge-grounded AI feedback for basketball shooting](https://doi.org/10.1145/3706598.3713324)
further illustrates the importance of translating coaching knowledge into
structured, actionable feedback rather than allowing a generic model output
to stand in for instruction.

Fencing systems have explored this translation through additional hardware
and immersive displays. Niță and Magyar's
[IoT feedback system](https://doi.org/10.3390/s23249801) measures torso angular
velocity with a body-mounted gyroscope and provides visual or smartwatch-based
haptic feedback. In their five-week comparison, statistically significant
improvement was reported for the visual-feedback group, but not for the haptic
or control groups, underscoring that sensing alone does not make feedback
effective. Dreimane and Zālīte-Supe instead developed an
[instructional design map for immersive VR fencing](https://doi.org/10.22364/htqe.2022.13)
from literature, existing VR experiences, and expert interviews. Their work
foregrounds en garde position, balance, weapon and body position, and tactical
experience, but proposes a design framework for simulated practice rather than
evaluating camera-based feedback during ordinary physical training.

These systems reveal a design tradeoff. Wearables and immersive environments
can provide specialized signals or controlled experiences, while phone video
reduces equipment burden and preserves the learner's physical practice
context. The latter also places greater demands on target tracking, temporal
grounding, and feedback selection because sensing is less controlled. We
therefore investigate how a commodity-camera system can divide work between
brief feedback during movement and richer evidence after movement, rather than
attempting to deliver a complete explanation through one channel.

### 2.3 AI-Generated Feedback, Explainability, and Trust

When detected movement is converted into advice, a recognition error becomes
an interaction error: a fluent correction may be understandable yet irrelevant
to the action the learner performed. Human-AI research distinguishes general
trust from
[appropriate reliance](https://doi.org/10.1145/3581641.3584066), in which
people accept useful AI advice while resisting incorrect advice. That work
treats explanations as one factor that can influence, rather than guarantee,
such discrimination. Likewise, research on
[presenting AI uncertainty](https://doi.org/10.1145/3637318) shows that
displaying uncertainty alone may be insufficient; how it is calibrated and
communicated affects whether people adjust their reliance.

Movement coaching adds a temporal requirement to these concerns. A learner
must know not only why a correction was produced, but also which repetition or
movement episode it describes. Delayed feedback can lose this referent, and a
detailed explanation can amplify an incorrect action label if the underlying
video evidence is not inspectable. For AI-assisted sports coaching,
interpretability therefore includes temporal provenance, concise corrective
language, access to post-session evidence, and a way to disagree with or defer
uncertain output. These requirements motivate our focus on cue timing,
recognition failures, and the division between immediate action and later
reflection, rather than measuring trust as an unconditional preference for the
system.

### 2.4 Positioning the Present Work

Prior work collectively shows that fencing actions can be recognized from
multimodal or 2D pose data, continuous practice can be segmented, weapon and
tactical patterns can be visualized, and interactive feedback can support
movement learning across several sports. What remains unresolved is not simply
the absence of one more sensing configuration. Recognition-focused fencing
systems do not establish how model outputs should become sparse,
fencing-specific corrections, while learner-facing systems in other sports do
not resolve the timing, terminology, and failure consequences of fencing
practice.

AI Fencing Coach addresses this design gap with a native Android artifact that
connects phone-based pose estimation and action recognition to
fencing-specific heuristics, prioritized visual and spoken cues, and
timestamped post-session cue history. Unlike wearable and immersive approaches,
it does not require body-worn sensors or a headset; unlike recognition and
visualization systems, its main unit of design is the feedback loop from
movement to correction and review. Our contribution is deliberately bounded:
we study the feasibility and breakdowns of this interaction, including whether
learners can understand and act on cues and how timing or misclassification
affects interpretation. We do not claim that the current prototype establishes
long-term skill improvement, replaces coaching judgment, or provides complete
technical and tactical assessment.

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

AI Fencing Coach is designed to reorganize solo video practice into a
continuous **notice-interpret-act-verify** loop. In ordinary self-review, the
learner must remember a movement, inspect the recording, decide what is wrong,
and translate that judgment into a correction. The prototype distributes this
work across three feedback layers: a live camera view makes the analyzed body
and action visible, short visual and spoken cues support the next attempt, and
a persistent review record supports slower interpretation after practice. Its
core interaction mechanism is therefore not action classification alone, but
the conversion of a continuous practice stream into **sparse correction during
movement and inspectable evidence after movement**.

This design follows the formative findings in Section 3. P1 could already
diagnose several gross errors, whereas P2 needed help naming a problem; both
could respond to concise directional cues, but both had difficulty grounding
delayed feedback in a specific action. Detailed review helped explain
unfamiliar issues, yet a false defensive-action label also showed that detail
can amplify an incorrect inference. We consequently designed the system to
reduce diagnostic work without treating the AI as the final authority. The
five goals below operationalize DR1-DR5 as intended interaction properties,
not evidence that the current prototype improves technique.

### 4.1 Design Goals

**DG1: Add diagnostic support without replacing self-assessment.** The system
should help learners notice and name movement features while preserving their
ability to compare the output with what they felt and saw. During practice, it
shows the camera image, optional skeleton overlay, current action state, and
detected issues rather than returning only a hidden score. After practice, it
retains recognized-action counts and a cue timeline so that a learner can
compare automated feedback with memory and other available evidence. This
design addresses the different starting points observed in P1 and P2:
assistance can add a label or priority for one learner while serving as
confirmation or a point of disagreement for another.

**DG2: Separate immediate action from later explanation.** Live feedback
should answer a narrow question: "What should I change on the next attempt?"
The system therefore uses short directional phrases such as "Stay lower" or
"Shorten the step," speaks at most one correction at a time, and keeps no more
than three issues visible. Diagnosis, recurrence, and practice suggestions are
deferred to the review layer. This division reflects the interviews, in which
step-size and balance cues supported immediate adjustment, while arm and
defensive-action feedback required more explanation. It also treats low
interruption and full understanding as separate outcomes rather than assuming
that a short cue accomplishes both.

**DG3: Preserve the temporal provenance of feedback.** A correction is useful
only if the learner can identify the movement episode it describes. The
prototype records each accepted cue with a frame index, converts that position
into a relative session time, and exposes recent and post-session timelines.
The selected-video mode similarly preserves frame-level analysis states for
optional annotation. These mechanisms support retrospective grounding, but
the live spoken cue does not yet name a repetition, action, or timecode. DG3
is therefore only partially realized: the system preserves when it produced
feedback without yet guaranteeing that the learner can connect that feedback
to the intended movement.

**DG4: Make uncertainty and failure visible enough to question.** The
interface distinguishes model loading, target search, stance checking, active
analysis, pause, and review states. When a non-idle action is recognized, the
live action label includes the classifier confidence, and the interface also
reports processing rate and latency. These states expose more of the
recognition process than a single authoritative diagnosis. However, confidence
is not propagated to each biomechanical cue, low-confidence action recognition
does not provide a complete abstention mechanism for all heuristics, and a
learner cannot yet dismiss or correct an individual output. The intended goal
is contestability; the current design provides partial inspectability.

**DG5: Preserve learner and coach authority.** The learner can pause analysis,
silence speech, resume or reset a session, emphasize or mute error categories,
restrict feedback to selected categories, choose whether a generated summary
is used, and delete stored sessions. The deterministic fencing playbook remains
available when no language-model service is selected or a request fails. These
controls make automation configurable rather than compulsory. The system is
nevertheless a practice aid, not a coach replacement: it currently provides no
coach-facing validation or correction workflow, and its rule thresholds should
not be interpreted as universal definitions of correct or safe technique.

### 4.2 Interaction Flow

The Android application supports two entry points into the same feedback
model: **Realtime** for feedback during practice and **Postgame** for analysis
of a selected video. A separate **History** area supports comparison across
saved sessions, while **User Settings** controls the practice mode, target
position in the frame, pose engine, language, speech, feedback focus, skeleton
overlay, optional generated summaries, and video export. This separation lets
the learner configure the attention and privacy tradeoffs before entering the
camera-first interaction.

In Realtime mode, the learner first confirms the current configuration and
starts the back camera. The system searches the frame for one intended fencer;
when two people are visible, the configured left or right position determines
the initial target and the other person may be retained as opponent context.
The interface then moves through three analysis states. In `IDLE`, it searches
for a suitable fencing stance at a reduced rate. In `CHECKING`, it waits for a
short sequence consistent with en garde rather than treating one bent-knee
frame as practice. In `ACTIVE`, movement frames enter the action and
biomechanical analysis pipeline. Returning to a standing posture, turning away,
or losing the target eventually returns the system to the idle state. Short
pose dropouts are bridged to avoid immediately switching identity or clearing
the practice context.

During active practice, the camera and skeleton occupy the upper part of the
display. An action indicator reports the current recognized action and, for
non-idle predictions, its confidence. The lower feedback panel presents the
highest-ranked issue as primary and can retain two additional issues as
secondary visual cues. If speech is enabled, the scheduler releases one
eligible issue through Android text-to-speech; this may differ from the first
visual issue when that issue is still in its speech cooldown. The learner does
not need to acknowledge every cue before continuing; they can make another
attempt, pause analysis, switch speech on or off, reset accumulated state, or
finish the session. Recent cue labels remain visible so that a spoken message
is not the only record of what the system reported.

Finishing practice freezes the live analysis and opens a review. The report
summarizes elapsed and active time, number of model checks and cues, the most
frequent recognized action, action counts, recurring issues, and a relative
cue timeline. Each issue can carry three levels of text from the fencing
playbook: a short cue, a diagnosis, and a suggested drill. The learner may
resume the same session, begin a new one, or return home. The report is saved
in an on-device session database, where history can be filtered by user,
opened at the session level, summarized across recent sessions, selected for a
custom recap, or deleted.

Postgame mode offers the same interpretive layers without asking the learner
to attend during movement. The learner selects one or more videos; the system
queues them, samples frames across each clip, and applies the same target,
action, heuristic, scheduling, and report-building logic used in Realtime
mode. Processing progress is shown before the report appears. The user may
optionally export an annotated copy containing the analyzed visual state. This
shared pipeline is intended to make the two modes complementary: live cues
support the next repetition, while video review supports evidence inspection
and explanation.

Across both modes, the interaction is deliberately iterative rather than
terminal. A session does not produce a pass/fail grade. It produces a bounded
set of observations that the learner can act on, revisit, compare with later
sessions, or reject. This is important for fencing, where the value and safety
of a correction depend on drill, skill, weapon context, and coach intent.

### 4.3 Feedback Timing and Prioritization

The formative study showed that speaking more quickly is not sufficient if the
learner still cannot identify the referenced action. Our design therefore
treats feedback timing as a coordination problem among detection, ranking,
presentation, and the next movement. Movement must first be recognized over a
short temporal window; biomechanical checks may then yield several concurrent
issues; visual attention and speech can present only part of that set. The
system must decide both **what** to present and **when** presenting it remains
useful.

The scheduler first filters issues by practice mode and the learner's muted or
selected categories. It then ranks the remaining issues using a playbook
weight together with persistence, novelty, time spent waiting behind other
issues, repeated presentation, and learner emphasis. Persistence raises an
issue that continues across checks; novelty favors a problem that has not yet
been spoken; aging prevents a lower-ranked issue from waiting indefinitely;
and a repetition penalty reduces domination by an issue already presented
several times. Learner-emphasized categories receive an additional boost.
This policy operationalizes prioritization as a changing practice state rather
than a fixed severity list.

Visual and spoken feedback use different attention budgets. Up to three
pending issues can remain visible, allowing the learner to inspect co-occurring
problems without hearing them all. Speech is serialized to one issue and is
rate-limited both globally and per error type. In the current prototype, an
error must wait at least 1.2 seconds after any spoken cue, the same error has a
4-second speech cooldown, and a recently detected issue can remain pending for
up to 5 seconds. These are implementation choices intended to reduce
interruption and starvation; they are not empirically validated timing
requirements.

The pending interval creates an important tradeoff. It lets a high-priority
cue survive a brief detection gap and lets skipped issues eventually surface,
but it can also produce exactly the ambiguity observed by P1 and P2: a
technically valid message may be delivered after the movement that triggered
it. The current interface partly repairs this after the fact through recent
cue labels and a timestamped timeline. It does not yet repair it during
practice because speech lacks an explicit action or repetition referent.
Future versions should compare immediate delivery, delivery at a detected
movement boundary, and deferral to review; should report capture-to-cue
latency rather than model inference time alone; and should test whether
learners can identify the referenced repetition.

The design also avoids using the live channel for full explanations. The
playbook's diagnosis and drill text are stored with the issue and exposed in
review, where the learner has time to compare recurring cues and decide what
to practice. An optional generated summary may reorganize the aggregate action
counts and the three most frequent playbook-grounded problems, but it does not
create the live detection evidence. This layered timing preserves a
deterministic path from detected issue to short cue to later explanation.

### 4.4 Safety, Uncertainty, and Coach Authority

AI Fencing Coach analyzes a bounded set of visible two-dimensional movement
features and six fencing-action classes. It does not observe blade contact,
force, pain, tactical intent, fatigue, floor conditions, or the coach's
exercise-specific objective. Consequently, a reported issue is a
prototype-level inference about an implemented rule, not a complete assessment
of fencing quality or safety. In particular, the live interface currently
uses "Good Technique" when no visual issue is scheduled. This state should be
read only as "no implemented rule is currently being presented"; it is not
evidence that the movement is correct. Similarly, reported action counts
summarize classified temporal windows and should not be interpreted as
verified counts of distinct fencing repetitions.

The connected camera, pose, action-recognition, heuristic, speech, and report
path runs on the phone. Raw camera frames and skeletons are not sent to a
remote service for the core coaching loop, and saved session records contain
aggregate counts, cue times, issue labels, and summaries rather than the
camera stream itself. If the learner enables Gemini or OpenAI summaries, the
application sends the user name, practice mode, target-side setting, aggregate
action counts, and the most frequent playbook-grounded issues to the selected
provider. A deterministic playbook summary is created first and remains the
fallback for missing configuration, network failure, quota failure, or an
empty response. This optional network boundary must be disclosed separately
from the on-device analysis path.

The summary prompt constrains the language model to supplied counts and
playbook entries and instructs it not to invent injuries, tactics, timecodes,
opponent behavior, or unseen technique. These constraints reduce the space of
possible claims but do not validate the underlying detections or guarantee
that generated wording is appropriate. Generated detail can still make a
false action or heuristic output appear more authoritative. For this reason,
the report preserves the detected categories and counts that grounded the
summary, and the playbook-only report remains available.

The prototype currently exposes uncertainty unevenly. It shows action
confidence and distinct loading, search, checking, active, and paused states,
and it can fall back to `Idle` when action confidence is below the classifier
threshold. However, pose quality, target interpolation, action confidence,
and heuristic evidence are not combined into a cue-level confidence estimate.
The system also cannot yet abstain selectively from a questionable cue, ask
the learner to confirm an action, or let the learner mark one diagnosis as
incorrect. Category-level muting is useful for configuring feedback, but it is
not a substitute for contesting a specific false positive.

We therefore define three authority boundaries for subsequent iterations.
First, uncertain evidence should produce abstention or an explicitly uncertain
state rather than confident coaching. Second, learners should be able to
dismiss, correct, or flag a cue and retain that disagreement in the review
record. Third, coaches should be able to inspect the linked movement evidence,
validate or override the diagnosis, and configure which rules are appropriate
for a drill. None of these boundaries is fully implemented in the current
prototype, so Sections 6-8 evaluate feasibility and breakdowns rather than
claiming autonomous coaching correctness.

Finally, learner control does not by itself establish physical safety. The
current application does not verify warm-up, free space, protective equipment,
injury status, or whether listening during a fast drill is appropriate.
Real-time use should therefore be limited to controlled solo practice where
brief audio is safe, with higher-speed or tactically complex activity reviewed
afterward. The next section describes the technical mechanisms through which
the current Android prototype realizes these design commitments and where
those mechanisms remain incomplete.

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
  evidence shape appropriate reliance on AI-generated fencing feedback?

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
| RQ3 | Appropriate reliance | Acceptance, questioning, or rejection of system diagnoses | Ordered interview records; feedback-trust rating |
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

## 8. Preliminary Study Results (4 Participants)

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

This study suggests that AI feedback can add useful diagnostic structure to
solo fencing practice, but not that automated coaching improves fencing skill.
Participants valued the system most for making errors easier to notice and
revisit, and both interviewees reported acting on concise step-size or balance
cues. However, usefulness depended on a longer interaction chain than accurate
recognition alone: the learner had to notice the cue, identify the movement it
described, understand the correction, attempt a change, and inspect what
happened next. Delayed delivery and incorrect action labels broke this chain
even when the wording of the advice was plausible. We therefore interpret the
prototype's contribution as support for a temporally grounded and contestable
practice-feedback loop, rather than as an autonomous source of technical
judgment.

### 9.1 How AI Feedback Fits Solo Fencing Practice

For RQ1, the findings indicate that AI feedback can reduce the diagnostic work
of self-review, but its added value depends on what a learner can already see.
P1 independently identified conspicuous center-of-mass and step-width problems,
whereas P2 struggled to name a problem before interviewer prompting. After
system use, both discussed concrete movement categories, and the aggregate
ratings for error awareness, advantage over self-review, and post-review
usefulness were high. The mechanism is not simply that the system reveals
otherwise invisible errors. It narrows a continuous video into candidate
problems, supplies fencing-specific labels, and preserves recurring issues for
later inspection. For a learner like P2, this can provide an initial diagnostic
vocabulary; for a learner like P1, it may refine, prioritize, or challenge an
existing judgment.

This distinction prevents an overly deficit-oriented account of solo learners.
Self-review was not uniformly ineffective, and the prototype did not always
surface the issue a participant considered most important. P1 did not recall
hearing an explicit cue for the vertical movement they had already identified.
AI support should therefore add evidence to self-assessment rather than assume
that the learner has no useful awareness. The current Android design responds
to this tension through visible action and error labels, cue history, session
reports, and controls for emphasizing or muting categories. These features
express a design hypothesis about configurable diagnostic support; they were
not independently evaluated in the preliminary study.

For RQ2, actionability arose from compression rather than completeness. Short
directional cues such as adjusting step size or center-of-mass position gave
both interviewees a plausible next action, while neither described earphone
speech itself as a major interruption. More technical arm and defensive-action
issues were harder to resolve from audio alone, and P1 valued post-session
analysis because it could provide detail that would be impractical during
movement. This explains the contrast between comparatively positive ratings
for actionable correction and low interruption and lower ratings for
correction understandability. A cue can be brief enough to act on without
being sufficiently explanatory for reflection.

The live and review channels consequently perform different cognitive work.
Live feedback reduces the next decision to one correction; post-session review
restores the evidence, terminology, recurrence, and practice rationale omitted
from that correction. This finding aligns with prior HCI work showing that the
timing, form, quantity, movement complexity, and athlete's exercise-specific
skill shape the value of feedback in
[rhythmic gymnastics](https://doi.org/10.1145/3613904.3642434). Our results
extend that account by showing that, for automated movement feedback,
**temporal provenance** is also part of explanation: a learner may understand
every word yet still be unable to use the advice if the referenced repetition
is unclear.

For RQ3, appropriate reliance was shaped less by a general attitude toward AI
than by whether a specific cue could be reconciled with the learner's memory
and the available video. P1 could reinterpret one hand-motion diagnosis as
plausible, whereas P2 rejected a defensive-action label because no defensive
action had occurred. These cases show calibrated responses rather than either
blanket acceptance or rejection. They also expose a consequential failure mode:
classification errors propagate into coaching errors. If the inferred action
is wrong, a well-phrased correction may be irrelevant, and a detailed report
may increase rather than reduce its apparent authority.

This result reframes the relationship between fencing recognition and coaching.
Systems such as
[FenceNet](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Zhu_FenceNet_Fine-Grained_Footwork_Recognition_in_Fencing_CVPRW_2022_paper.html)
demonstrate that fine-grained fencing actions can be modeled from skeleton
sequences. Our findings show that recognition is enabling infrastructure, not
the final interaction outcome. A coaching system must additionally manage when
an inference is presented, how it is linked to evidence, what happens when it
is uncertain, and how a learner can disagree. In this sense, the unit of design
is not an action label or a cue but the complete feedback loop from movement to
inspectable correction.

### 9.2 Design Implications for AI-Assisted Sports Coaching

The findings and current implementation motivate four implications for
camera-based systems that provide feedback across repeated movement episodes.
These implications transfer at the level of interaction mechanism, not as
evidence that the same rules or outcomes apply across all sports.

**Ground feedback in a movement episode.** Timing should be evaluated by
whether users can identify the action that produced a correction, not only by
model inference time or average delay. The current Android pipeline makes this
challenge concrete: action recognition requires a temporal window, and the
scheduler may hold or serialize detected issues to reduce repetition and
interruption. These choices can improve stability while weakening the link
between a cue and the movement that triggered it. Frame-indexed cue history and
post-session timelines support retrospective grounding, but spoken feedback is
not yet bound to a repetition identifier. Movement-feedback systems should
therefore report capture-to-cue latency, mark the relevant repetition or replay
segment, and defer a correction when its live referent can no longer be made
clear. Whether action-bound delivery improves correction remains to be tested.

**Layer immediacy and explanation.** Real-time channels should prioritize one
next action, while reflective channels should preserve the evidence and detail
needed to judge that action. The Android scheduler operationalizes this
principle by speaking one ranked cue, retaining a small visual set, and moving
diagnosis, practice suggestions, recurrence, and timelines into review. This
division manages a tradeoff between attentional load and explanatory depth:
adding detail during movement may interrupt performance, while removing detail
afterward can turn advice into an unsupported command. Systems should therefore
evaluate live actionability and later understanding separately rather than use
a single clarity or usability score for both.

**Adapt support to baseline diagnostic ability.** The contrast between P1 and
P2 suggests that the useful amount and type of feedback cannot be inferred
from detected error count alone. A system should first establish what the
learner notices unaided, then assess which feedback layer adds information.
Learners who can identify gross errors may benefit more from confirmation,
prioritization, temporal evidence, or uncommon technical details; learners who
struggle to self-diagnose may need naming and demonstration before
personalization. The current emphasis, mute, and focused-error controls offer
learner configuration, but they require prior knowledge about which categories
matter. Future designs should combine such control with coach-configured goals
or an explicit baseline task, while avoiding the inference that more automated
feedback is always better.

**Make uncertainty contestable at the level of the correction.** Showing an
action-confidence value is useful but insufficient when the user ultimately
acts on a biomechanical cue. Cue-level reliability depends on pose quality,
target identity, action evidence, heuristic assumptions, and delivery time.
The current interface exposes action confidence and system state, yet it does
not combine these signals into cue-level uncertainty or let the learner
dismiss, correct, or annotate a specific diagnosis. Systems should abstain
when the evidence needed for a correction is weak, preserve the source episode,
and allow disagreement to become part of the record. Evaluation should measure
appropriate reliance, including acceptance of supported feedback and rejection
of false feedback, rather than treating increased trust as an unconditional
success.

Together, these implications shift optimization from maximizing the number of
detected issues to maintaining a usable feedback relationship. A lower-volume
system that delivers one grounded, inspectable correction may support practice
better than a more sensitive system that reports several uncertain problems.
This remains a design hypothesis because the present study did not compare
scheduling policies, feedback layers, uncertainty displays, or correction
controls.

### 9.3 Relationship to Coaches and Existing Practice

AI Fencing Coach is best positioned between unaided self-review and continuous
expert observation. Its plausible role is to provide consistent attention to a
bounded set of visible features, remember repeated cues across a session, and
offer low-level prompts when a coach is unavailable. These capabilities may
help learners arrive at later coaching interactions with a more concrete
question or a timestamped example. The study did not examine coach-learner
communication, however, so this redistribution of diagnostic work is a future
deployment hypothesis rather than an observed benefit.

Coaches remain necessary for judgments the prototype cannot make. They can
decide which error matters for a particular learner and drill, determine
whether an attempted correction is technically and physically appropriate,
interpret blade interaction and opponent response, and recognize when fatigue,
injury, tactics, or safety should override a generic rule. The system's
single-camera 2D evidence and fixed heuristic thresholds cannot supply this
situated judgment. A state in which no implemented error is presented should
therefore mean only that the current pipeline has no cue to show, not that the
movement is correct or safe.

The current Android implementation partially preserves human authority.
Learners can pause analysis, disable speech, emphasize or mute categories,
restrict the feedback set, inspect session history, and retain a deterministic
playbook summary when optional language-model generation is disabled or fails.
It does not yet provide a coach-facing workflow for validating detections,
changing rule thresholds, correcting a session record, or configuring feedback
for an individual training plan. Category-level muting also cannot replace the
ability to contest one false positive. A role-compatible deployment should let
coaches review the linked episode, override the diagnosis, and make that
correction visible in subsequent reflection.

Optional generated summaries introduce a further authority and privacy
boundary. The core camera, pose, action, heuristic, and scheduling pipeline
runs on the phone. When a cloud summary is enabled, the current app sends
derived session information, including the user name, action counts, and
playbook-grounded problems, rather than raw camera frames. Prompt constraints
and deterministic fallback reduce unsupported elaboration, but they cannot
repair incorrect sensor evidence or guarantee that fluent wording will be
interpreted cautiously. Deployment should clearly separate local analysis from
optional network processing, minimize identifying information, support
informed opt-in, and keep the underlying counts and detections inspectable.

These role boundaries matter beyond fencing. In repetitive physical practice,
automated systems can offer consistency and memory, while human experts provide
context, norm setting, safety judgment, and responsibility. The goal is not to
simulate the total authority of a coach, but to design a useful handoff between
machine observation, learner interpretation, and expert review. The present
evidence supports this framing for a short, supervised fencing workflow and
identifies timing and recognition failures as central interaction problems. It
does not establish objective technique improvement, superiority to self-review,
safe unsupervised use, sustained adoption, or transfer across fencers, coaches,
settings, and sports. Those broader claims require technical validation,
controlled comparison, coach assessment, and longer-term field evidence.

## 10. Limitations and Future Work

### 10.1 Study Limitations

The preliminary study provides a small and uneven evidence base. Four
participants contributed questionnaire ratings, but detailed interview records
were retained for only P1 and P2. This is sufficient to document contrasting
cases and identify consequential breakdowns, but it cannot establish how common
those cases are, support subgroup comparisons, or claim qualitative saturation.
Recruitment source, inclusion criteria, age, gender, fencing experience,
compensation, and the researchers' prior relationships with participants are
also absent from the supplied artifacts. We therefore cannot characterize the
sample as representative of novice, intermediate, expert, or wider fencing
populations. A subsequent study should use an explicit sampling strategy,
report participant and researcher characteristics relevant to interpretation,
and include fencers with varied skill levels and coaches whose technical
judgment affects the intended practice.

The retained questionnaire file contains only item labels, means, and
`N=4`. It does not preserve the exact wording, response anchors,
participant-level responses, missing values, or dispersion. The reported means
can indicate which aspects were rated relatively more or less positively in
this sample, but they are not population estimates and cannot support
significance tests, confidence intervals, or claims about individual
consistency. The two qualitative files are ordered question-and-answer
summaries rather than complete verbatim transcripts. Reorganization,
translation, and loss of vocal or interactional context may have removed
ambiguity or disagreement important to the analysis. Future studies should
retain the full instrument, de-identified participant-level responses,
verbatim transcripts, translation procedures, and an auditable coding record.

The study sequence was fixed: unaided self-review always preceded real-time
feedback, a feedback-guided attempt, and post-session analysis. Repetition,
growing familiarity with the movement set, novelty, fatigue, and researcher
explanation are therefore alternative explanations for later perceptions or
reported adjustments. P2's baseline center-of-mass diagnosis followed an
interviewer prompt, developer explanations were interleaved with P1's session,
and P2's final post-attempt response was incomplete. These deviations weaken
the independence of the baseline and make the study unsuitable for claiming
that AI assistance was superior to self-review. The current evidence instead
supports the narrower observation that participants encountered actionable
cues and interpretable breakdowns within this fixed, supervised workflow.

Most importantly, the study did not measure whether the attempted corrections
were technically correct or durable. It includes no coach-blinded rating,
kinematic ground truth, participant-level pre/post performance measure, delayed
retention test, or transfer to an unpracticed drill. Reported action on a cue
and perceived improvement therefore should not be treated as evidence of skill
learning. The sessions were also short and supervised, so they do not establish
independent use, long-term adoption, changing reliance, or physical safety in
ordinary practice. These larger claims require objective movement assessment
and longitudinal evidence rather than additional preference ratings alone.

Coaches were not included in the retained sample, although the contribution
depends on preserving coach authority and technical validity. The study also
does not document the applicable ethics-review status, consent process, video
retention policy, withdrawal procedure, or safeguards for physical activity.
Those omissions must be resolved from original records before publication.
Absent those records, the paper can describe the analyzed artifacts and use
pseudonymous identifiers, but it cannot make a complete claim about the
ethical governance of data collection.

### 10.2 System Limitations

The prototype analyzes a single commodity-camera view using two-dimensional
pose estimates. Its outputs may change with camera angle and distance, lighting,
clothing, body proportions, occlusion, handedness, movement speed, or motion
outside the image plane. Short target dropouts are bridged through
interpolation, which supports continuity but may also preserve an inaccurate
trajectory. Although the tracker can retain one fencer and optional opponent
context, the pipeline has not been validated for crowded pistes, frequent
crossing, weapon occlusion, or arbitrary camera placement. The current artifact
should therefore be understood as a side-view practice prototype, not a robust
general-purpose fencing observer.

FenceNet recognizes a bounded set of six action classes, and the heuristic
layer checks twelve implemented error categories using fixed geometric and
temporal thresholds. These rules cannot assess complete technique, tactical
intent, blade contact, force, distance to an opponent, drill-specific goals, or
all safety-relevant conditions. The repository contains component tests, but
this paper does not yet report a representative benchmark of pose quality,
action classification, heuristic precision and recall, false-feedback
frequency, calibration, or performance across devices and practice conditions.
Consequently, the study demonstrates a connected feedback workflow but does
not establish that its detections are accurate enough for unsupervised
coaching.

Temporal ambiguity is partly produced by the architecture itself. Action
classification requires a multi-frame window, inference is performed at a
stride, and the scheduler can hold and serialize issues using persistence and
cooldown rules. These mechanisms stabilize feedback and reduce repeated speech,
but they can delay a cue beyond the action that generated it. Cue history
preserves frame positions for later review, yet live speech does not identify a
repetition or movement boundary. The system therefore records temporal
provenance without ensuring that users can interpret it during practice.
Current scheduler constants are engineering choices rather than empirically
validated attention or learning parameters.

Uncertainty is also exposed incompletely. The interface displays action
confidence and analysis states, and low-confidence action predictions can fall
back to `Idle`; however, it does not propagate pose quality, target
interpolation, action evidence, and heuristic assumptions into a cue-level
confidence estimate. Nor can a learner dismiss or correct one cue, confirm the
performed action, or retain disagreement for later coach review. Muting an
error category changes future scheduling but does not contest a specific
diagnosis. An incorrect action label can therefore continue into an otherwise
coherent cue or report. This limits the system's support for appropriate
reliance even when its general confidence display is visible.

The core camera-analysis loop runs on the device, and session history stores
derived counts, cue records, settings, and summaries rather than the raw live
camera stream. Nevertheless, the prototype does not provide a complete privacy
or security model for local history, exported annotated video, user-entered API
keys, or data deletion beyond session-level controls. If a learner enables an
optional Gemini or OpenAI summary, derived session information including the
user name, configuration, action counts, and detected problems is sent to the
selected provider. Prompt constraints and deterministic fallback limit the
summary's evidence, but they cannot correct false detections or guarantee
appropriate wording. Deployment would require explicit consent, data
minimization, secure credential storage, provider disclosure, retention
controls, and evaluation of whether generated language changes reliance.

Finally, the application does not determine whether a learner has warmed up,
has sufficient free space, is wearing appropriate equipment, is injured, or
can safely attend to speech during a drill. It has not been evaluated for
accessibility across hearing, vision, language, motor ability, or device
performance, nor for compatibility with established coaching workflows.
Pause, speech, display, and feedback-focus controls provide useful agency, but
they do not by themselves establish safe or inclusive use.

### 10.3 Claim Boundary

The following table summarizes the boundary between what the present evidence
supports and what remains untested.

| The current evidence supports | The current evidence does not support |
| --- | --- |
| The Android prototype can deliver live and post-session fencing feedback in a short supervised workflow. | The system is technically reliable across fencers, devices, viewpoints, and practice conditions. |
| Four participants rated error awareness and post-session review positively, and two interviews illustrate how the feedback was interpreted. | The reported means generalize to a fencing population or demonstrate superiority to video self-review. |
| Some concise step-size and center-of-mass cues were understandable enough to prompt reported adjustments. | Those adjustments were biomechanically correct, retained over time, transferred to other drills, or improved fencing performance. |
| Delayed cues and action misclassification were consequential interaction failures in the retained cases. | Current latency, false-feedback, and calibration levels are acceptable for independent practice. |
| Temporal grounding, layered explanation, baseline-sensitive support, and contestability are evidence-backed design requirements. | The current implementations of those requirements have been validated across users, coaches, settings, or sports. |
| The artifact is a plausible supplement for moments when continuous expert attention is unavailable. | The artifact can replace a coach, make complete safety judgments, or act as an independent authority on technique. |

### 10.4 Evidence-Driven Future Work

Future work should first establish the reliability of the evidence from which
feedback is generated. A technical benchmark should sample fencers across skill
levels, body types, handedness, clothing, movement speed, camera angle,
distance, lighting, occlusion, and relevant phone hardware. Expert annotation
should identify action intervals and error episodes so that the evaluation can
report per-class confusion matrices, heuristic precision and recall,
false-feedback frequency, latency distributions from capture to presentation,
confidence calibration, target-tracking failures, and abstention performance.
This study would determine which cues are sufficiently reliable for live use
and which should be restricted to review or withheld.

Once technical performance is characterized, a counterbalanced interaction
study should compare unaided video self-review with AI-assisted feedback using
equivalent movement sets. It should measure baseline diagnostic ability,
cue-referent identification, comprehension, interruption, next-attempt
correction, and coach-blinded technique ratings. Mechanism-level conditions
could compare immediate versus movement-boundary delivery, voice-only versus
layered review, and fixed versus learner- or coach-prioritized cues. This would
test whether temporal grounding and layering, rather than novelty or repeated
practice, account for any observed benefit.

A separate contestability study should deliberately include uncertain and
incorrect outputs. It should compare action confidence alone with cue-level
uncertainty, abstention, replay of the source episode, and controls to dismiss
or correct a diagnosis. Outcomes should include appropriate acceptance of
supported cues, rejection of false cues, recovery after error, understanding of
system limits, and alignment between learner and coach judgments. The goal
should be calibrated reliance rather than simply increasing trust.

Claims about learning require a longer study with coach-validated outcomes.
Participants should practice across multiple sessions and complete delayed
retention and transfer tests using an unpracticed drill. The evaluation should
track error recurrence, correction stability, dependence on feedback, and
whether learners improve their unaided ability to diagnose movement. Evidence
of immediate adjustment without retention or transfer would challenge a
skill-learning interpretation even if the system remained well liked.

Finally, a field deployment should examine how the system enters actual
coach-learner practice over several weeks. It should include coaches in setting
feedback priorities, reviewing uncertain episodes, overriding diagnoses, and
deciding when live audio is inappropriate. Logs should be interpreted alongside
observation and interviews to study feature abandonment, changing reliance,
privacy choices, false-feedback recovery, safety incidents, and effects on what
learners ask coaches and what coaches can observe. Only such deployment evidence
can establish whether the proposed handoff among automated observation,
learner interpretation, and expert judgment remains useful outside a short
supervised session.

## 11. Conclusion

We presented AI Fencing Coach, a native Android prototype for supporting
fencers when continuous coaching is unavailable. The system operationalizes a
layered feedback interaction: an on-device pipeline converts camera-based pose,
target tracking, action recognition, and fencing-specific heuristics into one
prioritized spoken cue and a small set of visual issues, while timestamped
post-practice reports preserve evidence for reflection. In a preliminary
mixed-method study with four participants, including detailed interviews with
two, participants rated error awareness, post-review usefulness, and advantage
over self-review most highly; the interviews indicated that brief step-size and
balance cues could guide reported next-attempt adjustments. At the same time,
delayed cues and action misclassification made otherwise understandable advice
difficult to connect to the movement that produced it. Together, these findings
identify temporal grounding, layered explanation, prioritization, and
contestability as requirements for interpretable AI movement feedback that
supports rather than replaces learner and coach judgment. Given the small,
partially retained corpus and absence of objective skill measures, this work
contributes feasibility evidence and design requirements rather than proof of
durable fencing improvement.

## Acknowledgments

## References
