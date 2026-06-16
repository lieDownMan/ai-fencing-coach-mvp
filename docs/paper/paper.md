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
positioning coaches as final authorities. We conducted a preliminary
mixed-methods study of an early prototype. Four participants contributed
post-use ratings, and retained records from two document unaided self-review,
AI-assisted practice, and detailed post-task interviews. Descriptive ratings
were highest for error awareness, post-review usefulness, and perceived
advantage over self-review (M=4.75 for each), and
lowest for timing accuracy and correction understandability (M=3.25 for each).
The two interviewees reported acting on step and center-of-mass cues; however,
delayed timing and action misclassification obscured which movement the
feedback referred to.
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
learner. In the retained unaided self-review records, the two interviewed
participants differed substantially in what they could infer from video alone.
Self-review helped one participant articulate a visible balance problem, while
the other remained unsure what to change. Video can show that something
happened without necessarily explaining what the movement means in fencing
terms or what the next repetition should look like.

The preliminary study suggests value in combining immediate and reflective
support. Among the 15 supplied aggregate questionnaire items, the highest means
were for the items labeled Error Awareness, Post-Review Usefulness, and Better
than Self-Review (`M=4.75` for each). Actionable Correction and Solo Practice
Support had means of `M=4.50`. In the interviews, both participants reported
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

1. **RQ1:** What differences emerge between the errors fencers identify through
   unaided video self-review and those they report after AI-assisted feedback?
2. **RQ2:** Which properties of live feedback make a correction actionable
   without unnecessarily interrupting practice?
3. **RQ3:** How do cue timing, action-recognition errors, and post-session
   evidence shape appropriate reliance on AI-generated fencing feedback?

To obtain initial evidence, we conducted a preliminary mixed-methods study of
an early prototype. Four participants contributed post-use ratings. Retained
records from two document unaided self-review, real-time feedback,
feedback-guided practice, post-session analysis, and detailed interviews. The
relative rating pattern suggests value in making errors visible and supporting
later review, while the two interviewees reported acting on several concrete
cues, especially those concerning step size and body balance, without
substantial interruption. At the same time, delayed cues and misclassified
actions made some feedback difficult to ground in a specific movement. We use
these results to characterize both the promise and the design requirements of
interpretable feedback, rather than as evidence of durable skill improvement.

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

## 2. Related Work

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

## 3. Formative Study and Analysis

Our formative work focused on the coaching knowledge and practice breakdowns
that should shape an AI fencing coach before evaluating whether the prototype
is useful. We asked four design questions: how coaches talk when they instruct
from a fencing video, what fencers find difficult during self-practice, how
coach language should be converted into system feedback, and how concurrent
errors should be queued and scoped by practice mode. This differs from a
simple "do users like the prototype" study. The formative goal was to decide
what the system should say, when it should say it, and which errors should be
eligible in different practice contexts.

### 3.1 Corpus and Participants

The formative corpus combines coach-facing and fencer-facing evidence. The
coach-facing material was an elicitation spreadsheet in which coaches were
asked how they would instruct actions observed in a fencing match video. The
locally retained design output of that analysis is the English coaching
playbook used by the Android system, `android/coach_playbook_en.json`. Each
playbook entry contains an error name, a diagnostic explanation, a short cue,
a practice suggestion, and a weight for cue queueing. The fencer-facing
material consists of two ordered Chinese-language interview records about
self-review and AI-assisted practice, plus aggregate post-use ratings from
four participants.

The coach form captured one timestamped submission per response and then
repeated the same instruction block for up to four errors observed in the
video. Each block asked for an error name, an explanation of the cause and why
it was incorrect, the oral cue the coach would say to the fencer, and a drill
or practice activity for improvement. The first three blocks used the order
error name, cause, oral cue, and practice; the fourth block retained the same
elements but placed the cause field after cue and practice. The external
spreadsheet should still be archived or exported into the repository before
submission so that coach count, experience, recruitment, video prompt details,
and all raw response examples can be reported precisely. The current paper
draft therefore treats the retained playbook and the supplied excerpted form
structure as verified, while leaving full coach sample metadata to be filled
from the original spreadsheet.

**Table 1. Retained formative evidence and its role in design.**

| Evidence source | Retained artifact | Role in the formative analysis |
| --- | --- | --- |
| Coach instruction elicitation | Timestamped Google Form spreadsheet; export needed for full publication record | How coaches name errors, explain causes, phrase oral cues, and propose practice drills |
| Coaching playbook | `android/coach_playbook_en.json` and bundled Android asset | Structured translation from coach instruction to diagnosis, short cue, drill, and weight |
| Fencer self-practice interviews | `interviewee_01_1Q1A_ordered.txt`; `interviewee_02_1Q1A_ordered.txt` | Pain points in self-diagnosis, cue interpretation, timing, and post-session review |
| Post-use ratings | `formative_study_feedback_statistics.csv` | Descriptive triangulation of perceived error awareness, actionability, interruption, trust, and review value |
| Android implementation | `FeedbackScheduler.kt`, `MainActivity.kt`, and playbook assets | Trace from formative requirements to cue queueing, mode filtering, and feedback controls |

The supplied fencer artifacts do not preserve recruitment source, age, gender,
fencing experience, compensation, session duration, or ethics-review and
consent procedures. The same care is needed for the coach elicitation once the
spreadsheet is exported. We therefore use the current evidence to support
design requirements and mechanism hypotheses, not population prevalence,
learning outcomes, or claims that the playbook represents all coaching styles.

### 3.2 Procedure and Materials

For the coach elicitation, coaches watched actions in a fencing match video
and filled the timestamped form described above. A single response could
therefore contain several observed errors from the same video, with each error
linked to a cause, an oral instruction, and a practice recommendation. The
analysis converted these responses into a playbook of twelve normalized error
categories. Each category keeps three levels of coach talk: a diagnostic
sentence for later explanation, a short imperative cue for live practice, and
a drill-like practice suggestion for follow-up work.

For the fencer-facing inquiry, participants performed short fencing movements
such as en garde, advance, retreat, lunge, and advance-lunge, then reviewed
video and discussed what they could identify without a coach. They then tried
the prototype's real-time earphone feedback and post-session video analysis.
These sessions are also reported later as preliminary feasibility evidence;
in this section, we use them only to identify self-practice pain points that
the coach-derived playbook and cue queue should address.

### 3.3 Analysis

We analyzed the coach-derived material by treating each completed error block
as an instruction unit with four primary components: the observed movement
problem, the cause and consequence of the error, the short correction a coach
would say, and the practice activity used to improve it. The raw responses
varied in granularity and tone. Some named broad categories such as step width
or center-of-mass position; others named more situated problems such as overly
forceful thrusting, dropping the weapon during attack, inaccurate feints, foot
alignment during footwork, or inward wrist rotation before attack. Oral cues
also ranged from technical prompts, such as relaxing the shoulder or using the
fingers to control the blade, to very short colloquial reminders. We
normalized these heterogeneous blocks into the playbook structure
`error_name -> diagnosis -> short_cue -> practice -> weight`.

We analyzed the fencer interviews with a complementary self-practice
breakdown frame: **notice**, **interpret**, **act**, and **verify**. This
captured whether a fencer could detect an error, understand what it meant,
choose a correction, and check whether the correction helped. We then compared
these breakdowns against the coach-derived playbook to ask where the system
should reduce self-diagnosis burden, where a short cue is enough, and where
post-session explanation is needed.

Finally, we examined the Android implementation as a traceability check. The
playbook weights are read by `PlaybookRepository` and used by
`FeedbackScheduler`. The scheduler combines the coach-derived weight with
dynamic factors such as persistence, novelty, aging behind other cues, repeat
penalty, and learner emphasis. Practice modes and user settings filter which
errors can appear. This implementation check prevents the paper from
describing a design rule that the current system does not actually enact.

### 3.4 Findings

#### Coaches do more than name an error; they connect diagnosis, cue, and drill

The playbook shows that coach instruction cannot be reduced to a label such
as "wide step" or "guard dropped." Each error needs a short cue for immediate
use, but also a reason and a follow-up practice action. In the supplied form
excerpt, for example, a coach described overly forceful thrusting as excessive
shoulder effort and cued the athlete to relax the shoulder and use the
fingers to control blade direction. Another response separated an attack
problem, an imprecise feint, a footwork-alignment problem, and a wrist-rotation
problem within the same timestamped submission, giving each a different cue
and, when available, a different practice response. This structure shows why
the system cannot treat coach knowledge as a flat checklist of errors.

The normalized playbook preserves that layered structure. "Stance Too High"
is not only cued as "Stay lower"; it is explained as reducing balance and
readiness to change direction and paired with advance-retreat sets that
maintain a bent front knee and quiet head height. "Foot Before Hand" becomes
the live cue "Hand first," but its diagnosis explains that the front foot
moves before the weapon arm creates threat, making the attack easier to read.
This became the basic design unit of the system: a live cue should be short,
while review should preserve the coach's diagnostic reasoning and practice
suggestion.

#### Fencers' self-practice pain point is translating video into action

The fencer interviews show that video self-review does not automatically
produce an actionable correction. P1 could identify visible posture and
footwork problems, especially vertical center-of-mass movement and narrow
steps, but still needed external support for more technical or unfamiliar
issues. P2 initially struggled to name a concrete problem and agreed with a
center-of-mass diagnosis only after prompting. This suggests that the pain
point is not simply lack of recording. It is the burden of deciding what is
wrong, what matters first, and how to change the next repetition.

The ratings are consistent with this interpretation. The highest retained
means were for Error Awareness, Better than Self-Review, and Post-Review
Usefulness (all `M=4.75`, `N=4`), while Timing Accuracy and Correction
Understandability were lower (both `M=3.25`). Participants valued the system
for surfacing and organizing errors, but the lower ratings show that a cue
still fails when its referent, timing, or correction logic is unclear.

#### Cue queueing should encode coaching priority, not only detection order

Several errors can be detected in the same movement window, but a learner
cannot use all of them at once. The coach-derived playbook therefore assigns
weights that act as a first approximation of pedagogical priority. The
highest weights emphasize foundational or safety-relevant form, including
stance too high (`10.0`), guard dropped (`9.7`), lunge overextension (`9.5`),
incomplete arm extension (`9.0`), and narrow step (`9.0`). Lower weights, such
as wide step (`4.0`) or over-parrying (`5.0`), can still matter but should not
always displace more foundational problems.

The Android scheduler turns these weights into a cue queue rather than a
fixed list. It boosts persistent errors, gives a novelty bonus to errors not
yet spoken, ages skipped errors so they do not starve, penalizes repeated
speech, and adds a focus boost when the learner emphasizes an error category.
This design treats coach priority as contextual: weight starts the ranking,
but persistence, repetition, and practice focus can change what is most useful
to say next.

#### Practice mode changes which coaching problems should be eligible

The coach and fencer evidence also argues against one universal feedback set.
The form responses themselves mix errors that belong to different practice
contexts: step width and foot alignment are footwork problems; attack release,
thrust control, hand timing, and arm extension belong more naturally to target
or attacking practice; feint precision, guard position, and parry compactness
may surface in freer tactical contexts. In footwork practice, the system
should therefore focus on stance, bounce, step width, and center-of-mass
control. In target practice, attack coordination and weapon-arm preparation
become more relevant, including hand-before-foot timing, complete arm
extension, lunge control, and guard position. In free bouting, the feedback
should stay broader and handle mixed movement and opponent context without
overloading the fencer.

The current Android implementation partially operationalizes this mode logic.
It defines three modes: Footwork, Target Practice, and Free Bouting. It also
restricts `foot_before_hand` and `incomplete_arm_extension` to Target
Practice, while most posture, guard, balance, step, and parry-related errors
remain available across modes. The settings screen further lets the learner
emphasize, mute, or restrict feedback to selected error categories. If the
final design requires each mode to focus only on a narrower error subset, the
mode filters in the scheduler and settings list should be tightened to match
the intended table before publication.

### 3.5 Design Requirements

We translated the formative analysis into five design requirements. Table 2
records how each requirement is supported and how it appears in the current
Android system. These implementation links establish traceability, not proof
that the requirement is fully satisfied in use.

**Table 2. From formative evidence to system requirements.**

| Requirement | Formative rationale | Current Android response | Remaining work |
| --- | --- | --- | --- |
| **DR1: Represent coach instruction as layered feedback.** | Coaches' instruction contains a diagnosis, short correction, and practice follow-up rather than only an error label. | The playbook stores `diagnosis`, `short_cue`, `practice`, and `weight` for twelve errors. | Archive coach responses and report examples showing how playbook entries were derived. |
| **DR2: Reduce self-practice diagnostic burden.** | Fencers may record video but still struggle to name what is wrong or what to change first. | Live cues name detected issues; review shows repeated errors, action counts, and cue history. | Test added value against unaided self-review with participant-level and coach-rated evidence. |
| **DR3: Queue cues by coaching priority and practice state.** | Multiple errors can co-occur, but coaches do not correct everything at once. | The scheduler combines playbook weights with persistence, novelty, aging, repeat penalty, and learner focus. | Validate the weights and queue order with coaches and measure whether learners can act on the selected cue. |
| **DR4: Scope feedback to practice mode.** | Footwork, target practice, and free bouting make different errors relevant and tolerable. | The app exposes three modes and partially filters mode-specific errors; users can focus or mute categories. | Tighten mode-specific eligibility if the final design requires each mode to show only its focused error subset. |
| **DR5: Preserve coach authority and learner contestability.** | Coach-derived rules guide feedback, but learners still encounter timing ambiguity and possible misclassification. | The app displays action confidence, stores cue history, supports playbook-only summaries, and lets users pause, mute, focus, or delete sessions. | Add coach review, cue dismissal/correction, and clearer uncertainty handling for questionable detections. |

These requirements position the formative study as a translation from coaching
practice to interaction design. The contribution is not that the current
prototype has proven learning gains. Rather, the formative work identifies
what an AI fencing coach should preserve from human instruction, which
self-practice pain points it should address, and where the current
implementation still diverges from the intended mode-specific feedback logic.

<!-- Insert design-traceability figure here.
Figure 2. Traceability from coach instruction elicitation and fencer
self-practice pain points to playbook entries, cue weights, practice-mode
filters, and unresolved implementation/evaluation questions. -->

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

This design follows the formative findings in Section 3. Coaches' video-based
instruction was translated into layered playbook entries: a diagnosis, a short
cue, a follow-up practice suggestion, and a queueing weight. Fencer interviews
then showed why this translation matters in self-practice: learners may have
video but still struggle to identify the error, choose what matters first, and
connect feedback to the next repetition. We consequently designed the system
to reduce diagnostic work while preserving the coach-derived rationale behind
each cue. The five goals below operationalize DR1-DR5 as intended interaction
properties, not evidence that the current prototype improves technique.

### 4.1 Design Goals

**DG1: Preserve coach instruction as a layered artifact.** The system should
not collapse coaching into action labels or scores. During practice, it uses
short playbook cues that resemble what could be said between repetitions. After
practice, it restores the longer diagnosis and practice suggestion so the
learner can see why the cue mattered and what drill to try next. This design
keeps live feedback lightweight while preserving the instructional reasoning
captured from coaches.

**DG2: Reduce self-practice diagnostic burden without replacing
self-assessment.** The system should help learners notice and name movement
features while preserving their ability to compare the output with what they
felt and saw. During practice, it shows the camera image, optional skeleton
overlay, current action state, and detected issues rather than returning only a
hidden score. After practice, it retains recognized-action counts and a cue
timeline so that a learner can compare automated feedback with memory, video,
and later coach judgment.

**DG3: Queue sparse feedback by coaching priority and practice state.** Live
feedback should answer a narrow question: "What should I change on the next
attempt?" The system therefore speaks at most one correction at a time, keeps
no more than three issues visible, and ranks candidate cues by playbook
weight, persistence, novelty, waiting time, repeated presentation, and learner
focus. This treats cue selection as a pedagogical queue rather than a raw
dump of all detected errors.

**DG4: Ground feedback in time and make failure visible enough to question.**
A correction is useful only if the learner can identify the movement episode
it describes and question it when the recognition is wrong. The prototype
records each accepted cue with a frame index, converts that position into a
relative session time, and exposes recent and post-session timelines. The
interface distinguishes model loading, target search, stance checking, active
analysis, pause, and review states. When a non-idle action is recognized, the
live action label includes the classifier confidence, and the interface also
reports processing rate. Per-frame state also records pose, classifier, and
total processing times plus estimated dropped frames for internal diagnostics,
but these latency values are not shown in the live interface. These states
expose more of the recognition process than a single authoritative diagnosis.
However, confidence is not propagated to each biomechanical cue,
low-confidence action recognition does not provide a complete abstention
mechanism for all heuristics, and a learner cannot yet dismiss or correct an
individual output. The intended goal is contestability; the current design
provides partial inspectability.

**DG5: Scope cues by practice mode while preserving coach and learner
authority.** The learner can choose Footwork, Target Practice, or Free Bouting,
pause analysis, silence speech, resume or reset a session, emphasize or mute
error categories, restrict feedback to selected categories, choose whether a
generated summary is used, and delete stored sessions. The deterministic
fencing playbook remains available when no language-model service is selected
or a request fails. These controls make automation configurable rather than
compulsory. The system is nevertheless a practice aid, not a coach
replacement: it currently provides no coach-facing validation or correction
workflow, and its rule thresholds should not be interpreted as universal
definitions of correct or safe technique.

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
summarizes elapsed and active time, model checks, retained report cues, the
most frequent recognized action, action counts, recurring issues, and a
relative cue timeline. Each issue can carry three levels of text from the
fencing playbook: a short cue, a diagnosis, and a suggested drill. The learner
may resume the same session, begin a new one, or return home. The report is saved
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

The formative evidence showed two related timing problems: coaches prioritize
what matters in a practice context, and learners must still identify which
movement a correction describes. Our design therefore treats feedback timing
as a coordination problem among detection, ranking, presentation, and the next
movement. Movement must first be recognized over a short temporal window;
biomechanical checks may then yield several concurrent issues; visual
attention and speech can present only part of that set. The system must decide
both **what** to present and **when** presenting it remains useful.

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
but it can also produce the ambiguity reported in the fencer interviews: a
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

We implemented the prototype as a native Android application in Kotlin and
Jetpack Compose (minimum Android API level 26). The application combines
CameraX for live capture, a YOLO ONNX model or selectable MediaPipe Tasks
backend for pose estimation, ONNX Runtime for fencing-action inference, Room
for local session storage, Android text-to-speech for spoken cues, and Media3
for optional annotated-video export. This Android application is the most
complete realization of the proposed system; the legacy Python components in
the repository are not part of the deployed mobile inference path.

The application has two video ingress paths. In live coaching, CameraX supplies
frames from the rear camera to a single analysis executor. It uses
`KEEP_ONLY_LATEST` backpressure so that inference does not accumulate an
increasing queue of stale frames. In post-session review, the application
samples a selected recording at approximately 30 frames per second and sends
the decoded frames through the same analysis pipeline. Reusing the pipeline
keeps the action labels and feedback rules consistent across immediate and
retrospective feedback, although the two paths have not yet been shown to have
equivalent timing or decoding behavior.

For each accepted frame, the pipeline performs:

1. pose extraction and conversion to a common skeleton representation;
2. target selection and short-gap tracking;
3. activity gating and body-centered spatial normalization;
4. windowed FenceNet action classification;
5. rule-based detection of fencing-form errors;
6. feedback prioritization, cooldown enforcement, and output; and
7. local aggregation of action counts, cue history, and pipeline diagnostics.

All pose, action, and heuristic analysis runs on the device. Network access is
optional and is used only to generate a post-session natural-language summary.
That request contains the stored profile name, selected practice mode and
target side, aggregate action counts, and the three most frequent error
categories with their playbook descriptions; it does not upload raw video or
pose trajectories. If the service is disabled or fails, the application
generates a deterministic local summary from the same playbook. This boundary
allows the main coaching loop and report generation to remain usable without a
remote model.

Figure 3 summarizes this architecture and distinguishes the implemented
mechanisms from user-facing configuration. In particular, target side,
practice mode, voice state, muted feedback categories, and a selected focus
category affect runtime behavior. Handedness, height, weight, and processing
profile are currently stored but are not used by the inference or heuristic
calculations; we therefore do not treat the current prototype as
anthropometrically personalized.

### 5.2 Pose and Action Analysis

The application defaults to a YOLO pose model executed through ONNX Runtime.
It letterboxes each frame to \(640 \times 640\), normalizes RGB values to
\([0,1]\), applies a 0.35 person-box threshold and a 0.35 keypoint threshold,
and performs non-maximum suppression at 0.45 intersection-over-union. Users
can instead select MediaPipe Pose Landmarker in video mode with the lite model,
at most two detected poses, and minimum detection and tracking confidence of
0.50. MediaPipe landmarks below 0.35 visibility are excluded from the mapped
skeleton. Both backends are mapped to a shared set of head, shoulder, elbow,
wrist, pelvis, knee, and ankle points. The YOLO mapping currently uses a fixed
right-side limb assignment whereas the MediaPipe mapping is target-side aware.
Results from the two backends should therefore not be pooled until their limb
semantics have been tested and aligned.

When multiple people are visible, the tracker initializes the target using the
leftmost or rightmost candidate according to the configured target side and
then favors track continuity and spatial proximity. It permits up to five
missing detections by extrapolating recent motion and rejects large frame-wise
position jumps. This supports brief occlusion but does not provide identity
recognition; a crossing opponent or bystander can still cause a target switch.

An activity gate reduces unnecessary downstream work and prevents obviously
non-fencing frames from entering the action buffer. In its idle state, pose
extraction is throttled to approximately 5 Hz. Motion can initiate a checking
state, after which five consecutive bent-knee posture indications activate the
pipeline. The gate returns to idle after 60 frames of standing posture, a
strongly turned-away body, or missing target evidence. A turned-away body is
detected when visible shoulder width is less than 5% of the frame width. These
constants are engineering settings rather than empirically calibrated fencing
thresholds.

For active frames, the spatial normalizer uses the nose position in the first
active frame as the origin and the vertical nose-to-front-ankle distance as the
scale. Nine two-dimensional joints produce an 18-channel skeleton vector. A
FIFO buffer holds 28 such vectors, and action inference runs every 10 accepted
frames. The deployed FenceNet model is a six-block causal temporal convolutional
network with channel sizes 32, 32, 64, 64, 128, and 128; kernel size 3; and
dilations 1, 2, 4, 8, 16, and 32. A final 64-unit layer predicts six action
classes: rapid lunge, incremental-speed lunge, waiting lunge, jumping-sliding
lunge, step forward, and step backward.

The classifier receives a tensor of shape \(1 \times 18 \times 28\). The
highest softmax probability is exposed as the action confidence, and
predictions below 0.60 are labeled `Idle` rather than assigned to one of the
six classes. This is an action-level abstention only: most biomechanical rules
can still emit feedback from the raw pose history when action confidence is
low. Moreover, report "action counts" count classification windows assigned to
each non-`Idle` class, not independently segmented fencing repetitions. Session
timestamps are currently derived from frame index under a nominal 30-fps
assumption. These representations are sufficient for prototype feedback and
relative session summaries, but they should not be interpreted as ground-truth
repetition counts or capture-clock-accurate timecodes.

### 5.3 Biomechanical Heuristics

In parallel with the normalized action buffer, the pipeline retains up to 60
recent skeletons in image coordinates. Twelve deterministic rules inspect this
history for form patterns that can be expressed with joint angles, relative
displacements, or body-scaled distances. Table 3 reports the implemented
conditions. The thresholds were selected to operationalize coach-relevant
concepts in the prototype; they have not yet been calibrated against
independent coach annotations.

**Table 3. Implemented biomechanical feedback rules and operating conditions.**

| Feedback rule | Implemented trigger | Context and current handling |
|---|---|---|
| Excessive bounce | Pelvis vertical range exceeds 0.33 of body-box height over at least five samples | Evaluated after a 15-cycle warm-up |
| Lunge overextension | Minimum front-knee angle is below \(120^\circ\) | Evaluated from recent raw poses |
| Dropped guard | Weapon wrist remains below the pelvis for more than five frames, or more than ten in free bout | Duration threshold varies by mode |
| Foot before hand | Front-ankle displacement peaks before wrist displacement; both exceed 0.01 body scale | Requires target-practice mode and an offensive action |
| Stance too high | Mean front-knee angle exceeds \(160^\circ\) over at least three samples | Fixed threshold across users |
| Incomplete arm extension | Weapon-elbow angle remains below \(155^\circ\) | Requires target-practice mode and an offensive action |
| Over-parrying | Maximum wrist sweep exceeds three times a shoulder-width proxy | Requires at least five samples |
| Step too wide | Front-to-rear ankle separation exceeds three times a shoulder-width proxy | Evaluated from the current pose |
| Step too narrow | Front-to-rear ankle separation is below 1.2 times a shoulder-width proxy | Evaluated from the current pose |
| Center of mass too far forward | Spine tilt exceeds \(15^\circ\), or shoulder tilt exceeds \(15^\circ\) | Geometric proxy rather than force measurement |
| Center of mass too far backward | Spine tilt is below \(-10^\circ\), or shoulder tilt is below \(-15^\circ\) | Geometric proxy rather than force measurement |
| Hand too high | Wrist-above-elbow angle exceeds \(60^\circ\) | Evaluated from the current pose |

The rules return categorical error keys rather than probabilities or
diagnostic traces. The scheduler then attaches playbook content and a dynamic
priority score. All twelve are deterministic Kotlin rules that share angle,
displacement, and body-scale utilities. Foot-before-hand and incomplete arm
extension are explicitly conditioned on both target-practice mode and an
offensive-action prediction. The other rules may run when FenceNet has
abstained to `Idle`. This design preserves form feedback when action recognition
is uncertain, but it also creates a path for out-of-context cues. A stricter
production design should jointly gate rules on pose quality, action coverage,
and temporal evidence.

### 5.4 Visual, Spoken, and Post-Session Feedback

The feedback scheduler converts simultaneous rule detections into a bounded set
of outputs. For each active error, it computes a priority score from a
playbook-defined base weight, an aging bonus of 2 per skipped opportunity, a
persistence bonus of 0.25 for up to eight active cycles, a novelty bonus of
0.75, a focus-category bonus of 4, and a repetition penalty of 1 for up to
three prior spoken instances. The scheduler removes muted or mode-incompatible
errors, displays at most three visual cues, and selects at most one cue for
speech. It enforces a 1.2-second global speech interval and a 4-second
per-error cooldown. An issue remains pending for up to 5 seconds after its
most recent detection, limiting how long stale advice remains eligible for
speech.

Labels, short corrective cues, diagnoses, drills, and base priorities are
loaded from a bilingual JSON playbook. During practice, the interface presents
the recognized action, confidence, processing state, and prioritized correction
cards. Android text-to-speech uses flush semantics so that a newly selected cue
replaces queued speech. Users can pause analysis, disable voice, focus on one
error category, mute categories, or terminate the session. When no visual rule
is active the interface shows "Good Technique"; this means only that no
implemented rule fired on the current evidence, not that the movement was
biomechanically correct.

The application logs a spoken cue, or otherwise the first visual cue, while
deduplicating the same label within a classifier stride. The live history keeps
five recent cues, the session timeline keeps up to 60, and the report displays
the 20 most recent entries. Room stores session metadata, aggregate
classification-window counts, and cue history. The post-session report combines
these records with playbook explanations and drill suggestions. If enabled,
annotated-video export renders the sampled analysis over the original recording
through Media3.

Several failure paths are handled explicitly. Missing pose or FenceNet assets
produce a model-unavailable state, optional language-model failure falls back
to the deterministic report, and short target loss is bridged by the tracker.
However, exceptions in the live camera analyzer are currently suppressed to
protect the session from a crash and are not surfaced to the learner. Better
diagnostic logging and a visible degraded-mode indicator are needed before the
system can support reliability claims.

<!-- Insert system overview figure here.
Figure 3. AI Fencing Coach pipeline from commodity-camera video through pose
extraction, target tracking, action recognition, fencing-specific heuristic
analysis, feedback prioritization, and visual, spoken, or post-session output. -->

## 6. Technical Evaluation

### 6.1 Current Verification and Evaluation Scope

We retain a separate technical-evaluation section because implementation
description and technical evidence answer different questions. Section 5
documents what the artifact does; this section states what has and has not been
verified. At the current prototype stage, the available evidence supports
Android build integrity, selected component behavior, and model-export claims,
but not pose accuracy, action recognition accuracy, biomechanical-error
accuracy, or real-time performance claims.

Table 4 summarizes reproducible checks on the artifact snapshot used for this
manuscript. The Android unit-test task completed 20 tests with no failures, and
the debug application assembled successfully. Three focused Python tests for
FenceNet and YOLO export contracts also passed. Comparing the deployed FenceNet
ONNX model with its PyTorch checkpoint on a synthetic tensor of the expected
shape generated with PyTorch seed 0 produced a maximum absolute logit
difference of \(2.98 \times 10^{-7}\). This establishes numerical fidelity for
that export check, not correctness of the checkpoint's action predictions.

**Table 4. Reproducible implementation checks and their claim boundaries.**

| Technical check | Observation | Supported interpretation |
|---|---|---|
| Android JVM tests | 20 passed; 0 failed, errored, or skipped | Tested scheduler, heuristic, normalization, tracking, classifier-label, report, and agent-utility behaviors remain internally consistent |
| Android debug build | Build completed successfully | The current mobile artifact and bundled model assets package together |
| Focused model-export tests | 3 passed | Expected FenceNet and YOLO ONNX interfaces can be loaded and exercised |
| FenceNet checkpoint-to-ONNX parity | Maximum absolute logit difference \(2.98 \times 10^{-7}\) on a synthetic input generated with PyTorch seed 0 | The tested export closely reproduces checkpoint computation |
| FenceNet deployed interface | Input \(1 \times 18 \times 28\); output \(1 \times 6\) | Android tensor construction matches the model contract |
| Repository-wide Python suite | 82 passed and 36 failed | The legacy Python tree is not a clean validation target; many failures reference absent modules or older interfaces |
| Coach-labeled benchmark and device trace | Not present in the current artifact | Accuracy, calibration, latency, energy, and thermal claims remain unmeasured |

The checkpoint contains model weights but no training manifest, split
identifiers, class distribution, camera conditions, or held-out metrics.
Consequently, we cannot reconstruct a defensible evaluation dataset from the
deployed artifact. The preliminary evaluation in Section 7 supplies usability
and design-diagnostic evidence only and is not reused as an accuracy benchmark.

### 6.2 Required Accuracy, Latency, and Calibration Measures

A complete technical evaluation should use an independently annotated video
corpus rather than the recordings used to tune rule thresholds. The corpus
should vary fencer experience, target side, body proportions, clothing, camera
distance and angle, illumination, movement speed, opponent presence, and
partial occlusion. It should include all six supported actions, non-action
intervals, correct technique, each implemented error type, and unsupported
movements. At least two qualified fencing coaches should annotate action
boundaries and feedback correctness, with disagreements retained and reported
rather than collapsed without explanation.

The evaluation should report:

1. pose availability, target-identity retention, reacquisition time, and
   landmark error under each recording condition;
2. per-class action precision, recall, F1, confusion matrices, and macro
   averages, with participant-independent test splits;
3. reliability diagrams or expected calibration error for action confidence,
   together with coverage-versus-error curves for the 0.60 abstention
   threshold;
4. per-rule precision, recall, false cues per minute, and coach agreement on
   whether each cue was appropriate and timely;
5. scheduler agreement with coach priority when several errors co-occur, plus
   the frequency of repeated, delayed, or expired cues;
6. end-to-end capture-to-display and capture-to-speech latency at the median
   and 95th percentile, effective update rate, dropped-frame rate, warm-up
   time, battery use, and thermal throttling across representative low-, mid-,
   and high-end Android devices; and
7. report grounding, measured as the proportion of generated statements
   supported by stored action or cue records.

These measures should be stratified by recording condition and accompanied by
confidence intervals. Component accuracy alone would not establish coaching
effectiveness, but it is necessary to determine whether subsequent user
responses concern the intended system behavior or failures in perception and
timing.

### 6.3 Failure Cases and Abstention Behavior

Code inspection and the formative study identify several cases that the future
benchmark must treat explicitly. One participant observed a defensive movement
being misclassified and described the resulting correction as delayed. This is
useful evidence of an end-to-end failure mode, but a single observation does
not estimate its frequency. Table 5 distinguishes the current handling from
the unresolved technical risk.

**Table 5. Current failure handling and unresolved technical risks.**

| Failure condition | Current behavior | Unresolved risk or required test |
|---|---|---|
| Low action confidence | Action output becomes `Idle` below 0.60 | Most form rules can still speak, so action abstention is not full feedback abstention |
| Low-visibility or missing landmarks | Unavailable joints are omitted; the tracker can extrapolate for five missing detections | Geometry may be computed from stale or incomplete evidence; pose-quality gating requires validation |
| Multiple people or crossing trajectories | Target selected by side and maintained by motion/proximity | No identity model prevents a switch to an opponent or bystander |
| MediaPipe/YOLO backend change | Both feed a nominally shared skeleton format | Different front-limb semantics can change rule outputs; backend-equivalence tests are required |
| Variable capture or decode rate | Cue time is derived from frame index at nominal 30 fps | Stored timecodes and temporal windows can drift from actual motion time |
| Simultaneous errors | Up to three visual cues and one spoken cue are prioritized with cooldowns | Priority scores and delay acceptability have not been validated with coaches |
| Unsupported movement or rule coverage gap | No rule may fire and the UI can display "Good Technique" | Absence of a detected error may be mistaken for verified correctness |
| Missing model or optional cloud failure | Model-unavailable state or deterministic report fallback | Live analyzer exceptions remain insufficiently visible to the learner |

The current technical claim is therefore deliberately narrow: the Android
prototype integrates on-device pose processing, a numerically faithful export
of the available FenceNet checkpoint, deterministic fencing heuristics, and
bounded multimodal feedback into an executable workflow. Whether the workflow
is accurate, calibrated, timely, and robust across fencers and environments
remains an empirical question. Retaining this section, rather than deleting it,
makes that evidence boundary explicit and turns the missing measurements into
a reproducible evaluation plan.

## 7. Preliminary Evaluation Design and Analysis

The fencer-facing four-participant sessions described in Section 3 also
provide the paper's preliminary evaluation evidence. No second participant
sample or independent validation study was conducted. Section 3 uses the
retained qualitative material formatively alongside the coach-derived playbook
to derive design requirements; Sections 7 and 8 organize the fencer-facing
corpus around the research questions and descriptive outcomes. The evaluation
therefore addresses feasibility, interpretation, and interaction breakdowns,
not skill acquisition or long-term effectiveness.

### 7.1 Research Questions

Following the questions introduced in Section 1.4, the study asked:

- **RQ1:** What differences emerge between the errors fencers identify through
  unaided video self-review and those they report after AI-assisted feedback?
- **RQ2:** Which properties of live feedback make a correction actionable
  without unnecessarily interrupting practice?
- **RQ3:** How do cue timing, action-recognition errors, and post-session
  evidence shape appropriate reliance on AI-generated fencing feedback?

### 7.2 Participants and Ethics

Participant and fencer-facing data coverage are summarized in Section 3.1.
Four participants contributed aggregate questionnaire ratings, while detailed
records were retained only for P1 and P2. Recruitment, participant
characteristics, compensation, and the applicable ethics, consent, retention,
and withdrawal procedures are absent from the supplied artifacts. As detailed
in Sections 3.1 and 10.1, the manuscript therefore makes no representativeness,
saturation, or complete ethical-governance claim.

### 7.3 Conditions and Baseline

As described in Section 3.2, each session followed four fixed phases: unaided
video self-review, real-time earphone feedback, a feedback-guided attempt, and
post-session video analysis. Self-review served as a current-practice
reference, but not as a controlled comparison condition. The order was not
counterbalanced, and P2's baseline included interviewer prompting; repetition,
task familiarity, and researcher guidance therefore remain alternative
explanations for later responses.

### 7.4 Procedure

The complete task sequence and retained corpus are reported in Section 3.2.
For evaluation, we treated verbalized baseline diagnoses, explanations of cue
meaning, reported correction attempts, reactions to incorrect detections, and
the 15 post-use ratings as complementary evidence. The visible protocol
deviations were retained in analysis: P2's prompted baseline, developer
explanations during P1's session, and P2's incomplete final post-attempt
response.

### 7.5 Measures

Table 6 maps each research question to the evidence available in the study
corpus. The questionnaire summary contains 15 item means, each based on four
participants. Item wording, response anchors, participant-level values, and
dispersion statistics were not included in the supplied file.

**Table 6. Research questions, constructs, and retained evidence.**

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

The qualitative procedure is reported in Section 3.3 and was not repeated as a
separate analysis. In Sections 8.2-8.5, we integrate those case findings with
the aggregate ratings through convergence, complementarity, and contradiction.
For example, the interviews explain why low-interruption ratings can coexist
with poor timing ratings, while P1's baseline self-diagnosis limits a simple
interpretation of the high advantage-over-self-review mean.

<!-- Insert user-study procedure figure here.
Figure 4. Fixed-order preliminary study procedure: unaided video self-review,
real-time earphone feedback, a feedback-guided attempt, post-session analysis,
questionnaire, and semi-structured interview. -->

## 8. Preliminary Evaluation Results

### 8.1 Evidence Coverage

This section reports no new participant pool beyond Section 3. All 15
questionnaire means use `N=4`; qualitative interpretation is limited to P1 and
P2, with the deviations summarized in Sections 7.3 and 7.4. Because
participant-level ratings and characteristics were unavailable, the results
cannot support individual trajectories, subgroup comparisons, or
distributional claims.

### 8.2 Reported Correction Attempts and Performance Boundary

Consistent with the formative cases in Section 3.4, P1 and P2 reported
attempting immediate changes to step size or center-of-mass position after
spoken cues. P1 could already diagnose major errors at baseline, whereas P2
did not independently state a correction before interviewer prompting. The
evaluation therefore records reported correction attempts, not a uniform gain
over self-review.

No blinded coach rating, coded kinematic outcome, or participant-level pre/post
performance measure was retained. We cannot determine whether the attempted
changes were technically correct, improved fencing performance, or persisted
beyond the session.

### 8.3 Experience, Trust, and Actionability

Across the 15 questionnaire items, the unweighted overall mean was 4.20. The
five learning-support items had the highest category mean (4.55), followed by
training value (4.45) and feedback quality (3.60). The relative contrast is
important: items concerning noticing and reviewing errors had higher means than
items concerning whether individual cues arrived at the right time and clearly
explained how to correct the movement. Without the response anchors, these
values support only descriptive comparisons within the supplied item set.

**Table 7. Aggregate post-use ratings from the four-participant sample.**

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

The highest item means were Error Awareness, Better than Self-Review, and
Post-Review Usefulness (all M=4.75). Timing Accuracy and Correction
Understandability were lowest (both M=3.25). The Low Interruption item also had
a comparatively high mean (M=4.25), matching the interviews: neither P1 nor P2
identified voice feedback itself as a major disruption. Their difficulty was
temporal reference, not merely the presence of audio.

### 8.4 Mixed-Methods Results by Research Question

**RQ1: Error awareness alongside self-review.** Error Awareness and Better than
Self-Review were among the highest item means (both `M=4.75`), but the case
evidence qualifies this pattern. P2 struggled to name a baseline problem,
whereas P1 already identified major posture and footwork issues. The retained
evidence therefore supports differing forms of added value, such as initial
diagnosis for one learner and added detail or confirmation for another, rather
than a general claim that AI reveals errors users cannot see.

**RQ2: Actionability without unnecessary interruption.** The item labeled
Actionable Correction had a comparatively high mean (`M=4.50`), and neither
interviewee identified earphone audio itself as a major disruption, consistent
with the Low Interruption mean (`M=4.25`). The interviews add the mechanism:
directional step-size and balance cues implied a concrete next action, while
unfamiliar arm or defensive-action issues needed richer post-session
explanation. Correction Understandability remained lower (`M=3.25`),
indicating that a cue can be brief and tolerable without fully explaining the
correction.

**RQ3: Timing, recognition errors, and appropriate reliance.** Timing accuracy
was among the lowest-rated items (`M=3.25`), matching both interviewees'
uncertainty about which movement a cue described. The Feedback Trust mean
(`M=3.75`) also concealed meaningful variation: participants accepted some
step and balance cues but questioned defensive-action and lunge-related
classifications. Post-session evidence added useful detail, yet could also make
a false classification appear more authoritative. Appropriate reliance
therefore depended on temporal provenance and contestability, not confidence
in the system as a whole.

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

Section 3.5 translates the formative corpus into DR1-DR5 and records the
current Android response. We do not repeat that traceability table here
because the evaluation is not independent evidence that those design responses
are effective. Sections 9 and 10 instead discuss the implications and the
measures needed to test them.

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

For RQ1, the findings suggest that AI feedback may add diagnostic structure
beyond self-review, but its value depends on what a learner can already see.
P1 independently identified conspicuous center-of-mass and step-width problems,
whereas P2 struggled to name a problem before interviewer prompting. After
system use, both discussed concrete movement categories, and Error Awareness,
Better than Self-Review, and Post-Review Usefulness were among the highest
aggregate item means. The mechanism is not simply that the system reveals
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
movement. This explains the contrast between the comparatively high means for
the items labeled Actionable Correction and Low Interruption and the lower
Correction Understandability mean. A cue can be brief enough to act on without
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
can indicate which aspects received relatively higher or lower means in this
sample, but they are not population estimates and cannot support
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
| The retained study records document live and post-session prototype feedback in a short supervised workflow, and the current Android repository implements both modes. | The evaluated prototype version is identical to the current Android build or technically reliable across fencers, devices, viewpoints, and practice conditions. |
| Error Awareness and Post-Review Usefulness had among the highest aggregate means for four participants, and two retained interviews illustrate how the feedback was interpreted. | The reported means generalize to a fencing population or demonstrate superiority to video self-review. |
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
mixed-methods study of an early prototype, four participants contributed
post-use ratings and two had detailed retained interviews. Error Awareness,
Post-Review Usefulness, and Better than Self-Review had the highest aggregate
means; the interviews indicated that brief step-size and balance cues could
guide reported next-attempt adjustments. At the same time, delayed cues and
action misclassification made otherwise understandable advice difficult to
connect to the movement that produced it. Together, these findings identify
temporal grounding, layered explanation, prioritization, and contestability as
requirements for interpretable AI movement feedback that supports rather than
replaces learner and coach judgment. Given the small, partially retained corpus
and absence of objective skill measures, this work contributes feasibility
evidence and design requirements rather than proof of durable fencing
improvement.

## Acknowledgments

## References

1. Filip Malawski and Bogdan Kwolek. 2018. Recognition of action dynamics in
   fencing using multimodal cues. *Image and Vision Computing* 75, 1-10.
   https://doi.org/10.1016/j.imavis.2018.04.005
2. Kevin Zhu, Alexander Wong, and John McPhee. 2022. FenceNet: Fine-Grained
   Footwork Recognition in Fencing. In *Proceedings of the IEEE/CVF Conference
   on Computer Vision and Pattern Recognition Workshops*, 3589-3598.
   https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Zhu_FenceNet_Fine-Grained_Footwork_Recognition_in_Fencing_CVPRW_2022_paper.html
3. Filip Malawski and Marek Krupa. 2023. Temporal Segmentation of Actions in
   Fencing Footwork Training. *Computer Science Research Notes*.
   https://doi.org/10.24132/CSRN.3301.28
4. Mingdong Zhang, Li Chen, Xiaoru Yuan, Renpei Huang, Shuang Liu, and Junhai
   Yong. 2019. Visualization of technical and tactical characteristics in
   fencing. *Journal of Visualization* 22, 109-124.
   https://doi.org/10.1007/s12650-018-0521-3
5. Takehiro Sawahata, Alessandro Moro, Sarthak Pathak, and Kazunori Umeda.
   2024. Instance Segmentation-Based Markerless Tracking of Fencing Sword
   Tips. In *2024 IEEE/SICE International Symposium on System Integration*.
   https://doi.org/10.1109/SII58957.2024.10417603
6. Atima Tharatipyakul, Kenny T. W. Choo, and Simon T. Perrault. 2020. Pose
   Estimation for Facilitating Movement Learning from Online Videos. In
   *Proceedings of the International Conference on Advanced Visual
   Interfaces*. https://doi.org/10.1145/3399715.3399835
7. Alessandra Semeraro and Laia Turmo Vidal. 2022. Visualizing Instructions for
   Physical Training: Exploring Visual Cues to Support Movement Learning from
   Instructional Videos. In *CHI Conference on Human Factors in Computing
   Systems*. https://doi.org/10.1145/3491102.3517735
8. Leonor Portugal da Fonseca, Francisco Nunes, and Paula Alexandra Silva.
   2024. Understanding Feedback in Rhythmic Gymnastics Training: An
   Ethnographic-Informed Study of a Competition Class. In *Proceedings of the
   CHI Conference on Human Factors in Computing Systems*.
   https://doi.org/10.1145/3613904.3642434
9. Jian-Jia Weng, Calvin Ku, Jo Chien Wang, Chih-Jen Cheng, Tica Lin, Yu-An
   Su, Tsung-Hsun Tsai, You-Yi Lin, Lun-Wei Ku, Hung-Kuo Chu, and Min-Chun Hu.
   2025. Bridging Coaching Knowledge and AI Feedback to Enhance Motor Learning
   in Basketball Shooting Mechanics Through a Knowledge-Based SOP Framework.
   In *Proceedings of the 2025 CHI Conference on Human Factors in Computing
   Systems*. https://doi.org/10.1145/3706598.3713324
10. Valentin-Adrian Niță and Petra Magyar. 2023. Improving Balance and Movement
    Control in Fencing Using IoT and Real-Time Sensorial Feedback. *Sensors*
    23, 24. https://doi.org/10.3390/s23249801
11. Lana Frančeska Dreimane and Zinta Zālīte-Supe. 2022. Instructional Design
    Map for Immersive Fencing Training in Virtual Reality. In *Human,
    Technologies and Quality of Education, 2022*.
    https://doi.org/10.22364/htqe.2022.13
12. Max Schemmer, Niklas Kuehl, Carina Benz, Andrea Bartos, and Gerhard
    Satzger. 2023. Appropriate Reliance on AI Advice: Conceptualization and the
    Effect of Explanations. In *Proceedings of the 28th International
    Conference on Intelligent User Interfaces*.
    https://doi.org/10.1145/3581641.3584066
13. Shiye Cao, Anqi Liu, and Chien-Ming Huang. 2024. Designing for Appropriate
    Reliance: The Roles of AI Uncertainty Presentation, Initial User Decision,
    and User Demographics in AI-Assisted Decision-Making. *Proceedings of the
    ACM on Human-Computer Interaction*.
    https://doi.org/10.1145/3637318
