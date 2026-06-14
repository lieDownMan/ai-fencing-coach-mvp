# What Makes Strong HCI System and Conclusion Sections?

Status: Research synthesis and paper-writing guide  
Last updated: 2026-06-14  
Target venue context: ACM CHI  
Project context: AI-assisted fencing coaching

## Executive Summary

A strong HCI **System** section explains a research argument embodied in an
interactive artifact. It creates a traceable chain:

> human evidence or theory -> design requirements -> interaction concept ->
> implemented mechanisms -> technical behavior and boundaries -> evaluation

It is not a feature list, a software manual, or a dump of model and framework
names. A reader should understand:

1. why this system is an appropriate response to the human problem;
2. what the central interaction idea is;
3. how a person encounters and uses that idea;
4. how the interface and technical pipeline produce the claimed behavior;
5. which design choices matter to the HCI contribution;
6. what was actually implemented and evaluated; and
7. where uncertainty, failure, safety, privacy, and human authority enter.

A strong **Conclusion** section closes the same argument in compact form:

> problem -> research response -> evidence -> bounded contribution

It does not introduce a new result, repeat the full abstract, promise an
unbuilt future system, or broaden the claim beyond the evidence. In the ten
reviewed papers, conclusions were generally one compact paragraph of roughly
90-194 words.

For the AI Fencing Coach paper, the central System contribution should not be
described as "an app containing pose estimation, FenceNet, heuristics, and an
LLM." The stronger account is an interaction design for turning uncertain,
fencing-specific movement observations into brief live cues and inspectable
post-practice evidence while preserving learner and coach authority.

## 1. Scope and Corpus

This guide is based on a close reading of the System, Design, Implementation,
Discussion, and Conclusion sections of ten recent CHI system papers.

Two adjacent project guides provide deeper treatment of neighboring sections:

- `GOOD_HCI_SYSTEM_DESIGN_IMPLEMENTATION_AND_EVALUATION.md` focuses on the
  full design-to-evaluation chain; and
- `GOOD_HCI_SYSTEM_DISCUSSION_AND_LIMITATIONS.md` focuses on interpretation,
  claim boundaries, and future work.

This document concentrates on how the System section constructs the artifact
argument and how the Conclusion closes that argument.

### 1.1 Award-status clarification

The two supplied papers, **MR.Drum** and **RoomDreaming**, are CHI full papers
and are included as required. However, the official SIGCHI program data does
not label either paper as a Best Paper recipient. The corpus therefore
contains:

- eight officially verified CHI Best Papers from 2023-2025; and
- the two supplied CHI papers as additional system-writing comparisons.

Official award records:

- [CHI 2023 Best Papers](https://programs.sigchi.org/chi/2023/awards/best-papers)
- [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers)
- [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers)

### 1.2 Papers reviewed

| Paper | Year and status | System-section strategy | Conclusion strategy |
| --- | --- | --- | --- |
| [CiteSee](https://doi.org/10.1145/3544548.3580847) | 2023 Best Paper | Moves from exploratory interviews to three design goals, explains citation-augmentation types, provides a usage scenario, maps features back to goals, and closes with implementation details. | Restates the tool, its design basis, the lab comparison, and the field-deployment value. |
| [DataParticles](https://doi.org/10.1145/3544548.3581472) | 2023 Best Paper | Derives two design decisions from formative and content analyses, explains the language-oriented pipeline and block interaction, then demonstrates the complete workflow. | Returns to the domain problem, names the two interaction mechanisms, summarizes expert evidence, and gives a restrained forward-looking implication. |
| [DynaVis](https://doi.org/10.1145/3613904.3642639) | 2024 Best Paper | Defines the dynamic-widget concept, presents a three-module architecture, explains preprocessing, LLM synthesis, programmatic validation, and UI implementation. | Uses three moves: interaction concept, resulting system behavior, and the principal comparative result. |
| [Time-Turner](https://doi.org/10.1145/3613904.3641985) | 2024 Best Paper | Converts cognitive and metacognitive consequences into numbered requirements and design elements, then explains the prototype through the learner flow. | Organizes closure around empirical, formative, and design contributions and ends with a bounded reinterpretation of multitasking. |
| [Piet](https://doi.org/10.1145/3613904.3642711) | 2024 Best Paper | Derives design goals from professional practice, gives a system overview, explains linked representations and interactions, reports implementation details, and ends with a walkthrough. | Reconnects the formative problem, tool response, expert evaluation, workflow fit, and broader design opportunity. |
| [AACessTalk](https://doi.org/10.1145/3706598.3713792) | 2025 Best Paper | Connects each rationale to formative evidence, explains a dyadic turn-taking flow, documents AI pipelines for both users, and reports implementation choices. | Restates the reciprocal-engagement goal, explains the paired AI support, reports two-week deployment findings, and ends with the human significance. |
| [Code Shaping](https://doi.org/10.1145/3706598.3713822) | 2025 Best Paper | Treats the system as an evolving research probe across three design-study stages; each iteration responds to observed ambiguity, interpretability, or context-switching problems. | Defines the interaction paradigm, summarizes what the iterative studies learned, identifies the resulting mechanisms, and states the transferable insight. |
| [Traversing Dual Realities](https://doi.org/10.1145/3706598.3713949) | 2025 Best Paper | Bounds the cross-reality context before describing three interaction techniques and their design dimensions, allowing later studies to test specific mechanisms. | Reconstructs the two-study progression and closes with lessons for making desktop and AR work together. |
| [MR.Drum](https://doi.org/10.1145/3706598.3714156) | 2025 supplied CHI paper; not listed as Best Paper | Turns learner pain points into interface implications, explains iterative UI refinement, then separates interface design from hardware and software implementation. | Connects the learning framework, MR realization, comparative evidence, domain-level implication, and open-source artifact. |
| [RoomDreaming](https://doi.org/10.1145/3613904.3642901) | 2024 supplied CHI paper; not listed as Best Paper | Starts from divergent and convergent design exploration, states system goals, explains novice and advanced interaction, details the generation pipeline, and reports latency. | Summarizes the system, design motivation, multi-study process, participants, and the bounded potential for exploration and communication. |

### 1.3 Length is not the quality criterion

Across the extracted papers:

- ordinary System or Design sections were approximately 1,200-4,100 words;
- the longer Code Shaping account integrated three system iterations and their
  studies rather than using a conventional standalone System section; and
- Conclusions were approximately 90-194 words.

These ranges describe this corpus, not venue requirements. The necessary length
depends on how much interaction logic, technical novelty, and implementation
detail the contribution requires.

## 2. Core Definition of a Good HCI System Section

A good HCI System section is a **selective, evidence-backed explanation of how
an implemented artifact operationalizes the paper's research contribution**.

The word "selective" matters. A paper does not need to document every screen,
class, dependency, or engineering decision. It should explain the parts needed
to:

- understand the interaction concept;
- audit the connection between evidence and design;
- reproduce or critically assess consequential behavior;
- interpret the evaluation;
- distinguish the research artifact from prior systems; and
- identify important limits or failure paths.

The section should allow a reviewer to answer:

> If the proposed interaction mechanism produced the reported result, what
> exactly did the system do, why was it designed that way, and what alternative
> explanation or failure should I consider?

## 3. Characteristics of a Strong System Section

### 3.1 Begin with the design logic, not the technology stack

Strong papers establish why the system has its particular form before listing
technical components.

- CiteSee maps interview findings to three goals concerning unknown, familiar,
  and previously encountered citations.
- Time-Turner derives design elements from cognitive and metacognitive
  consequences of multitasking.
- Piet derives linked, multi-level color representations from professional
  workflow breakdowns.
- AACessTalk derives reciprocal support for both parent and child from a
  formative finding that parent-led interaction can reduce child agency.
- MR.Drum derives first-person and progressive demonstration from instructor
  practice and learner difficulty.

A useful opening pattern is:

> Based on **[formative finding, theory, or prior evidence]**, we derived
> **[requirements]**. Together, these requirements motivate **[system concept]**,
> which supports **[user and activity]** by **[central interaction mechanism]**.

Do not begin with "The frontend uses React and the backend uses Python." That
explains construction before purpose.

### 3.2 State the system concept and scope in one clear paragraph

Before entering details, define:

- the intended user;
- the activity and setting;
- the input;
- the system's main transformation or interaction;
- the output;
- the role the system assumes; and
- what falls outside its scope.

Examples in the corpus include personalized citation augmentation, dynamic
widgets synthesized from language, block-based animated-story authoring,
bichronous recovery from missed lecture content, and transitions between
desktop and AR.

For an AI system, the concept should be an interaction claim rather than a list
of models. "An LLM-powered interface" is not yet a concept. "Persistent widgets
convert a one-time natural-language command into an inspectable and repeatedly
adjustable control" is.

### 3.3 Make design requirements traceable

Requirements should have a source and an observable response.

| Element | Question |
| --- | --- |
| Evidence | What human observation, theory, risk, or prior result motivates the requirement? |
| Requirement | What must the interaction support, prevent, reveal, or preserve? |
| System response | Which implemented mechanism addresses it? |
| Evaluation question | What evidence would show whether the response works? |

A requirement such as "The system should be easy to use" is too broad. A
stronger requirement names the consequence and mechanism:

> Reduce the need to shift attention between a moving body and a detailed
> explanation by delivering one brief live cue and retaining richer evidence
> for post-practice review.

Traceability does not prove effectiveness. It proves that the artifact has a
defensible design rationale that can later be tested.

### 3.4 Explain the user flow before the internal pipeline

Several papers use a scenario or walkthrough because it reveals when system
states and mechanisms matter.

A useful flow should show:

1. how a person begins;
2. what information or artifact they provide;
3. what the system makes visible;
4. what action the person takes;
5. how the system responds;
6. how the person revises, accepts, rejects, or continues; and
7. what remains available after the interaction.

The flow should include transitions, waiting, interruption, recovery, and
inspection where those affect the experience. In AI systems, generation time,
misinterpretation, correction, and fallback behavior are part of the
interaction, not incidental implementation details.

### 3.5 Organize around interaction mechanisms, not screens or source files

The strongest subsection headings explain what each mechanism accomplishes:

- discovering and contextualizing citations;
- maintaining correspondence between narrative and visualization;
- synthesizing a persistent control from a language command;
- supporting cognitive recovery after multitasking;
- linking colors across element, scene, and video levels;
- balancing conversational support for two participants; or
- moving an object across realities.

Weak organization follows menus, tabs, classes, or implementation order. It
forces the reader to infer the contribution from interface inventory.

For each key mechanism, explain:

1. the problem or requirement;
2. the interaction;
3. the system behavior;
4. the design tradeoff; and
5. the expected consequence that the evaluation will examine.

### 3.6 Use figures to establish a shared mental model

Most system papers need at least two different visual explanations:

1. **Interaction figure:** the user workflow, interface states, or an annotated
   screenshot.
2. **Architecture figure:** data and control flow among consequential system
   components.

Additional useful figures include:

- a before-and-after design iteration;
- a timing diagram;
- an example of input, intermediate representation, and output;
- a failure and recovery sequence; or
- a mapping from design requirements to interface mechanisms.

The prose should explain why each transition or component matters. A diagram
with boxes and arrows is not self-explanatory.

### 3.7 Explain technical detail at the contribution boundary

Include a technical detail when it changes at least one of the following:

- what the user can do;
- what the system can infer or generate;
- latency or responsiveness;
- reliability or failure behavior;
- privacy or data exposure;
- safety;
- reproducibility;
- the fairness of the evaluation; or
- the novelty of the interaction mechanism.

Useful details in the corpus include:

- DynaVis's data summarization, LLM synthesis, schema checks, retries, and
  program analysis;
- AACessTalk's dialogue representations and separate parent and child
  generation pipelines;
- RoomDreaming's segmentation, depth estimation, generation controls, and
  measured image latency;
- MR.Drum's MIDI input, electronic drum set, MR headset, and timing behavior;
  and
- Piet's linked color representations and real-time rendering.

Version numbers and framework names alone rarely establish rigor. State what a
component does in the research system and why that choice matters.

### 3.8 Describe AI as a fallible component

AI-mediated System sections should document:

- model or service and version, where known;
- the information supplied to the model;
- prompt, representation, or grounding strategy;
- output format;
- validation and post-processing;
- latency;
- retry, fallback, or abstention behavior;
- how uncertainty or error becomes visible;
- what users can correct, reject, or override;
- privacy implications of transmitted data; and
- what deterministic behavior remains when generation fails.

DynaVis is particularly instructive because it does not stop at "we use an
LLM." It explains compact data grounding, constrained output, JSON and schema
validation, compilation checks, repair attempts, and program analysis.

For high-stakes or instructional systems, model output should not silently
become authoritative feedback.

### 3.9 Distinguish design, implementation, and evaluation

These have different jobs:

- **Design:** why the interaction has its form.
- **Implementation:** how the tested artifact realizes that interaction.
- **Evaluation:** whether and how the artifact produced the investigated human
  consequences.

A System section may mention an expected benefit, but it should not present
that benefit as established before the study. Write "to support," "intended
to," or "designed to" until evidence justifies stronger language.

### 3.10 Report what exists, not the idealized product

Clearly distinguish:

- implemented and used in the study;
- implemented but not evaluated;
- simulated or Wizard-of-Oz;
- manually prepared;
- available only in a subset of conditions;
- optional or externally hosted;
- planned future work; and
- conceptual rather than functional.

This distinction is essential when a polished interface suggests capabilities
that the prototype did not actually provide.

### 3.11 Let the System section predict the evaluation

After reading the System section, a reviewer should be able to anticipate:

- which mechanism is compared or observed;
- which tasks expose it;
- which baseline is credible;
- which errors or failures matter;
- which behavioral and experiential measures are needed; and
- which claims the study cannot establish.

Examples:

- DynaVis's persistent widgets motivate comparison with a language-only
  interface and measures of repeated editing and confidence.
- Time-Turner's recovery mechanisms motivate learning-outcome evaluation under
  multitasking.
- MR.Drum's progression and first-person demonstrations motivate comparison
  with instructional video.
- AACessTalk's paired guidance motivates analysis of both parent and child
  participation rather than parent satisfaction alone.

If the later study could be conducted without understanding the System
section, the design account may be descriptive rather than argumentative.

## 4. Recommended System and Implementation Structure

The following is a flexible template for an artifact-centered CHI paper.

### 4.1 Design basis

Briefly restate or reference the evidence that drives the system.

Include:

- two to five numbered design requirements;
- the source of each requirement;
- consequential tensions among requirements; and
- the system concept that responds to them.

Avoid repeating the full formative findings.

### 4.2 System overview and scope

In one subsection:

- define the intended user and setting;
- name the central interaction mechanism;
- summarize inputs, processing, outputs, and user control;
- state the prototype boundary; and
- introduce the main interaction figure.

### 4.3 Interaction flow

Describe one realistic end-to-end use sequence. Include:

- setup and configuration;
- normal interaction;
- system feedback;
- revision or response;
- failure or uncertainty;
- post-use review; and
- exit, persistence, or sharing.

### 4.4 Key interaction mechanisms

Use one subsection per research-relevant mechanism. For each:

```text
Requirement or problem:
Why does this mechanism need to exist?

Interaction:
What does the user see and do?

System behavior:
What state, inference, rule, or transformation produces the response?

Tradeoff:
What does this design prioritize or sacrifice?

Evaluation link:
What later observation or measure examines this mechanism?
```

### 4.5 Safety, uncertainty, privacy, and agency

This deserves a dedicated subsection when the system gives advice, uses AI,
handles sensitive data, or affects vulnerable users.

Report:

- failure cases;
- confidence and abstention;
- correction and override;
- user and expert authority;
- data storage and transmission;
- physical or psychological safety;
- normative assumptions; and
- fallback behavior.

### 4.6 Implementation and architecture

Start with an architecture figure, then explain only consequential components:

- sensing or input capture;
- preprocessing and representation;
- inference or generation;
- deterministic rules;
- orchestration and state;
- feedback presentation;
- persistence;
- external services; and
- performance constraints.

End by stating the tested hardware, software, model, and network environment.

## 5. Useful System Figures and Tables

### Figure A: Design traceability

```text
formative evidence -> design requirement -> system mechanism ->
evaluation question
```

### Figure B: User interaction flow

Show interface states and user actions rather than only backend components.

### Figure C: System architecture

Show:

- input and output types;
- component boundaries;
- local versus remote processing;
- feedback loops;
- stored data;
- confidence or error paths; and
- optional components.

### Table A: Mechanism specification

| Mechanism | Input | Behavior | Output | User control | Failure or fallback |
| --- | --- | --- | --- | --- | --- |

### Table B: Design-to-evaluation traceability

| Requirement | Implemented response | Expected consequence | Evaluation evidence |
| --- | --- | --- | --- |

## 6. Common System-Section Failure Modes

### Feature tour

The paper lists screens and buttons without explaining the interaction concept
or design rationale.

### Technology-led description

The section begins with models, libraries, and services, leaving the human
problem disconnected from the artifact.

### Untraceable requirements

Design goals sound reasonable but cannot be connected to formative evidence,
theory, prior work, or an explicit hypothesis.

### Architecture without behavior

A pipeline diagram names components but does not explain what information
moves through them, what triggers decisions, or how errors affect users.

### Vague AI

"The system uses AI" replaces documentation of prompts, grounding, output
constraints, validation, latency, failure, and user recourse.

### Evaluation leakage

The System section claims that a design is effective, intuitive, trustworthy,
or accurate before presenting evidence.

### Product inflation

The prose describes the intended product rather than the prototype used by
participants.

### Irrelevant engineering detail

Long lists of frameworks, endpoints, and low-level classes consume space
without affecting reproducibility or the HCI claim.

### Missing temporal behavior

The paper explains what appears but not when it appears, how long it takes, or
which prior action it refers to.

### Hidden failure path

The normal workflow is documented while low confidence, generation failure,
tracking loss, invalid output, or user disagreement is omitted.

### Orphan mechanism

A feature receives substantial explanation but is not connected to a design
requirement, research question, or evaluation measure.

## 7. Core Definition of a Good HCI Conclusion

A good HCI Conclusion is a **short, evidence-calibrated reconstruction of the
paper's completed argument and its durable contribution**.

It should answer:

1. What human or interaction problem did the work address?
2. What did the paper contribute in response?
3. What evidence was produced?
4. What did that evidence show?
5. What bounded HCI knowledge should remain with the reader?

The conclusion is not the place to rescue an unclear contribution. It should
make an already established argument memorable.

## 8. Characteristics of a Strong Conclusion

### 8.1 Name the contribution, not only the artifact

Weak:

> We built an Android application for fencing.

Stronger:

> We developed and investigated a layered feedback interaction that converts
> camera-based fencing observations into prioritized live cues and inspectable
> post-practice evidence.

The artifact is included, but the contribution is the interaction knowledge
embodied and examined through it.

### 8.2 Summarize the evidence architecture

Name the study type, participants or deployment duration when important, and
the main comparison or empirical basis.

The reviewed conclusions commonly mention:

- formative evidence;
- the system or interaction technique;
- lab, expert, comparative, or deployment study;
- the principal finding; and
- a broader but bounded implication.

Do not restate every measure or theme.

### 8.3 Select one or two principal results

A conclusion should not reproduce the Results section. Select findings that
directly support the central contribution.

For a preliminary study, appropriate closure may concern:

- feasibility;
- interpretation;
- workflow fit;
- breakdowns;
- design requirements; or
- hypotheses for a stronger study.

It should not convert participant preference into learning effectiveness.

### 8.4 Calibrate every verb

| Evidence | Defensible wording |
| --- | --- |
| Short usability or feasibility study | indicated, revealed, identified, suggested, exposed |
| Controlled comparison with suitable measures | improved, reduced, increased, outperformed, within the tested setting |
| Qualitative or deployment evidence | participants described, appropriated, valued, resisted, or experienced |
| Preliminary or underpowered evidence | provides initial evidence, motivates, generates a hypothesis |
| No direct outcome measure | was designed to support, rather than improved |

Avoid "proves," "solves," "ensures," or "demonstrates effectiveness" unless the
design and evidence genuinely support those claims.

### 8.5 End with the transferable insight

The last sentence should state what the work changes for HCI design, research,
or theory.

Examples of transferable forms:

- persistent controls can complement one-shot language commands;
- paired support can rebalance agency in mediated communication;
- domain learning progression can organize immersive instruction;
- transitions should be treated as part of a cross-reality workflow; or
- AI feedback requires temporal grounding and contestability, not only correct
  labels.

The final sentence should remain inside the evidence boundary.

### 8.6 Add limitations only when needed to prevent overclaiming

Detailed limitations belong in the Discussion or Limitations section. A short
scope clause is useful in the Conclusion when readers might otherwise infer a
strong causal or generalizable claim.

Example:

> These preliminary findings identify interaction requirements rather than
> establish durable skill improvement.

### 8.7 Do not introduce new material

A Conclusion should not contain:

- a new study result;
- a new design rationale;
- a new citation-dependent argument;
- an unreported feature;
- a new limitation that changes interpretation;
- a new research question; or
- a broad social claim unsupported elsewhere.

## 9. Recommended Conclusion Pattern

For most artifact-centered HCI papers, use one paragraph of approximately
100-180 words.

```text
Problem and response:
We addressed [specific human or interaction problem] by developing/studying
[artifact, interaction technique, framework, or method].

Contribution:
The system operationalizes [central design idea] through [one or two defining
mechanisms].

Evidence:
In [study type and scope], [main result], while [important breakdown or
boundary] revealed [design consequence].

Transferable conclusion:
These findings show/suggest/identify [bounded HCI insight].

Optional scope:
Given [important evidence limit], the work supports [appropriate claim] rather
than [overclaim].
```

The order can vary, but all four functions should be present.

## 10. Common Conclusion Failure Modes

- **Abstract repetition:** repeating the abstract with minor word changes.
- **Contribution laundry list:** restating every bullet from the Introduction.
- **New-result surprise:** adding evidence that did not appear in Results.
- **Future-work ending:** closing on what was not done instead of what was
  learned.
- **Artifact-only claim:** treating implementation as the entire contribution.
- **Unbounded generalization:** moving from a small study to all users,
  settings, or domains.
- **Preference-to-effectiveness leap:** treating liking or willingness to use
  as improved performance or learning.
- **Technology triumphalism:** claiming that AI transforms a domain without
  evidence about practice, institutions, risks, or sustained use.
- **Empty significance:** ending with "This has important implications" without
  naming the implication.
- **Defensive limitation dump:** allowing caveats to erase the contribution
  rather than bound it.

## 11. Review Rubric

Score each dimension as `0 = absent`, `1 = partial`, or `2 = strong`.

### 11.1 System section

| Dimension | Review question |
| --- | --- |
| Design basis | Are consequential design choices grounded in evidence, theory, prior work, or an explicit hypothesis? |
| System concept | Can the central interaction idea be stated independently of the technology stack? |
| Scope | Are intended users, setting, role, inputs, outputs, and exclusions clear? |
| User flow | Can a reader follow normal interaction, revision, recovery, and post-use review? |
| Mechanisms | Are key mechanisms explained through purpose, behavior, tradeoff, and expected consequence? |
| Traceability | Can each major mechanism be linked to a requirement and later evaluation evidence? |
| Architecture | Does the architecture explain consequential data and control flow? |
| Technical detail | Are implementation choices reported where they affect behavior, reproducibility, latency, privacy, or failure? |
| AI transparency | Are grounding, generation, validation, uncertainty, correction, and fallback documented? |
| Agency and safety | Can users inspect, reject, pause, override, or seek expert judgment where appropriate? |
| Prototype honesty | Is implemented and evaluated behavior distinguished from planned or conceptual capability? |
| Visual explanation | Do figures and tables make the interaction and architecture auditable? |

Interpretation:

- **20-24:** strong, reviewer-auditable System account;
- **14-19:** sound structure with important gaps;
- **8-13:** feature description more than research argument;
- **0-7:** insufficient to interpret or evaluate the artifact.

### 11.2 Conclusion

| Dimension | Review question |
| --- | --- |
| Problem | Is the addressed human or interaction problem clear? |
| Response | Is the artifact or research response named precisely? |
| Contribution | Does the conclusion state knowledge beyond "we built a system"? |
| Evidence | Is the study basis and principal result summarized? |
| Calibration | Are claims no broader than the method and data permit? |
| Transfer | Is the final HCI insight specific and useful? |
| Closure | Does the section close the paper without introducing new material? |
| Economy | Is every sentence necessary? |

Interpretation:

- **14-16:** strong and submission-ready;
- **10-13:** credible but could be sharper or better bounded;
- **6-9:** mostly summary with weak contribution closure;
- **0-5:** missing, inflated, or disconnected from the evidence.

## 12. Application to the AI Fencing Coach Paper

The current manuscript already has an appropriate high-level split:

```text
4. System Design
5. Implementation
6. Technical Evaluation
7-8. User Study and Results
9. Discussion
10. Limitations and Future Work
11. Conclusion
```

The System and Implementation sections should preserve that division.

### 12.1 Recommended Section 4: System Design

#### 4.1 Design goals

Reference the five requirements already established in Section 3.5:

- reduce diagnostic burden without assuming zero self-awareness;
- keep live feedback brief and prioritized;
- ground feedback in a movement episode;
- make uncertainty inspectable and contestable; and
- preserve learner and coach authority.

Do not simply repeat the traceability table. Explain the central system concept:

> AI Fencing Coach uses layered feedback to divide work across time: one
> prioritized cue supports the next movement, while timestamped post-practice
> evidence supports interpretation, questioning, and reflection.

Also state that this is a design hypothesis. The current study does not prove
that the implemented mechanisms satisfy all five requirements.

#### 4.2 Interaction flow

Describe the end-to-end Android experience:

1. The learner chooses training mode, pose backend, target side, voice, and
   feedback preferences.
2. The camera captures a side-view practice session.
3. The system acquires and tracks the selected fencer and ignores idle frames.
4. After enough active movement data is available, FenceNet predicts a fencing
   action and exposes a confidence value.
5. Action-conditioned heuristics inspect visible movement features.
6. The scheduler ranks active issues, speaks at most one prioritized cue, and
   retains a small visual set.
7. The learner may pause, disable voice, focus or mute categories, or end the
   session.
8. Post-practice review presents session metrics, action counts, repeated
   issues, and a recent cue timeline.
9. A deterministic playbook produces feedback when optional language-model
   generation is disabled or unavailable.

Include at least one failure path: tracking loss, insufficient active frames,
low or incorrect action confidence, delayed feedback, or disagreement with a
cue.

#### 4.3 Feedback timing and prioritization

Explain:

- why simultaneous corrections can overload movement practice;
- how issue priority, persistence, and cooldown affect selection;
- why speech and on-screen feedback have different capacities;
- how the cue history supports later inspection;
- what timestamp or frame reference is stored;
- the current end-to-end latency;
- when a cue is suppressed or repeated; and
- the unresolved problem of binding a live cue to a specific repetition.

This subsection should directly motivate technical latency measures and user
measures of referent identification, interruption, and next-attempt action.

#### 4.4 Safety, uncertainty, and coach authority

Report current behavior honestly:

- action confidence is visible;
- users can pause, disable voice, and configure feedback categories;
- optional summaries are constrained by detected counts and playbook content;
- rule-based feedback remains available without an LLM;
- the system examines only a bounded set of visible movement features;
- cue-level confidence, user correction, dismissal, and reliable abstention are
  not yet complete; and
- no coach-validation study currently establishes technical correctness.

Position the system as a practice and reflection aid, not a coach replacement.

### 12.2 Recommended Section 5: Implementation

#### 5.1 System architecture

Use an architecture figure based on the current Android flow:

```text
CameraX frame
-> MediaPipe or YOLO pose backend
-> skeleton mapping
-> target tracking and short-gap handling
-> activity gatekeeper
-> spatial normalization
-> FenceNet ONNX action inference
-> action-conditioned biomechanical heuristics
-> feedback scheduler
-> visual overlay and Android TextToSpeech
-> post-practice report and local session persistence
-> deterministic playbook or optional LLM summary
```

For each arrow, name the data type or state transition. Mark which processing
is on-device and which optional step may use a remote service.

#### 5.2 Pose and action analysis

Report:

- supported pose backends and why two are available;
- mapping from detected landmarks to the skeleton expected by the pipeline;
- target selection and short-gap behavior;
- conditions under which a frame is treated as active;
- normalization and temporal-window construction;
- FenceNet input shape, action classes, confidence calculation, and model
  version;
- cadence, warm-up, and latency; and
- known failure conditions such as occlusion, framing, opponent confusion, and
  out-of-distribution movement.

#### 5.3 Biomechanical heuristics

Use a table with:

| Error | Applicable action or mode | Required joints | Computation | Threshold | Output | Known failure |
| --- | --- | --- | --- | --- | --- | --- |

Explain why the rule is fencing-relevant and where the threshold came from.
Separate coach-informed, literature-informed, empirically tuned, and provisional
thresholds. Do not imply biomechanical validity merely because a formula is
implemented.

#### 5.4 Visual, spoken, and post-session feedback

Explain:

- how active errors become ranked feedback candidates;
- how one spoken cue is selected;
- the number and ordering of visual issues;
- cooldown and repeat behavior;
- cue wording and localization;
- the distinction between live correction and reflective explanation;
- local cue and session persistence;
- playbook-based summaries;
- optional LLM prompting and fallback; and
- what evidence a generated summary is allowed to claim.

### 12.3 Required figures and tables

The manuscript should include:

1. **Design traceability figure:** formative evidence -> DR1-DR5 -> Android
   mechanisms -> unresolved evaluation questions.
2. **Interaction figure:** setup, live coaching, and post-practice review.
3. **Architecture figure:** on-device inference and feedback pipeline,
   including optional remote generation.
4. **Feedback-rule table:** signals, formulas, thresholds, applicable actions,
   confidence or abstention, timing, and intended correction.
5. **Failure table:** tracking loss, missing joints, action
   misclassification, delayed cue, false heuristic, LLM failure, and user
   recourse.

### 12.4 Claims the current System section should avoid

Do not claim that the current artifact:

- improves fencing skill or retention;
- produces technically correct coaching;
- reliably identifies every relevant movement error;
- has solved temporal grounding;
- provides calibrated cue-level uncertainty;
- is equivalent or superior to a coach;
- is safe for unsupervised use in all drills; or
- generalizes across camera positions, bodies, ability levels, and fencing
  styles without supporting evaluation.

## 13. Paper-Ready Conclusion Draft for AI Fencing Coach

The following draft is calibrated to the evidence currently described in
`docs/paper/paper.md`:

> We presented AI Fencing Coach, an Android prototype that combines on-device
> pose and fencing-action analysis with prioritized visual, spoken, and
> post-practice feedback for solo training. A preliminary mixed-method study
> with four participants, including detailed interviews with two, indicated
> that participants valued the system most for making errors visible and
> supporting later review. Short cues about step size and body balance could
> support reported next-attempt adjustments, while delayed feedback and action
> misclassification made some advice difficult to connect to the movement that
> produced it. These findings suggest that interpretable AI movement feedback
> depends not only on recognition accuracy, but also on temporal grounding,
> layered explanation, prioritization, and mechanisms for questioning uncertain
> analysis. Given the small and partially retained study corpus, this work
> establishes feasibility observations and design requirements rather than
> evidence of durable fencing-skill improvement.

Before submission, revise this paragraph to match the final System
implementation, technical evaluation, participant records, and completed
Discussion.

## 14. Final Reviewer-Facing Checklist

### System and implementation

- Can the system's central interaction idea be stated without naming a model or
  framework?
- Is every major mechanism linked to evidence, theory, or an explicit design
  hypothesis?
- Are the intended user, setting, role, input, output, and scope clear?
- Does the paper show a realistic end-to-end user flow?
- Are design choices explained through tradeoffs rather than preference?
- Does the architecture show meaningful data, control, and failure paths?
- Are implementation details sufficient to understand behavior and reproduce
  consequential choices?
- Are latency and temporal behavior reported?
- Are AI input, grounding, output, validation, and fallback documented?
- Can users inspect, reject, pause, correct, or override the system where
  appropriate?
- Are privacy, safety, and expert authority addressed?
- Is implemented and evaluated behavior separated from planned capability?
- Does the System section clearly motivate the technical and user evaluation?

### Conclusion

- Does the conclusion name the human problem and research response?
- Does it state knowledge beyond the existence of the artifact?
- Does it summarize the evidence and only the principal findings?
- Are causal, effectiveness, and generalization claims calibrated?
- Does the final sentence state a specific HCI insight?
- Does the paragraph avoid new facts, citations, features, and arguments?
- Is it meaningfully different from the abstract?
- Can any sentence be removed without losing an essential function?

## 15. Bottom Line

A strong HCI System section makes the artifact intellectually and technically
auditable. It explains why the interaction has its form, how the implemented
mechanisms realize that form, and where human control and system failure enter.

A strong Conclusion then compresses the completed research argument into its
most durable form: the problem addressed, the response contributed, the
evidence produced, and the bounded knowledge that should travel beyond the
prototype.

## Sources

- Official SIGCHI Best Paper records for
  [CHI 2023](https://programs.sigchi.org/chi/2023/awards/best-papers),
  [CHI 2024](https://programs.sigchi.org/chi/2024/awards/best-papers), and
  [CHI 2025](https://programs.sigchi.org/chi/2025/awards/best-papers)
- The ten DOI-linked papers in Section 1.2
- Author or repository full-text copies used for close reading:
  [CiteSee](https://arxiv.org/pdf/2302.07302),
  [DataParticles](https://creativity.ucsd.edu/papers/dataparticles.pdf),
  [DynaVis](https://arxiv.org/pdf/2401.10880),
  [Time-Turner](https://www.cs.ubc.ca/labs/socius/files/papers/chi2024-timeturner.pdf),
  [Piet](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/Piet.pdf),
  [AACessTalk](https://arxiv.org/pdf/2409.09641),
  [Code Shaping](https://arxiv.org/pdf/2502.03719), and
  [Traversing Dual Realities](https://arxiv.org/pdf/2504.00371)
- User-provided PDFs: `MR.Drum.pdf` and `RoomDreaming.pdf`
