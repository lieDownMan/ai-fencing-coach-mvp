# What Makes Good HCI System Design, Implementation, and Evaluation?

Status: Research synthesis and paper-writing guide  
Last updated: 2026-06-14  
Target venue context: ACM CHI  
Project context: AI-assisted fencing coaching  
Scope: Ten recent CHI system papers, including the two supplied PDFs

## Executive Definition

A strong HCI system paper presents one coherent argument:

> human problem -> evidence or theory -> design goals -> interaction mechanism -> implementation -> evaluation -> bounded contribution

The three system sections have different jobs:

- **System Design** explains why the interaction works the way it does.
- **Implementation** explains how the claimed interaction was actually realized, including the representations, models, parameters, data flow, latency, safeguards, and limitations that can affect use.
- **Evaluation** tests the paper's claims at the correct layer, such as component validity, end-to-end behavior, interaction quality, human outcomes, or sustained use.

The central quality test is traceability. A reviewer should be able to follow every major design feature backward to evidence and forward to an evaluation question.

A polished feature list is not enough. A long technology stack is not enough. A usability questionnaire is not enough. The system becomes an HCI contribution when the paper shows:

1. why the chosen interaction is necessary;
2. which mechanism is new or consequential;
3. how the implementation preserves that mechanism;
4. where the implementation can fail;
5. which evidence supports each claim; and
6. where the evidence stops.

## 1. Corpus and Award Verification

This is a purposive review of system-oriented papers, not a systematic review. "Recent" was operationalized as CHI 2023-2025. The corpus was selected to cover formative-to-system pipelines, AI-mediated interaction, creativity support, educational systems, visualization tools, and controlled interaction experiments.

Award status was rechecked on June 14, 2026 using the official SIGCHI conference program records:

- [CHI 2023 Best Papers](https://programs.sigchi.org/chi/2023/awards/best-papers)
- [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers)
- [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers)
- [CHI 2023 program cache](https://files.sigchi.org/conference/cache/10088/250/program)
- [CHI 2024 program cache](https://files.sigchi.org/conference/cache/10107/111/program)
- [CHI 2025 program cache](https://files.sigchi.org/conference/cache/10127/224/program)

Important correction: the official program records do **not** label `MR.Drum` or `RoomDreaming` as Best Paper Award recipients. They are included because the user explicitly required both supplied papers. The corpus therefore contains eight verified Best Papers and two required recent CHI papers.

| Paper | Year and status | System argument reviewed | Main lesson |
| --- | --- | --- | --- |
| [Code Shaping](https://doi.org/10.1145/3706598.3713822) | 2025, Best Paper | Three design-study stages progressively reconcile sketches, AI interpretation, and code editing | Treat a prototype as a research instrument and show how evidence changes the interaction |
| [PAIGE](https://doi.org/10.1145/3706598.3713460) | 2025, Best Paper | Configuration, outline, fact-checking, transcript, and speech-generation pipeline tested in a 3x3 study | Describe enough of an AI pipeline to connect generation choices to learning outcomes and failures |
| [Lost in Magnitudes](https://doi.org/10.1145/3706598.3713487) | 2025, Best Paper | Constraint-based design-space generation, expert inspection, then controlled comparison | Use staged evaluation to reduce a design space before testing the strongest candidates |
| [DynaVis](https://doi.org/10.1145/3613904.3642639) | 2024, Best Paper | Modular generated widgets, declarative chart state, LLM synthesis, program analysis, and user comparison | Make architecture choices visibly support the interaction claim |
| [Piet](https://doi.org/10.1145/3613904.3642711) | 2024, Best Paper | Expert-derived goals become synchronized multi-level color views and group editing | Explain the representation and interaction rules, not only the interface panels |
| [Constrained Highlighting](https://doi.org/10.1145/3613904.3642314) | 2024, Best Paper | A theory-derived 150-word interface limit tested through pilots and delayed comprehension | A simple system can make a strong contribution when the mechanism and evaluation are precise |
| [Debate Chatbots](https://doi.org/10.1145/3613904.3642513) | 2024, Best Paper | Social identity and rhetorical style are implemented as controlled chatbot factors | Expose exactly how experimental constructs become system behavior |
| [DataParticles](https://doi.org/10.1145/3544548.3581472) | 2023, Best Paper | Language-oriented and block-based editing realized through parsing, visual state, and transition rules | Give enough algorithmic detail to audit the interaction mechanism |
| [MR.Drum](https://doi.org/10.1145/3706598.3714156) | 2025, required CHI paper, not listed as Best Paper | Formative studies produce a micro-progression framework, MR interface, MIDI implementation, and comparative study | Maintain visible traceability from domain practice to feature to outcome |
| [RoomDreaming](https://doi.org/10.1145/3613904.3642901) | 2024, required CHI paper, not listed as Best Paper | Formative work, generative-AI architecture, quality assessment, comparative studies, redesign, and co-design | Evaluate different uncertainties in separate studies instead of asking one study to prove everything |

## 2. What the Ten Papers Show

### 2.1 Code Shaping: Let Evidence Change the System

Code Shaping does not introduce one finished interface and then ask whether participants like it. It uses three stages:

1. observe how programmers naturally sketch edits;
2. study interpretation errors and repair strategies; and
3. redesign the interaction around feedforward interpretation, code references, and cross-layer gestures.

The final system includes sketch recognition, AI interpretation, abstract-syntax-tree analysis, affected-code highlighting, staged diffs, and accept or reject gestures. These details matter because the contribution concerns how programmers coordinate sketches, code, and uncertain AI interpretation.

The evaluation reports error categories, repair strategies, action flows, latency, accidental gesture activation, and participants' adaptations to model behavior. It does not hide that users changed how they sketched to accommodate the system.

**Lesson:** A good system section records design evolution, and a good evaluation examines the breakdowns that generated that evolution.

### 2.2 PAIGE: Make the AI Pipeline Auditable

PAIGE explains the podcast pipeline as:

> textbook and profile context -> configuration -> outline -> transcript -> fact check -> text-to-speech post-processing -> audio

The paper identifies the source textbooks, personalization fields, generation model, outline strategy, fact-checking step, voice model, and speaker consistency. This makes it possible to reason about what "personalization" actually changes.

The evaluation compares generalized podcasts, personalized podcasts, and textbook reading across three subjects with 180 students. It separates learning outcomes from experience and reports that effects differ by subject. Enjoyment does not automatically imply learning, and personalization can become distracting when analogies do not fit the content.

**Lesson:** For generative-AI systems, describe the generation and validation pipeline, then evaluate the claimed human outcome rather than treating output fluency as success.

### 2.3 Lost in Magnitudes: Narrow the Space Before the Main Experiment

Lost in Magnitudes defines boundaries, dimensions, scales, and integrity constraints before generating candidate visualizations. The authors systematically create designs, inspect them for encoding and decoding problems, derive guidelines, and only then select candidates for a controlled experiment.

The experiment compares the selected designs with meaningful existing alternatives. It includes pilots, power analysis, exclusion rules, device requirements, comprehension checks, accuracy, response time, confidence, and task-specific interpretation.

**Lesson:** When the contribution is a design space or family of techniques, evaluation should first establish coverage and constraints, then test representative designs on tasks for which they are intended.

### 2.4 DynaVis: Make Architecture Serve the Interaction Claim

DynaVis claims that natural language is useful for broad intent, while persistent widgets better support repeated fine adjustment. Its implementation makes that claim concrete:

- a declarative Vega-Lite chart representation localizes edits;
- widgets are modular HTML plus JavaScript callbacks;
- a Data Summarizer, Chart Engine, and Widget Engine divide responsibilities;
- templates constrain generated code;
- parsing, compilation, and program analysis detect or repair invalid output;
- a React and TypeScript front end communicates with a Python server;
- model selection explicitly trades accuracy against interactive latency.

The within-subject study compares DynaVis with a credible natural-language baseline, then uses task completion, interaction logs, workload, preference, error analysis, latency, and retry counts. The strongest result is not simply that users preferred DynaVis. It is that logs show a changed strategy: users used natural language for broad edits and widgets for repeated exploration.

**Lesson:** The architecture should explain why the proposed interaction is possible, reliable enough, and behaviorally different from the baseline.

### 2.5 Piet: Explain the Representation Behind the Interface

Piet begins with expert workflow evidence and derives three design goals: contextual feedback, group-wise manipulation, and progressive authoring. The system then represents color at three synchronized levels:

- global video theme;
- scene-level distribution over time; and
- element-level detail.

The paper explains its Lottie data representation, color extraction, CIE LAB conversion, clustering, palette geometry, sorting, grouping, synchronized highlighting, and group adjustment behavior. The implementation section is relatively short, but the system section already exposes the core computational and interaction logic.

The expert evaluation uses realistic recoloring tasks and examines not only ratings but panel-use sequences and editing strategies. This reveals non-linear but progressive movement between overview and detail.

**Lesson:** System description should expose the conceptual representation that users manipulate. Framework names alone do not explain the contribution.

### 2.6 Constrained Highlighting: Evaluate the Mechanism, Not the Complexity

Constrained Highlighting tests a deliberately simple mechanism: readers may highlight no text, up to 150 words, or unlimited text. The interface shows progress and rejects highlights that exceed the limit.

The strength comes from precision. The limit follows psychological theory and multiple pilots. The main study uses a large between-subjects sample, delayed testing after 24 hours, behavioral logs, highlighting strategy analysis, and workload measures.

The paper shows that the constraint changes behavior: participants select fewer and shorter phrases without spending substantially more time. It also reports frustration, document-type limitations, and questions about how constraints should adapt.

**Lesson:** HCI contribution strength depends on the importance and validation of the interaction mechanism, not the number of features or technical novelty.

### 2.7 Debate Chatbots: Operationalize the Constructs

Debate Chatbots turns social identity and rhetorical style into controlled system factors. The paper explains persona presentation, profile-image generation, debate procedure, conversation requirements, and the distinction between persuasive and eristic behavior.

The mixed factorial study measures critical thinking, stance, engagement, motivation, and perception. Qualitative analysis explains why some chatbot behavior fails: repetitive arguments, poor context use, weak evidence, aggression, and lack of reciprocity.

The paper also treats AI-generated identity and hallucination as ethical and validity concerns rather than implementation footnotes.

**Lesson:** When a paper manipulates concepts such as identity, explanation, tone, or agency, it must show exactly how the implementation produces those constructs and where that operationalization may be incomplete.

### 2.8 DataParticles: Specify State and Transformation Rules

DataParticles derives two design decisions from formative interviews and content analysis:

- infer visual behavior from story language; and
- keep narrative and visualization state together in editable blocks.

The system section explains the pipeline from text to data selection, operations, visual encoding, and animation. It also documents state inheritance, encoding priorities, transition ordering, block operations, and propagation behavior.

The expert evaluation uses reproduction and open creation tasks, then examines ease, workflow, exploration, and limitations in expressiveness and downstream control.

**Lesson:** For authoring systems, the paper should explain how state changes, how edits propagate, and what remains invariant when users reorganize work.

### 2.9 MR.Drum: Preserve the Formative-to-System Chain

MR.Drum translates instructor and learner studies into explicit implications:

- first-person demonstrations on the learner's own drum set;
- a side view for foot motion;
- BPM-based control;
- selective display of limb subsets; and
- real-time error detection and visualization.

The design section reports alternatives and refinements, including limb transparency, color coding, strike feedback, visual metronome, progress display, and AR layout. The implementation names Unity, Quest 3, MIDI input, the drum hardware, recorded expert demonstrations, and the 16-stage progression.

The comparison with instructional video controls several shared features, counterbalances conditions and learning material, and measures both performance and subjective preference. Its limits remain important: 12 participants, short practice, no delayed retention, and some results interpreted at `p < .10`.

**Lesson:** A persuasive artifact paper shows why every major feature exists, how it was realized, and which consequence was tested.

### 2.10 RoomDreaming: Use Multiple Studies for Multiple Uncertainties

RoomDreaming separates several questions:

1. What problems do owners and designers have?
2. Can the AI generate technically plausible alternatives?
3. Does the interaction support breadth and depth better than current tools?
4. Is the iterative interaction better than generation without preference support?
5. What changes after user feedback?
6. How might the system affect owner-designer collaboration?

Its architecture includes a web interface, image analyzer, prompt composer, and design generator. The implementation explains segmentation and depth maps, Stable Diffusion, ControlNet, GPT prompting, adherence controls, preference sampling, caching, hardware, and component latency.

The paper also exposes API failure, spatial-rationality limits, and the difference between estimated time savings and observed project duration.

**Lesson:** A complex system usually needs a program of evaluation. One small user study cannot validate output quality, interaction quality, collaboration, and long-term professional impact at once.

## 3. A Strong System Design Section

### 3.1 Its purpose

The System Design section should explain the research logic embodied by the artifact. It answers:

- What activity or workflow is being changed?
- Which evidence, theory, or prior system limitation motivates the design?
- What design goals follow?
- What is the core interaction mechanism?
- How does the user move through the system?
- What tradeoffs, uncertainty, agency, and safety boundaries shape the design?

This section should be understandable before the reader knows the programming language or framework.

### 3.2 Start with evidence-backed design goals

Each design goal should contain:

1. **Observed problem:** what fails in current practice.
2. **Mechanism:** what interaction property may address it.
3. **Expected consequence:** what should change for the user.
4. **Evaluation question:** how the consequence will later be tested.

Example:

| Evidence | Design goal | Implemented response | Evaluation question |
| --- | --- | --- | --- |
| Learners cannot connect delayed feedback to a specific repetition | Ground each cue in a movement episode | Timestamped cues, repetition identifier, and linked video moment | Can users identify the movement referenced by a cue? |

Avoid goals such as "easy to use," "intelligent," or "real-time" unless the paper defines what those terms mean and how they will be measured.

### 3.3 Describe the mechanism, not a feature tour

A feature tour says what screens and buttons exist. A system argument explains:

- the information available to the user;
- the actions the user can take;
- the system state that changes;
- the feedback returned;
- the timing of that feedback;
- the division of labor between person and automation; and
- why this loop supports the claimed activity.

For example, DynaVis is not mainly "a chart editor with widgets." Its mechanism is the conversion of broad natural-language intent into persistent, manipulable controls for repeated fine adjustment.

### 3.4 Expose design alternatives and tradeoffs

Good papers state what was considered and why choices were made:

- persistent versus transient feedback;
- immediate versus post-task explanation;
- automatic versus user-confirmed action;
- global versus local control;
- strict constraint versus recommendation;
- model flexibility versus predictability;
- output quality versus latency;
- broad coverage versus reliable abstention.

This makes the design look reasoned rather than inevitable.

### 3.5 Describe interaction over time

Many systems are temporal even when the interface is not animated. Show:

- startup and onboarding;
- normal task flow;
- iteration and correction;
- recovery from wrong output;
- history, undo, and review;
- exit or completion;
- repeated and long-term use where relevant.

A usage scenario is useful when it demonstrates the actual interaction logic. It should not substitute for specification.

### 3.6 Integrate uncertainty, agency, and safety

For AI systems, uncertainty handling is part of interaction design. State:

- when the system acts;
- when it asks;
- when it abstains;
- how confidence is communicated;
- how users inspect evidence;
- how users reject, correct, or override output;
- which consequences require expert review; and
- what the system refuses to claim.

## 4. A Strong Implementation Section

### 4.1 Its purpose

The Implementation section should make the artifact technically inspectable. It answers:

- How does information move through the system?
- What representations and interfaces connect components?
- Which algorithms, models, rules, prompts, or parameters determine behavior?
- What runs locally or remotely?
- What latency and resource constraints affect the experience?
- How are invalid, uncertain, or missing outputs handled?
- Which parts are research contributions and which are reused infrastructure?

The target is not full source-code documentation. The target is enough detail to understand, reproduce, and critique the claimed interaction.

### 4.2 Show the end-to-end architecture

Include one architecture figure with:

- inputs;
- preprocessing;
- model or algorithm components;
- state stores;
- decision logic;
- user-facing outputs;
- external services; and
- feedback loops.

Label whether each component runs on-device, locally, or in the cloud. Show asynchronous boundaries and cached results when they affect interaction.

### 4.3 Explain representations and contracts

Representations often determine what a system can and cannot do. Report:

- input format and assumptions;
- coordinate systems or normalization;
- window or sequence structure;
- intermediate state;
- output schema;
- timestamps and identifiers;
- confidence values;
- persistence and history;
- how one component's output becomes another component's input.

Piet's Lottie representation, DynaVis's declarative Vega-Lite specification, and DataParticles' block-level visual state are central implementation contributions because they enable the interaction.

### 4.4 Report consequential algorithms and parameters

Include values when changing them could change the user experience or study result:

- model and version;
- confidence threshold;
- time window and stride;
- retry count;
- prompt template and generation settings;
- ranking weights;
- cooldowns;
- word or action limits;
- smoothing and interpolation;
- timeout and caching;
- grouping or clustering parameters;
- fallback and abstention rules.

Do not bury these only in code or supplemental material.

### 4.5 Explain AI grounding and validation

For generative or predictive systems, document:

- training or grounding data;
- prompt inputs and output constraints;
- model version and access date;
- parsing and validation;
- fact checking or rule checking;
- retries and repair;
- hallucination or mismatch handling;
- deterministic fallback;
- privacy implications;
- what happens when the model or API is unavailable.

### 4.6 Report performance as experienced

Mean model inference time is rarely enough. Report end-to-end behavior:

- median and tail latency, such as `p50` and `p95`;
- warm-up delay;
- update frequency;
- dropped frames;
- retries;
- network delay;
- battery or thermal constraints where relevant;
- latency from user action to visible or audible response.

RoomDreaming reports latency per pipeline stage. DynaVis reports model latency and automatic retries. Code Shaping reports interpretation latency. These measurements help explain the interaction rather than merely benchmark a component.

### 4.7 Describe failure handling

Readers should know what happens when:

- no person or object is detected;
- multiple candidates appear;
- confidence is low;
- required data is missing;
- a generated program is invalid;
- an API fails;
- output contradicts a rule;
- two components disagree;
- the user rejects the result.

Failure handling should be designed and evaluated, not described as future work after the main study.

## 5. A Strong Evaluation Program

### 5.1 Evaluate at the layer of the claim

Different claims require different evidence:

| Claim | Minimum appropriate evidence |
| --- | --- |
| A component detects an event accurately | Labeled dataset, per-class metrics, uncertainty, and failure analysis |
| The end-to-end system responds in time | Device-level latency, update rate, dropped work, and realistic workload |
| An interaction mechanism changes behavior | Credible baseline or ablation, controlled tasks, logs, and user evidence |
| A system improves task performance | Objective task outcome with an appropriate comparison |
| A system improves learning | Pre/post or repeated performance, retention, transfer, and suitable comparison |
| A system supports real work | Realistic tasks with intended users or field deployment |
| A system changes collaboration | Multiple stakeholder roles and interaction-level evidence |
| A system supports long-term adoption | Longitudinal deployment and evidence of adaptation, abandonment, and maintenance |

Do not use recognition accuracy to claim coaching effectiveness. Do not use preference to claim learning. Do not use a short lab session to claim adoption.

### 5.2 Separate evaluation layers

A strong system paper may use several layers:

1. **Component validation:** model, heuristic, generation, or sensing quality.
2. **End-to-end technical evaluation:** latency, robustness, calibration, and failure propagation.
3. **Interaction evaluation:** comprehension, control, workload, strategy, and usability.
4. **Human-outcome evaluation:** task success, learning, correction, decision quality, or behavior.
5. **Deployment evaluation:** sustained use, social context, appropriation, and breakdown.

Not every paper needs every layer. It must cover every layer required by its claims.

### 5.3 Use a meaningful baseline or ablation

A baseline should isolate the proposed mechanism:

- current practice;
- a credible existing tool;
- the same system without the mechanism;
- a simpler model or interaction;
- expert or manual support where relevant.

Examples from the corpus:

- DynaVis adds generated widgets to an otherwise similar natural-language interface.
- RoomDreaming removes preference-driven iteration while preserving its generative capabilities.
- MR.Drum shares several visual supports across conditions to isolate the first-person progression experience.
- Constrained Highlighting compares no highlighting, limited highlighting, and unlimited highlighting.

### 5.4 Match participants and tasks to the contribution

Use participants who possess the knowledge or experience needed to judge the claim:

- domain experts for professional authoring systems;
- intended disabled users for accessibility claims;
- students for learning systems;
- affected stakeholders for sociotechnical systems;
- coaches and learners as separate roles for sports systems.

Tasks should produce observable evidence and preserve the consequential part of real practice. Short tasks may test an interaction mechanism, but they cannot establish long-term workflow change.

### 5.5 Combine complementary measures

Useful measures may include:

- task completion and error;
- quality judged by independent experts;
- time and latency;
- behavioral logs and interaction sequences;
- workload and usability;
- confidence and reliance;
- comprehension and recall;
- learning retention and transfer;
- interviews, observations, and think-aloud data;
- negative cases and failure recovery.

Choose each measure because it answers a research question. A standard scale is not automatically valid for every claim.

### 5.6 Analyze mechanisms and failures

The strongest evaluations explain why outcomes occurred. Include:

- usage strategies;
- condition-specific behavior;
- repair and recovery;
- disagreement with automation;
- subgroup or context differences when justified;
- non-significant results;
- technical failures;
- excluded and missing data;
- examples where the proposed mechanism did not help.

### 5.7 Calibrate conclusions

Use language that matches the evidence:

- "enabled participants to..." for observed behavior;
- "participants preferred..." for preference;
- "reduced task errors in this controlled setting..." for experimental outcomes;
- "suggests a design opportunity..." for formative evidence;
- "did not establish..." for missing long-term or causal evidence.

## 6. Traceability Templates

### 6.1 Design traceability table

| ID | Evidence or theory | Design requirement | System response | Intended consequence | Evaluation |
| --- | --- | --- | --- | --- | --- |
| DR1 | [Observed problem] | [Required interaction property] | [Implemented feature or rule] | [Expected user change] | [Measure or study] |

### 6.2 Claim-evidence matrix

| Claim | Required evidence | Evidence provided | Result | Boundary |
| --- | --- | --- | --- | --- |
| [Bounded claim] | [Technical or human evidence] | [Study or dataset] | [Supported, mixed, unsupported] | [Where it may not transfer] |

### 6.3 System component table

| Component | Input | Processing | Output | Key parameter | Failure behavior | User consequence |
| --- | --- | --- | --- | --- | --- | --- |
| [Component] | [Format] | [Algorithm/model] | [Format] | [Value] | [Fallback/abstention] | [Visible effect] |

### 6.4 Evaluation layer table

| Layer | Question | Method | Measure | Current evidence |
| --- | --- | --- | --- | --- |
| Component | Does the detector identify the target event? | Labeled benchmark | Precision, recall, calibration | [Available/missing] |
| End-to-end | Does feedback arrive reliably and in time? | Device test | `p50/p95` latency, false cues, dropped frames | [Available/missing] |
| Interaction | Can users understand and act on feedback? | Comparative user study | Referent identification, actionability, workload | [Available/missing] |
| Outcome | Does practice improve? | Repeated or longitudinal study | Correction, retention, transfer | [Available/missing] |

## 7. Recommended Section Structure

```text
4 SYSTEM DESIGN
4.1 Evidence-Based Design Goals
4.2 User Roles, Context, and Scope
4.3 Core Interaction Loop
4.4 Feedback, Control, and Recovery
4.5 Uncertainty, Safety, and Human Authority
4.6 Usage Scenario and Design Tradeoffs

5 IMPLEMENTATION
5.1 End-to-End Architecture
5.2 Inputs, Representations, and State
5.3 Models, Algorithms, and Parameters
5.4 Interaction and Feedback Realization
5.5 Latency, Resource Use, and Deployment
5.6 Validation, Failure Handling, and Fallbacks

6 EVALUATION
6.1 Claims and Evaluation Questions
6.2 Technical Dataset and Ground Truth
6.3 Component and End-to-End Metrics
6.4 User Study Design
6.5 Results by Claim
6.6 Failures, Negative Evidence, and Limitations
```

The exact headings should reflect the contribution. A simple interaction technique may combine design and implementation, while a complex AI system may need separate technical and user evaluations.

## 8. Common Failure Modes

- Presenting screens in order without explaining the interaction mechanism.
- Listing frameworks, models, and APIs without explaining why they were chosen.
- Hiding consequential thresholds, prompts, rules, or model versions.
- Describing a feature as evidence that a design requirement has been satisfied.
- Treating implementation completeness as proof of usefulness.
- Reporting only average latency while ignoring delayed or failed responses.
- Evaluating an AI model offline and claiming the whole system works for users.
- Using SUS or preference as the only evaluation of a learning or performance claim.
- Comparing with an unrealistically weak baseline.
- Changing several mechanisms at once without an ablation or clear interpretation.
- Ignoring model uncertainty, false feedback, and user override.
- Omitting technical failures, API outages, exclusions, or researcher assistance.
- Reporting only successful examples or statistically significant effects.
- Claiming long-term impact from one short session.
- Treating experts and novices as interchangeable participants.
- Describing limitations generically instead of narrowing the contribution.

## 9. Review Rubric

Score each dimension as `0 = absent`, `1 = partial`, or `2 = strong`.

| Dimension | Review question |
| --- | --- |
| Human problem | Is the system tied to a specific activity, user, and consequence? |
| Design evidence | Do design goals follow from formative evidence, theory, or prior work? |
| Core mechanism | Is the interaction contribution more precise than a feature list? |
| Traceability | Can features be traced backward to evidence and forward to evaluation? |
| Alternatives | Are major design tradeoffs and rejected options explained? |
| User agency | Can users inspect, control, correct, or reject system behavior? |
| Architecture | Is the end-to-end data and control flow clear? |
| Representations | Are the states, formats, and component contracts explained? |
| Parameters | Are consequential models, thresholds, prompts, and timing rules reported? |
| Failure handling | Are invalid, missing, low-confidence, and unavailable states handled? |
| Technical evidence | Are accuracy, latency, robustness, and calibration tested as needed? |
| Baseline | Does the comparison isolate the claimed mechanism? |
| Human evidence | Do participants, tasks, and measures fit the human claim? |
| Mechanism analysis | Do results explain behavior, strategies, and failures? |
| Scope | Are claims bounded by setting, duration, sample, and system limits? |
| Reproducibility | Are artifacts and procedural details available where feasible? |

Interpretation:

- **27-32:** strong and internally coherent system paper;
- **20-26:** promising, with identifiable evidence or reporting gaps;
- **12-19:** substantial mismatch among design, implementation, and evaluation;
- **0-11:** artifact description rather than a defensible HCI system contribution.

The score is diagnostic, not a mechanical publication rule.

## 10. Application to the AI Fencing Coach Paper

The current paper already contains a strong starting traceability table with five design requirements. Sections 4-6 should now make those requirements concrete and testable.

### 10.1 Recommended system-design argument

The core contribution should not be framed as:

> We combine pose estimation, FenceNet, heuristics, text-to-speech, and reports.

A stronger interaction claim is:

> AI Fencing Coach coordinates uncertain movement recognition with prioritized, temporally grounded feedback so learners can notice one actionable issue during practice, inspect richer evidence afterward, and retain authority to question or reject the system.

The design section should organize the artifact around the existing requirements:

1. reduce diagnostic burden without assuming no self-awareness;
2. keep live feedback brief and move explanation to review;
3. ground each cue in a movement episode;
4. make recognition uncertainty inspectable and contestable; and
5. preserve learner and coach authority.

For each requirement, describe the interaction state, not only the feature.

### 10.2 Recommended interaction flow

Document the complete loop:

1. The learner selects practice mode, target side, pose backend, voice, and feedback focus.
2. The camera searches for and locks onto the intended fencer.
3. The system distinguishes idle, checking, and active fencing states.
4. Active frames fill a motion window.
5. The system predicts an action and checks fencing-specific movement rules.
6. Multiple issues are ranked.
7. Up to three visual cues remain available while one spoken cue is scheduled.
8. The learner may continue, pause, mute categories, or end practice.
9. The session report presents counts, repeated cues, and a cue timeline.
10. The learner or coach reviews evidence and decides what to practice next.

Also document recovery:

- no target;
- target switch;
- pose dropout;
- model warm-up;
- low action confidence;
- contradictory or implausible cue;
- user disagreement;
- summary service unavailable.

### 10.3 Recommended implementation architecture

The current Android runtime can be reported as:

```text
CameraX frame
-> MediaPipe or YOLO pose backend
-> pose mapping and target tracking
-> activity gatekeeper
-> spatial normalization
-> 28-frame FenceNet window with stride 10
-> action prediction with 0.60 confidence threshold
-> fencing-specific heuristic checks
-> feedback ranking and cooldown
-> visual overlay, speech, cue history, and session report
```

The paper should explain that the current implementation:

- runs the connected coaching path on-device;
- supports MediaPipe and YOLO pose backends;
- tracks one selected target through short pose dropouts;
- prevents idle frames from filling the action-recognition window;
- classifies six FenceNet actions;
- evaluates heuristics over recent raw skeletons;
- ranks errors using base priority, persistence, novelty, aging, repetition, and user focus;
- shows up to three visual cues;
- applies a 4-second per-error voice cooldown and 1.2-second global voice cooldown;
- keeps pending errors for 5 seconds;
- records cue timestamps, action counts, latency, FPS, and estimated dropped frames; and
- retains deterministic playbook feedback when optional generated summaries are unavailable.

These values should be described as implementation choices, not validated design truths. The evaluation must determine whether they produce understandable timing, acceptable interruption, and appropriate prioritization.

### 10.4 Required technical evaluation

The technical evaluation should use coach-labeled fencing recordings rather than only the training data used by FenceNet.

Recommended conditions:

- beginner, intermediate, and advanced skill;
- left- and right-facing stance;
- different body proportions and clothing;
- camera distance, height, and angle;
- lighting and background variation;
- slow and fast movement;
- single fencer and opponent present;
- partial occlusion and pose dropout;
- supported and unsupported actions;
- correct technique and deliberately enacted errors.

Recommended measures:

| Component | Measures |
| --- | --- |
| Pose and tracking | pose availability, target-lock accuracy, identity switches, dropout duration, interpolation frequency |
| Action recognition | per-class precision, recall, macro F1, confusion matrix, calibration, coverage at the threshold |
| Heuristics | per-error precision and recall against coach labels, false cues per minute, missed actionable errors |
| Feedback scheduler | cue selection agreement with coach priorities, repeated-cue rate, time between concurrent issues and presentation |
| End-to-end pipeline | capture-to-cue `p50/p95` latency, FPS, dropped frames, warm-up time, battery and thermal behavior |
| Grounding | percentage of cues correctly linked to the intended repetition or timestamp |
| Abstention | coverage versus error under confidence and pose-quality thresholds |

Report performance by error type and condition. An aggregate accuracy can hide a harmful cue that is consistently wrong.

### 10.5 Required interaction evaluation

The current four-participant study can support feasibility and design-diagnostic claims. It cannot establish skill improvement.

For the next comparison, use a credible baseline such as ordinary recorded-video self-review. A counterbalanced within-subject design could compare matched drill sets if learning carryover is managed.

Recommended outcomes:

- whether the learner notices the same issue as a coach;
- whether the learner identifies which repetition a cue refers to;
- whether the cue is understood without researcher explanation;
- whether the next repetition changes in the intended direction;
- interruption, workload, and cue overload;
- agreement, rejection, and correction of system feedback;
- trust calibrated to correct and intentionally incorrect cues;
- coach-blinded technique ratings;
- delayed retention and transfer to an unpracticed drill.

Separate the following:

- **error awareness:** "I noticed a problem";
- **diagnostic correctness:** "The named problem was correct";
- **actionability:** "I knew what to change";
- **immediate correction:** "The next attempt changed";
- **learning:** "The change persisted and transferred."

### 10.6 Claim-evidence plan for the paper

| Proposed claim | Current evidence | What is still needed |
| --- | --- | --- |
| The prototype can run a complete phone-based coaching workflow | Implemented Android pipeline and preliminary use | Device-level reliability and latency across phones and conditions |
| Prioritized feedback can reduce diagnostic burden | Participant reports and examples | Baseline comparison with independent coach judgment |
| Layered live and post-session feedback supports interpretation | High post-review ratings and interviews | Interaction logs and a larger study isolating the layers |
| Temporal ambiguity is a central failure mode | Repeated interview evidence | Measured cue latency and referent-identification accuracy |
| Recognition errors affect appropriate trust | Observed misclassification cases | Controlled exposure to correct and incorrect cues, plus correction or abstention controls |
| The system improves fencing skill | Not established | Objective correction, retention, transfer, and coach-blinded assessment |

### 10.7 Suggested paper figures and tables

1. **Design traceability figure:** formative evidence -> DR1-DR5 -> interaction components -> evaluation questions.
2. **System architecture figure:** camera to pose, tracking, activity, recognition, heuristics, scheduling, and outputs.
3. **Feedback timeline figure:** movement episode, detection window, inference, cue scheduling, speech, and next repetition.
4. **Feedback-rule table:** trigger, required joints, action context, threshold, confidence or abstention, output, and fencing rationale.
5. **Technical evaluation table:** per-error accuracy, false-cue rate, latency, coverage, and failure cases.
6. **Claim-evidence table:** what the preliminary study supports and what remains untested.

## 11. Paper-Ready Checklist

### System Design

- [ ] The human activity and current failure are concrete.
- [ ] Each design goal follows from evidence, theory, or prior work.
- [ ] The core interaction mechanism is stated in one sentence.
- [ ] User roles and division of labor with AI are explicit.
- [ ] Normal, iterative, and recovery flows are described.
- [ ] Major alternatives and tradeoffs are visible.
- [ ] Uncertainty, correction, privacy, safety, and authority are designed.
- [ ] A traceability table links goals to implementation and evaluation.

### Implementation

- [ ] An end-to-end architecture figure is included.
- [ ] Input, intermediate, and output representations are specified.
- [ ] Models, versions, data sources, prompts, and dependencies are named.
- [ ] Consequential thresholds, windows, weights, and timing values are reported.
- [ ] On-device, server, and external-service boundaries are clear.
- [ ] Latency and resource constraints are reported as experienced by users.
- [ ] Invalid, unavailable, and low-confidence behavior is described.
- [ ] Reused infrastructure is distinguished from the research contribution.
- [ ] Code, prompts, materials, or supplementary details are shared where feasible.

### Evaluation

- [ ] Every claim has an evaluation layer and measure.
- [ ] Technical and human evaluations are separated where necessary.
- [ ] The baseline represents current practice or isolates the mechanism.
- [ ] Participants have the expertise or lived experience needed for the claim.
- [ ] Tasks preserve the consequential part of real use.
- [ ] Objective, behavioral, subjective, and qualitative evidence are combined only when needed.
- [ ] Effect sizes, uncertainty, exclusions, and missing data are visible.
- [ ] Failure cases, negative results, and user disagreement are analyzed.
- [ ] Conclusions do not exceed the setting, sample, duration, or system capability.

## Final Definition

A good HCI system paper makes the artifact inspectable as both a designed interaction and an implemented technical system. Its design goals are grounded in human evidence or theory; its core mechanism is more precise than a feature list; its implementation exposes the representations, algorithms, parameters, timing, and failure handling that shape use; and its evaluation tests each claim at the correct technical, interaction, human-outcome, or deployment layer.

Across this corpus, the strongest papers differ greatly in technical complexity. Their shared strength is a visible chain of reasoning. The reader can see why the system was designed, how it works, what evidence supports it, how it fails, and exactly what the paper is entitled to claim.

## Sources

- Official SIGCHI award and program records linked in Section 1
- The ten DOI-linked papers in the corpus table
- User-provided PDFs: `MR.Drum.pdf` and `RoomDreaming.pdf`
- Current project implementation under `android/app/src/main/java/com/aifencingcoach/runtime`

