# What Makes a Good HCI Formative Study and Analysis?

Status: Methodological review and project guidance  
Last updated: 2026-06-13  
Target venue context: ACM CHI

## 1. Scope and Corpus

This document synthesizes ten recent CHI papers, with emphasis on how they use formative studies and analysis to move from an uncertain problem to defensible design knowledge.

This is a purposive methodological review, not a systematic literature review. The corpus was selected to cover qualitative need-finding, expert practice elicitation, design probes, iterative design studies, field deployments, psychophysics, and quantitative design-space analysis.

### Award-status clarification

The two supplied papers, **MR.Drum** and **RoomDreaming**, are CHI full papers and are included as requested. However, as of June 13, 2026, neither appears in the official CHI Best Paper lists for its year. The corpus therefore contains:

- eight verified CHI Best Paper Award papers from 2022-2025
- the two supplied CHI papers as additional methodological anchors

Official award lists:

- [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers)
- [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers)
- [CHI 2023 Best Papers](https://programs.sigchi.org/chi/2023/awards/best-papers)
- [CHI 2022 Best Papers](https://programs.sigchi.org/chi/2022/awards/best-papers)

### Papers reviewed

| Year | Paper | Status in this corpus |
|---|---|---|
| 2025 | [MR.Drum: Designing Mixed Reality Interfaces to Support Structured Learning Micro-Progression in Drumming](https://doi.org/10.1145/3706598.3714156) | Supplied CHI paper; not listed as Best Paper |
| 2024 | [RoomDreaming: Generative-AI Approach to Facilitating Iterative, Preliminary Interior Design Exploration](https://doi.org/10.1145/3613904.3642901) | Supplied CHI paper; not listed as Best Paper |
| 2025 | [AACessTalk: Fostering Communication between Minimally Verbal Autistic Children and Parents](https://doi.org/10.1145/3706598.3713792) | Best Paper |
| 2025 | [Code Shaping: Iterative Code Editing with Free-form AI-Interpreted Sketching](https://doi.org/10.1145/3706598.3713822) | Best Paper |
| 2025 | [Traversing Dual Realities: Investigating Techniques for Transitioning 3D Objects between Desktop and AR](https://doi.org/10.1145/3706598.3713949) | Best Paper |
| 2024 | [Mitigating Barriers to Public Social Interaction with Meronymous Communication](https://doi.org/10.1145/3613904.3642241) | Best Paper |
| 2024 | [Sensible and Sensitive AI for Worker Wellbeing](https://doi.org/10.1145/3613904.3642716) | Best Paper |
| 2023 | [Understanding the Benefits and Challenges of Deploying Conversational AI Leveraging LLMs for Public Health Intervention](https://doi.org/10.1145/3544548.3581503) | Best Paper |
| 2023 | [CiteSee: Augmenting Citations in Scientific Papers with Persistent and Personalized Historical Context](https://doi.org/10.1145/3544548.3580847) | Best Paper |
| 2022 | [AirRacket: Perceptual Design of Ungrounded, Directional Force Feedback](https://doi.org/10.1145/3491102.3502034) | Best Paper |

## 2. Working Definition

### Formative study

A good HCI formative study is a **decision-oriented empirical inquiry conducted before or during design that reduces consequential uncertainty about people, practices, contexts, mechanisms, risks, or design alternatives**.

Its purpose is not merely to prove that a problem exists or that participants like an idea. It should change what the researchers understand, build, prioritize, avoid, or evaluate.

### Formative analysis

A good formative analysis is a transparent reasoning process that turns observations or measurements into defensible design knowledge:

> research uncertainty -> research question -> sample and setting -> data -> codes or measures -> findings -> design rationale -> implemented choice -> later evaluation

The quality of the analysis depends on the strength of this chain. A large interview count cannot repair a weak chain, while a small but information-rich study can be valuable when its claims are narrow and traceable.

## 3. What the Ten Papers Demonstrate

| Paper | Strong formative or analytical move | Limitation or caution |
|---|---|---|
| **MR.Drum** | Combined instructor demonstrations, think-aloud interviews, and learner interviews. Two researchers used reflexive thematic analysis; a second instructor wave checked whether the framework needed refinement. The analysis converted 82 exercises into a two-dimensional, 16-stage learning framework and then into five interface implications. | Small, specialized samples; the claimed sufficiency of the framework is stronger than a formal saturation account. |
| **RoomDreaming** | Asked owners and designers to bring real completed projects, grounding interviews in concrete artifacts and episodes. The study covered both sides of the owner-designer relationship and translated frictions into breadth, depth, quality, and control goals. | The paper does not clearly report recording, transcription, coding, analyst roles, or how themes were produced. The design traceability is clearer than the analytical procedure. |
| **AACessTalk** | Interviewed nine experts across several autism-related professions and five parents. It used a video prototype and comic-strip recall task to elicit concrete experiences, desired behavior, and AI risks. Findings map explicitly to three design rationales and safeguards. | One researcher open-coded the data before team discussion. More detail about disagreement, negative cases, and parent-expert tensions would strengthen the analysis. |
| **Code Shaping** | Used three consecutive design-study stages with six new programmers each. Sketches, logs, recordings, think-aloud, observation, and interviews were analyzed to produce a sketch codebook, error taxonomy, repair strategies, and workflow model. Each stage directly redesigned the next. | Convenience samples and short artificial tasks constrain claims about professional programming practice. |
| **Traversing Dual Realities** | Used pilot testing, an initial controlled study, explicit technique refinement, and then an expert study in computational chemistry workflows. Mixed performance data, observations, questionnaires, and thematic analysis supported both refinement and domain validation. | The initial task was intentionally simple and artificial; the sample was gender-skewed and the expert study was small. |
| **Meronymous Communication** | A 20-scholar formative study identified a central tension: anonymity lowers social risk but weakens trust. The analysis produced six named design goals, a stakeholder model, and a system built on an existing social platform, followed by a month-long field study. | The formative sample was US-based and dominated by computing fields; the primary author led coding. |
| **Sensible and Sensitive AI** | Converted theory and prior qualitative work into a four-factor hypothesis space. A 28-person pilot established a baseline, then 110 workers evaluated 1,059 randomized vignettes. Power analysis and mixed-effects models accounted for repeated responses and individual differences. | Vignettes reveal judgments about hypothetical deployments, not actual adoption behavior; the sample was limited to US information workers. |
| **CareCall** | Studied an already deployed public-health system through 34 people across users, teleoperators, and developers. Focus-group observation and interviews were analyzed together; peer debriefing produced a documented codebook with 10 parent and 24 child codes. | Users were regular adopters and mostly low-income men in their 50s and 60s. Developer perspectives outnumbered direct user interviews, and conversation logs were unavailable. |
| **CiteSee** | Five researchers performed real paper-search and think-aloud tasks before reacting to eight design probes. The analysis produced three direct design goals. Controlled and field studies then tested whether the resulting mechanism worked in practice. | The preliminary sample was very small and analysis was led by one researcher, so the formative claims should remain focused on design direction rather than prevalence. |
| **AirRacket** | Used a chain of studies to answer different design uncertainties: initial experience, perceived force magnitude, acceptable duration, detection threshold, and final model comparison. Established psychophysical methods and counterbalancing turned user reactions into calibrated design parameters. | Laboratory experiences and mostly young or novice participants limit generalization to expert athletes and extended use. |

## 4. Characteristics of a Good Formative Study

### 4.1 It begins with a design uncertainty

A formative study should state what the researchers do not yet know and what decision depends on learning it.

Strong examples:

- How do instructors decompose a complex skill?
- Which interaction breakdowns prevent a child from taking conversational agency?
- Which identity signals reduce anxiety without destroying trust?
- Which dimensions of workplace sensing cause utility or harm?

Weak framing:

- "We interviewed users to understand their needs."
- "We wanted feedback on our idea."
- "We asked whether participants would use AI."

### 4.2 It studies current practice before the imagined system

Good formative work first investigates:

- what people currently do
- where and when breakdowns occur
- what workarounds already exist
- what expertise is tacit
- what costs, values, and power relations shape behavior

Questions about a proposed AI feature should usually come later. Otherwise, the study risks measuring enthusiasm for a description rather than understanding the underlying activity.

### 4.3 It elicits concrete episodes, artifacts, or performances

The strongest studies do more than ask for opinions. They make practice observable:

- MR.Drum asked instructors to demonstrate teaching and explain their reasoning.
- RoomDreaming used participants' completed design projects.
- AACessTalk used conversation-recall comics and an early video prototype.
- CiteSee observed real paper search and reading.
- Code Shaping collected the sketches, edits, errors, and repair actions themselves.

Concrete evidence reduces hindsight bias and exposes details that generic interviews miss.

### 4.4 Sampling follows the knowledge and power structure

Formative sampling is usually purposive, not statistically representative. Participants should be selected because they illuminate different parts of the activity.

Useful dimensions include:

- novice, intermediate, and expert users
- direct users and secondary users
- people who create, supervise, maintain, or are affected by the system
- successful adopters, reluctant users, and dropouts
- people with different constraints, abilities, and institutional power

CareCall is strong because users, teleoperators, and developers reveal different incentives and risks. AACessTalk similarly avoids relying only on parents or only on clinicians.

### 4.5 The method matches the uncertainty

| Uncertainty | Suitable methods |
|---|---|
| Current workflow and breakdowns | Contextual inquiry, observation, diary, interview grounded in recent episodes |
| Tacit expert practice | Demonstration, think-aloud, stimulated recall, artifact walkthrough |
| Reactions to a possible interaction | Design probes, storyboards, video prototypes, Wizard-of-Oz |
| Competing design parameters | Controlled experiment, psychophysics, method of adjustment |
| A risky design space that cannot be deployed | Factorial survey or experimental vignette |
| Multi-stakeholder consequences | Stakeholder interviews, field observation, policy or workflow analysis |
| Longitudinal appropriation | Field deployment, logs, diaries, repeated interviews |

No method is inherently "more HCI." Quality comes from fit between the question, method, evidence, and claim.

### 4.6 Data collection creates an auditable corpus

A credible paper explains what was captured:

- recordings and transcripts
- observation or field notes
- screenshots, sketches, or physical artifacts
- interaction logs and errors
- survey instruments and task outcomes
- who produced each item and under what condition

Readers should be able to understand what the analysis operated on. "We interviewed participants and found three themes" is not enough.

### 4.7 Analysis is explicit and appropriate

For qualitative analysis, report:

- the analytical approach and why it fits
- whether coding was inductive, deductive, or hybrid
- who coded what
- how the codebook or themes changed
- how disagreements or alternative interpretations were handled
- whether data sources were analyzed together or separately
- how negative cases and stakeholder disagreements were treated
- what software or representation supported the analysis, when relevant

For quantitative analysis, report:

- variables and their operationalization
- design and counterbalancing
- sample-size or power rationale
- exclusions and missing data
- model choice and assumptions
- repeated-measure handling
- effect sizes and uncertainty, not only significance
- robustness or sensitivity checks

Inter-rater reliability is not mandatory for reflexive thematic analysis, but the paper must still explain how interpretations were developed and challenged.

### 4.8 Findings become an intermediate knowledge artifact

Strong formative analysis produces more than a list of complaints. Common outputs include:

- workflow or process model
- taxonomy of breakdowns or errors
- design dimensions and tradeoffs
- stakeholder or value model
- learning progression
- hypothesis space
- design principles or requirements
- prioritized opportunity areas

Examples in the corpus include MR.Drum's micro-progression framework, Code Shaping's error and repair taxonomy, Meronymity's stakeholder model and six design goals, and Sensible and Sensitive AI's four-factor hypothesis space.

### 4.9 Findings preserve tensions rather than averaging them away

Good HCI findings often take the form of a tension:

- privacy versus credibility
- automation versus human control
- immediate correction versus interruption
- realism versus hardware capability
- broad exploration versus convergence
- parental guidance versus child agency

A useful analysis explains when each side matters and which design response manages the tension. A weak analysis simply reports the most common preference.

### 4.10 Design implications are traceable

Every major feature should be traceable to evidence, and every major finding should have a clear consequence.

Use a table like this:

| Evidence-backed finding | Design requirement | Implemented response | Evaluation question |
|---|---|---|---|
| Learners cannot monitor subtle errors while executing movement | Feedback must reduce divided attention | Detect and flag the moment automatically | Do learners notice and correct errors without disrupting practice? |
| Coaches prioritize foundational errors before minor details | Feedback needs a severity and dependency order | Show one high-priority correction at a time | Does prioritization match coach judgment and improve actionability? |

Avoid the unexplained leap: "Participants wanted support, therefore we built an AI assistant."

### 4.11 Formative evidence is allowed to reject the original idea

A study is not genuinely formative if the design cannot change. Researchers should report:

- ideas abandoned after evidence
- features narrowed or delayed
- assumptions contradicted
- cases where no intervention is appropriate
- risks that require human control or escalation

Code Shaping is especially strong here: features introduced in one stage were found disruptive or unnecessary and were removed or redesigned in the next.

### 4.12 Claims are calibrated to the evidence

Small qualitative studies can support claims about:

- observed practices
- mechanisms and tensions
- design opportunities
- plausible requirements
- concepts that should be tested

They normally cannot establish:

- population prevalence
- causal effectiveness
- long-term adoption
- superiority over alternatives
- general learning or health outcomes

The paper should identify its setting, sample skew, missing stakeholders, and plausible alternative explanations.

## 5. What Good Findings Look Like

A strong finding usually contains four parts:

1. **Claim:** a pattern, mechanism, tension, or process.
2. **Evidence:** episodes, artifacts, counts, quotations, or model estimates.
3. **Variation:** who or when the finding does not apply.
4. **Consequence:** what the finding changes about design or evaluation.

Example:

> During complex footwork, learners allocate attention to movement execution and cannot reliably self-monitor timing. This appeared across observed error episodes and participant accounts, but experienced learners sometimes detected gross errors through bodily sensation. Therefore, the system should prioritize low-interruption detection for subtle timing errors while preserving manual review and coach override.

This is stronger than:

> Participants wanted real-time feedback.

## 6. Common Failure Modes

- Interviewing only convenient users while ignoring experts, maintainers, or affected stakeholders.
- Asking mostly hypothetical feature questions.
- Treating interest, novelty, or stated willingness as adoption evidence.
- Reporting themes without explaining how they were produced.
- Listing quotations without making an analytical claim.
- Converting every request into a feature rather than identifying the underlying need.
- Counting mentions as if frequency alone established importance.
- Hiding disagreement, failed ideas, or negative cases.
- Presenting obvious findings without a new model, mechanism, or design consequence.
- Claiming saturation without explaining the sampling sequence or what stopped changing.
- Using formative interviews to justify a design that was already fixed.
- Calling a usability test "formative research" when it only identifies interface defects.

## 7. Recommended Reporting Structure

### 7.1 Purpose

- What consequential uncertainty motivated the study?
- Which design or research decision depended on it?

### 7.2 Research questions

Use narrow questions about practice, breakdowns, mechanisms, values, or tradeoffs.

### 7.3 Participants and setting

- Who participated and why were they information-rich?
- Which roles, expertise levels, and power positions were included?
- Who is missing?

### 7.4 Procedure and materials

- What did participants do, recall, demonstrate, or react to?
- Which concrete artifacts or probes were used?
- How did the procedure avoid leading participants?

### 7.5 Data corpus

- What was recorded or logged?
- How much data was collected?
- How was it prepared and protected?

### 7.6 Analysis

- Analytical approach
- Analyst roles
- Coding or modeling process
- Iterations and checks
- Treatment of disagreements and negative cases

### 7.7 Findings

Organize by analytical claims, not interview questions. For each finding, include evidence, variation, and implications.

### 7.8 Design translation

Show a finding-to-requirement-to-feature traceability table. State which ideas were rejected or bounded.

### 7.9 Limitations and reflexivity

Discuss sample and setting boundaries, researcher relationships, institutional power, interpretation choices, and what the study cannot establish.

## 8. A Formative Study Blueprint for the AI Fencing Coach

### 8.1 Design uncertainties to resolve

1. How do coaches notice, interpret, and prioritize fencing errors?
2. Which errors can students self-detect, and which remain invisible during solo practice?
3. When is immediate feedback useful, distracting, unsafe, or pedagogically premature?
4. What evidence makes AI feedback understandable and trustworthy to students and coaches?
5. Which decisions must remain under coach or learner control?

### 8.2 Recommended staged study

#### Stage A: Expert coaching practice

Recruit approximately 6-8 coaches with varied coaching experience and student levels.

Use:

- teaching demonstrations
- think-aloud correction of standardized and real fencing clips
- prioritization tasks when multiple errors co-occur
- walkthroughs of drills, notes, and coaching vocabulary

Capture:

- detected error
- evidence used
- severity and prerequisite relationships
- correction language
- preferred drill or intervention
- conditions under which the coach would withhold feedback

Primary output:

- an error-prioritization and coaching-progression framework

#### Stage B: Learner practice and workarounds

Recruit approximately 8-12 beginner and intermediate fencers who practice alone or with limited supervision.

Use:

- observation of real practice
- video-stimulated recall immediately after selected episodes
- walkthroughs of mirrors, recordings, teammate comments, and delayed coach questions
- discussion of concrete recent mistakes before introducing the proposed system

Primary output:

- a breakdown taxonomy of error detection, interpretation, correction, and follow-through

#### Stage C: Feedback probes and co-design

Use low-fidelity or Wizard-of-Oz feedback examples that vary:

- timing: immediate, between repetitions, post-set, post-session
- modality: audio, visual overlay, short text, replay
- granularity: one correction versus several
- certainty: confident detection versus uncertain suggestion
- source framing: AI observation, coach-authored rule, or combined evidence

Ask participants to act on the feedback, not merely rate screenshots.

Primary output:

- bounded design requirements and rejected alternatives

### 8.3 Suggested analysis

Build a multimodal corpus of transcripts, observed episodes, video timestamps, artifacts, and probe interactions.

Use a hybrid analysis:

- deductive codes from the practice-feedback cycle: perform, notice, interpret, prioritize, correct, verify
- inductive codes for unexpected strategies, tensions, and failures
- an episode matrix comparing student interpretation with coach interpretation
- a negative-case log for examples that contradict the emerging framework

Two researchers should independently examine an initial subset, discuss interpretations, refine the codebook or thematic structure, and then continue with documented peer debriefing. This is not simply an agreement exercise; its purpose is to expose assumptions and improve the analytical account.

Candidate analytical dimensions:

- movement or tactical error
- visible evidence
- who notices it
- confidence
- severity
- prerequisite dependency
- ideal feedback moment
- correction action
- verification method
- interruption cost
- trust requirement
- privacy or embarrassment risk

### 8.4 Required outputs before system redesign

- a workflow model of current solo-practice feedback
- a taxonomy of missed errors and failed workarounds
- a coach-informed prioritization framework
- a set of feedback timing and modality tradeoffs
- a finding-to-design traceability table
- a list of AI behaviors the system should not perform
- explicit hypotheses for summative evaluation

### 8.5 Example traceability

| Formative finding | Design implication | Evaluation measure |
|---|---|---|
| Students notice errors too late after a practice set | Preserve timestamped moments and provide brief between-repetition cues | Detection latency and next-repetition correction |
| Coaches prioritize foundational posture before secondary speed errors | Rank feedback by dependency and show one correction first | Coach agreement and learner actionability |
| Learners distrust advice without visible evidence | Pair feedback with a short replay and highlighted movement evidence | Explanation usefulness and appropriate reliance |
| Immediate cues disrupt complex sequences | Let feedback timing adapt to drill phase and user preference | Interruption, workload, and performance continuity |
| Some cases cannot be judged reliably from one camera | Communicate uncertainty and request another view or coach review | Calibration, false-confidence rate, and override behavior |

## 9. Reviewer-Facing Checklist

Before submitting, the formative section should let a reviewer answer "yes" to the following:

- Is the design uncertainty explicit?
- Do the research questions match that uncertainty?
- Are participants justified by the knowledge and stakeholder structure?
- Does the study examine real practices, episodes, artifacts, or behavior?
- Is the captured data corpus clear?
- Is the analysis method transparent and appropriate?
- Are findings more analytical than descriptive?
- Are variation, disagreement, and negative cases visible?
- Does each design implication trace to evidence?
- Did the study materially change or constrain the design?
- Are claims proportionate to the evidence?
- Are ethics, power, privacy, and missing stakeholders addressed?
- Does later evaluation test the consequences of the formative findings?

## 10. Bottom Line

A good CHI formative study is not defined by interviews, sample size, or the word "thematic." It is defined by whether it produces a credible and useful transformation:

> from an underspecified human problem to a structured account of practice, breakdowns, mechanisms, values, and tradeoffs that visibly changes the design and creates testable evaluation claims.

The best papers in this corpus make that transformation inspectable. Their formative work produces frameworks, design goals, taxonomies, parameter ranges, or hypothesis spaces; their later studies then test whether those products survive contact with use.
