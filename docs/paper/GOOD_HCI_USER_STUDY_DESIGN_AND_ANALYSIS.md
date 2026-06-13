# What Makes a Good HCI User Study Design and Analysis?

Status: Methodological review and paper-writing guide  
Last updated: 2026-06-13  
Target venue context: ACM CHI  
Project context: AI-assisted fencing coaching

## Executive Summary

A good HCI user study creates a visible chain from:

> research claim -> study design -> participants and tasks -> measures -> analysis -> evidence -> bounded conclusion -> design consequence

The ten reviewed papers show that quality does not come from sample size alone. Strong papers:

- state exactly what uncertainty each study resolves;
- select participants who can provide the required evidence;
- use realistic tasks, meaningful baselines, and controlled procedures;
- combine behavioral, subjective, and qualitative evidence when the claim requires it;
- describe exclusions, failures, analyst decisions, and limitations;
- report effect sizes, uncertainty, and non-significant findings rather than only favorable p-values; and
- connect findings to concrete design changes or reusable HCI knowledge.

For a preliminary study with only 4-6 users, the defensible goal is normally to identify feasibility problems, interpretation failures, workflow breakdowns, risks, and design requirements. It is generally not enough evidence to claim that a system improves learning, performance, trust, or adoption.

No complete 4-6-participant result dataset was supplied for the AI fencing coach. Section 6 therefore provides a paper-ready reporting structure with placeholders. It must be filled with observed data rather than presented as completed findings.

## 1. Review Scope

This is a purposive methodological review, not a systematic literature review. It examines ten recent CHI papers selected for their useful study-design patterns.

### 1.1 Award-status clarification

The supplied papers, **MR.Drum** and **RoomDreaming**, are CHI full papers and are included as requested. As of June 13, 2026, they are not listed as Best Paper recipients on the official award pages checked. The corpus therefore consists of:

- eight verified CHI Best Paper Award papers from 2022-2025; and
- two requested CHI comparison papers.

Official award records:

- [SIGCHI FY2022 Annual Report](https://sigchi.org/about/annual-reports/fy-2022/)
- [SIGCHI FY2023 Annual Report](https://sigchi.org/about/annual-reports/fy-2023/)
- [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers)
- [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers)

### 1.2 Papers reviewed

| Paper | Year and status | Study architecture reviewed | Main methodological lesson |
| --- | --- | --- | --- |
| [MR.Drum](https://doi.org/10.1145/3706598.3714156) | 2025, requested comparison | Two formative studies with 8 instructors and 8 learners, followed by a 12-person within-subject evaluation | Derive the interaction from an explicit learning progression, then test objective and subjective consequences |
| [RoomDreaming](https://doi.org/10.1145/3613904.3642901) | 2024, requested comparison | Five studies across owners and designers, from need-finding to AI-output assessment and co-design | Use staged studies to connect stakeholder needs, technical quality, self-guided use, redesign, and collaborative use |
| [AirRacket](https://doi.org/10.1145/3491102.3502034) | 2022, Best Paper | Five sequential studies with 72 unique participants | Resolve separate perceptual and interaction uncertainties through focused experiments before the final comparison |
| [Mobile-Friendly Content Design for MOOCs](https://doi.org/10.1145/3491102.3502054) | 2022, Best Paper | Survey of 134 learners, 21 interviews, analysis of 101 MOOCs, and 11 expert interviews | Triangulate user experience, objective artifact evidence, and practitioner knowledge |
| [DataParticles](https://doi.org/10.1145/3544548.3581472) | 2023, Best Paper | Six-expert formative study and nine-expert evaluation | Match an expert authoring contribution with expert participants, realistic creation tasks, and qualitative usage analysis |
| [Take My Hand](https://doi.org/10.1145/3544548.3581415) | 2023, Best Paper | Preliminary study with 12 blindfolded participants, followed by a target-population study with 7 blind or low-vision participants | Use a preliminary study to repair the protocol, but validate accessibility claims with the intended population |
| [AI in Homeless Services](https://doi.org/10.1145/3544548.3580882) | 2023, Best Paper | In-depth work with 21 frontline workers and currently or formerly unhoused people | Adapt the method to literacy, stigma, access, and power, and make researcher positionality visible |
| [Constrained Highlighting](https://doi.org/10.1145/3613904.3642314) | 2024, Best Paper | Three pilots followed by a 127-person between-subjects comparison with delayed testing | Pilot task difficulty and parameters before testing the hypothesis; report effect sizes and uncertainty |
| [Piet](https://doi.org/10.1145/3613904.3642711) | 2024, Best Paper | Six-expert formative study and 13-expert evaluation | Observe authentic expert workflows, then evaluate both task completion and patterns of tool use |
| [Letters from Future Self](https://doi.org/10.1145/3706598.3714206) | 2025, Best Paper | One-week, three-condition study with 36 participants and follow-up interviews | Use validated scales, reliability checks, repeated measurement, and mixed-method analysis without turning time effects into false condition effects |

## 2. Definition of a Good HCI User Study

A good HCI user study is an empirical design in which the participants, setting, tasks, comparisons, measures, and analysis are sufficient to answer a clearly bounded question about human interaction with a technology, practice, or sociotechnical system.

The standard is not "Did the study find significance?" The standard is:

1. Was the research question consequential and answerable?
2. Did the design isolate or expose the phenomenon relevant to that question?
3. Did the participants and tasks represent the knowledge or use context being claimed?
4. Were the measures valid for the construct?
5. Was the analysis appropriate and transparent?
6. Do the conclusions stay within the evidence?
7. Did the result create reusable knowledge or a justified design decision?

### 2.1 Match the claim to the study type

| Intended claim | Suitable study evidence | Claims not justified by that evidence alone |
| --- | --- | --- |
| Users currently experience a workflow breakdown | Observation, contextual inquiry, artifact walkthrough, episode-based interview | Population prevalence or causal effect |
| A concept is understandable and usable | Think-aloud usability study, task completion, errors, interpretation probes | Long-term adoption or learning improvement |
| One interface causes a short-term difference | Randomized or counterbalanced comparison with controlled conditions | Generalization outside tested users, tasks, and duration |
| A system supports real work over time | Field deployment, logs, diaries, repeated interviews | Effects beyond the deployment setting or duration |
| A tool improves learning | Suitable baseline, pre/post measures, retention or transfer tests, controlled exposure | Durable expertise from immediate performance alone |
| A design is accessible | Evaluation with the intended disabled population in relevant contexts | Accessibility based only on non-disabled simulation |

## 3. What Strong Study Design Should Include

### 3.1 An explicit uncertainty and research question

Each study should resolve one or more named uncertainties:

- **Formative:** What do people do, need, value, or struggle with?
- **Design-space:** Which parameters or interaction alternatives are viable?
- **Feasibility:** Can people complete the workflow and understand the output?
- **Comparative:** Does the proposed design change an outcome relative to a meaningful baseline?
- **Explanatory:** Why, when, or for whom does the effect occur?
- **Longitudinal:** How does use change after novelty and initial training?

AirRacket is especially strong because separate studies estimate magnitude perception, acceptable duration, detection, and final user experience. One oversized experiment would have made those uncertainties harder to diagnose.

### 3.2 A staged evidence strategy

A strong system paper often needs more than one study:

1. **Formative study:** understand practice and derive requirements.
2. **Pilot:** test tasks, timing, instrumentation, and failure cases.
3. **Main evaluation:** test the central contribution.
4. **Field or follow-up study:** examine retention, appropriation, or real-world consequences when the claim requires it.

RoomDreaming and AirRacket show the value of staged evidence. Take My Hand shows that a preliminary study should change the protocol: it exposed fatigue and an unsuitable assembly task before the target-population study.

### 3.3 Participants justified by the claim

Report:

- inclusion and exclusion criteria;
- recruitment source;
- relevant experience, ability, and role;
- demographic or contextual characteristics that affect interpretation;
- compensation;
- recruited, completed, excluded, and analyzed counts; and
- why this sample can answer the question.

Do not call a sample "representative" without a sampling design that supports that claim.

Specialized claims require specialized participants. DataParticles and Piet recruited experienced creators. Take My Hand did not rely on blindfolded participants for its final accessibility claims. The homeless-services study included people affected by the system, not only administrators or developers.

### 3.4 Sample size tied to analytical purpose

Sample-size reasoning differs by study:

- For experiments, use an a priori power or precision rationale when feasible.
- For qualitative studies, justify information richness, role coverage, and sampling sequence.
- For expert evaluations, explain why the selected expertise is necessary.
- For small pilots, state that the goal is protocol repair or feasibility, not effect estimation.

Avoid an unsupported statement such as "five users are enough." Four to six users may reveal important usability problems, but issue discovery depends on task coverage, participant diversity, system maturity, and problem frequency.

### 3.5 Realistic and diagnostic tasks

Tasks should:

- represent the activity named in the paper;
- require participants to act, decide, create, interpret, or correct something;
- expose the mechanism the interface is intended to support;
- have observable success or failure criteria; and
- avoid unnecessary ceiling or floor effects.

Strong papers use concrete materials:

- RoomDreaming used real design projects.
- Piet asked professionals to author and recolor motion graphics.
- Mobile MOOC examined actual course videos in addition to self-report.
- Constrained Highlighting piloted reading duration and highlight limits before the final study.

### 3.6 Meaningful conditions and baselines

A baseline should isolate the contribution rather than merely make the new system look favorable.

Depending on the claim, suitable comparisons may include:

- current practice;
- a standard tool;
- an instructional video;
- the same system without the proposed mechanism;
- unconstrained interaction;
- a human-authored alternative; or
- multiple plausible designs.

For within-subject designs, report counterbalancing, order, practice, fatigue, carryover, and washout. MR.Drum counterbalanced system order and learning material. AirRacket repeatedly used balanced orders and shuffled stimuli.

For between-subject designs, report random assignment, allocation counts, attrition, and baseline comparability. Letters from Future Self used three randomly assigned conditions and separated changes over time from differences between conditions.

### 3.7 A reproducible procedure

Readers should be able to reconstruct:

- location and equipment;
- consent and training;
- task instructions;
- session phases and duration;
- number of trials or artifacts;
- break and fatigue management;
- condition order;
- researcher intervention;
- use of deception or Wizard-of-Oz behavior;
- data captured; and
- debriefing.

Wizard-of-Oz control can be valid when the purpose is to isolate an interaction concept, as in Take My Hand, but it must be disclosed and bounded. A perfect simulated recognizer cannot justify claims about a deployed model's reliability.

### 3.8 Measures aligned with the construct

Use multiple evidence types only when each answers part of the research question.

**Behavioral and performance measures**

- task success;
- errors and recovery;
- time or latency;
- number and type of actions;
- correction on a subsequent attempt;
- retention or transfer;
- interaction logs; and
- abandonment or help requests.

**Subjective measures**

- validated workload, usability, trust, learning, or experience scales;
- preference when preference is actually the construct;
- confidence and perceived control; and
- discomfort, fatigue, or sickness.

**Qualitative measures**

- think-aloud observations;
- stimulated recall;
- semi-structured interviews;
- field notes;
- artifacts and sketches; and
- accounts of disagreement, risk, and unexpected use.

Do not substitute satisfaction for effectiveness. "Participants liked the feedback" does not show that they understood it or corrected their movement.

### 3.9 Ethics, safety, and power

Report ethics review or the applicable review process, informed consent, compensation, data protection, and withdrawal.

For AI coaching and embodied activity, also address:

- physical safety and fatigue;
- recording of bodies and training spaces;
- bystanders in video;
- model uncertainty and incorrect advice;
- embarrassment and social comparison;
- coach authority and learner autonomy; and
- who can inspect, correct, or delete generated feedback.

The homeless-services paper demonstrates that method design itself can reduce harm: one-on-one sessions, accessible materials, de-identified cross-stakeholder ideas, and explicit positionality were part of research validity, not administrative details.

### 3.10 Data-quality and failure reporting

State:

- attention or comprehension checks;
- technical failures;
- missing observations;
- exclusion criteria and when they were decided;
- outlier handling;
- protocol deviations;
- unusable recordings; and
- whether pilot data entered the final analysis.

RoomDreaming reports that an API failure reduced one co-design study from four pairs to three usable pairs. Such failures should remain visible because they affect both sample and ecological validity.

## 4. What Strong Analysis Should Include

### 4.1 Quantitative analysis

Before selecting a test, identify:

- unit of analysis;
- independent and dependent variables;
- within- or between-subject structure;
- repeated observations per person or item;
- distribution and scale type;
- planned contrasts;
- missingness and exclusions; and
- whether the analysis is confirmatory or exploratory.

Report:

- descriptive statistics for every condition;
- participant-level or distribution plots where possible;
- model or test and why it fits;
- assumption checks;
- correction for multiple comparisons;
- exact p-values;
- effect sizes;
- confidence intervals or other uncertainty estimates; and
- non-significant and contrary results relevant to the research question.

Examples from the corpus:

- AirRacket used non-parametric comparisons, correction for repeated testing, and effect sizes.
- Constrained Highlighting used non-parametric tests, Holm correction, effect sizes, and bootstrap confidence intervals.
- Letters from Future Self checked scale reliability and separated time, condition, and interaction effects.
- Mobile MOOC normalized artifact data so courses with different video lengths could be compared.

Avoid:

- changing the significance threshold after seeing the data;
- calling a marginal p-value "significant";
- treating many trials from one participant as independent people;
- selecting only favorable dependent variables;
- interpreting a time effect as evidence that the new interface beat the baseline;
- claiming equivalence from a non-significant difference; and
- using pilot tuning and final confirmation on the same data without disclosure.

### 4.2 Qualitative analysis

Name the method and its role. Possible approaches include reflexive thematic analysis, codebook thematic analysis, content analysis, grounded-theory procedures, interaction analysis, or a structured framework analysis. "We coded the interviews" is not an analysis description.

Report:

- the complete data corpus;
- transcription or preparation;
- inductive, deductive, or hybrid orientation;
- analyst identities and relevant positionality;
- who examined which data;
- how codes, categories, or themes changed;
- meetings, memos, or affinity representations;
- how disagreement and alternative interpretations were handled;
- treatment of negative cases and stakeholder conflicts;
- selection and translation of quotations; and
- how findings produced design or theoretical claims.

Reliability statistics are method-dependent:

- Mobile MOOC used a structured coding process and reported Cohen's kappa.
- The homeless-services paper used reflexive thematic analysis and explicitly did not treat inter-rater reliability as its validity goal.

Neither choice is universally correct. The paper should make its epistemological and procedural logic coherent.

Strong qualitative results include:

1. a clear analytical claim;
2. evidence from episodes, observations, artifacts, or quotations;
3. variation or a negative case;
4. the mechanism or tension represented; and
5. the consequence for design, theory, or later evaluation.

### 4.3 Mixed-method integration

Mixed methods are not simply a survey plus an interview. Explain how evidence relates:

- **Convergence:** behavioral and interview evidence support the same interpretation.
- **Complementarity:** one source explains the mechanism behind another.
- **Expansion:** different methods address different research questions.
- **Contradiction:** evidence conflicts and narrows the claim.

For example, a learner may rate feedback as useful while repeatedly misinterpreting the highlighted joint. That contradiction is a result, not noise to hide.

### 4.4 Trace findings to claims and design decisions

Use an evidence table:

| Finding | Evidence | Boundary or negative case | Design consequence | Later evaluation |
| --- | --- | --- | --- | --- |
| Learners miss subtle timing errors during execution | Observed repetitions, recall interview, video timestamps | Experienced learners notice large errors through bodily sensation | Delay low-priority feedback until the repetition ends | Interpretation accuracy and next-attempt correction |
| Multiple corrections overload attention | Abandoned tasks, long review times, participant accounts | Some advanced users request a full technical breakdown | Default to one prioritized correction with an optional detail view | Workload, review time, and correction choice |

This table prevents a common weak leap: "Users wanted feedback, so we built an AI coach."

## 5. Cross-Paper Principles

The strongest recurring principles across the corpus are:

1. **One study should not answer every question.** Separate need-finding, parameter tuning, feasibility, comparison, and field use when they require different evidence.
2. **Pilots should change something.** Report what failed and how the main protocol changed.
3. **Target users matter.** Simulations and convenience samples can debug a method but cannot replace the population implicated by the claim.
4. **Behavior matters alongside opinion.** Observe whether participants complete, interpret, correct, remember, or appropriate the interaction.
5. **The baseline should test the contribution.** Compare against current practice or an ablated mechanism, not an intentionally weak alternative.
6. **Use multiple measures with a reason.** Each measure should map to a construct or research question.
7. **Analysis must respect dependence.** Repeated trials, multiple artifacts, and repeated measurements require appropriate models or aggregation.
8. **Qualitative rigor is procedural and interpretive.** It is not guaranteed by multiple coders or a kappa value.
9. **Report unfavorable evidence.** Technical failures, non-significant effects, reversals, and stakeholder disagreement define the boundary of the contribution.
10. **Claims should be smaller than the evidence, never larger.** A short study may show feasibility; durable learning and adoption require stronger designs.

## 6. Reporting a Preliminary Study with 4-6 Users

### 6.1 Appropriate purpose

A 4-6-user preliminary study is suitable for:

- checking whether the end-to-end workflow functions;
- identifying severe usability and interpretation failures;
- testing whether tasks and instructions are understandable;
- locating safety, privacy, fatigue, and trust problems;
- deciding which measures are practical;
- refining feedback timing, format, and granularity;
- discovering unexpected strategies or negative cases; and
- producing design changes before a larger study.

It is generally not suitable for:

- estimating population prevalence;
- claiming thematic saturation without a strong sampling argument;
- demonstrating statistically reliable superiority;
- claiming improved fencing skill;
- validating long-term trust or adoption; or
- establishing accessibility for a broad population.

### 6.2 Recommended participant plan for the AI fencing coach

For a user-facing feasibility study, recruit **4-6 target fencers**, selected purposively rather than treated as a representative sample.

Include variation that matters to the design:

- beginner and intermediate experience;
- prior solo-practice frequency;
- familiarity with self-video;
- device and camera setup constraints; and
- relevant mobility, injury, or accessibility considerations.

If coaches are included, report them as a separate stakeholder stratum. Do not combine four learners and two coaches into a claim such as "five of six users found the coaching useful," because the roles and evaluation criteria differ.

### 6.3 Preliminary research questions

Suggested questions:

- **PRQ1:** Can fencers complete recording, analysis, and feedback review without researcher rescue?
- **PRQ2:** Do fencers correctly interpret the detected error, supporting evidence, and confidence?
- **PRQ3:** Can fencers translate the feedback into a specific next action?
- **PRQ4:** Where do timing, wording, visualization, or model failure interrupt practice?
- **PRQ5:** What safety, privacy, autonomy, or trust concerns should constrain the main study?

These questions concern feasibility and interaction quality. They do not claim learning improvement.

### 6.4 Suggested 45-60 minute protocol

1. Consent, background, and recent solo-practice experience: 5-10 minutes.
2. Standardized explanation and practice task: 5 minutes.
3. Record one or two representative fencing drills: 10 minutes.
4. Review AI feedback while thinking aloud: 10-15 minutes.
5. Explain the feedback in the participant's own words and choose a correction: 5 minutes.
6. Perform another attempt when physically safe: 5 minutes.
7. Stimulated-recall interview using selected video and interface moments: 10-15 minutes.
8. Debrief, discomfort check, and data-withdrawal reminder: 5 minutes.

Do not compare multiple interface conditions unless the purpose is to choose between prototypes. With 4-6 participants, treat such comparison as design exploration and show participant-level results rather than inferential evidence.

### 6.5 Data to capture

**Behavioral**

- task completion and researcher assistance;
- recording or tracking failures;
- time to locate and explain feedback;
- interpretation errors;
- selected correction action;
- whether the next attempt follows the intended correction;
- ignored, dismissed, or overridden feedback; and
- fatigue, discomfort, or unsafe moments.

**Subjective**

- perceived clarity, actionability, workload, and control;
- confidence in understanding the feedback;
- confidence that the system is correct;
- preferred timing and modality; and
- reasons for trust or distrust.

**Qualitative**

- think-aloud comments;
- observed hesitation and workarounds;
- concrete examples from solo practice;
- reactions to incorrect or uncertain feedback;
- participant explanations of what they would do next; and
- suggestions and rejected suggestions.

### 6.6 Analysis for 4-6 users

Use participant-level, descriptive, and qualitative analysis.

1. Build a matrix with rows for participants and columns for tasks, failures, interpretations, corrections, and concerns.
2. Mark every critical incident with a video or log timestamp.
3. Group incidents into a small set of analytical findings.
4. Record negative cases and differences by experience.
5. Assign issue severity using consequence, recoverability, and recurrence.
6. Report counts as sample descriptions, such as "4 of 6 participants," not population estimates.
7. For numeric measures, show individual values plus median and range.
8. Avoid null-hypothesis significance testing. With 4-6 users, p-values are unstable and add little to a feasibility claim.
9. Connect each supported finding to a design decision or main-study change.

Suggested severity scheme:

| Severity | Definition | Example |
| --- | --- | --- |
| Critical | Creates physical risk, invalid feedback, or prevents the core task | Feedback encourages an unsafe correction or analyzes the wrong person |
| High | Causes a likely misunderstanding or blocks independent use | Participant cannot identify which repetition the feedback describes |
| Medium | Adds substantial effort but has a workaround | Participant must replay the video several times to understand the overlay |
| Low | Cosmetic or minor friction with little effect on interpretation | Label placement is inconsistent |

### 6.7 Required result tables

#### Participant table

| ID | Role | Fencing experience | Solo-practice frequency | Self-video experience | Relevant setup or access context |
| --- | --- | --- | --- | --- | --- |
| P1 | [learner/coach] | [months/years] | [frequency] | [level] | [context] |

#### Task and measure table

| Measure | P1 | P2 | P3 | P4 | P5 | P6 | Descriptive summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Completed without assistance | | | | | | | [n/N] |
| Correctly explained feedback | | | | | | | [n/N] |
| Chose an actionable correction | | | | | | | [n/N] |
| Technical failures | | | | | | | [count and types] |
| Review time | | | | | | | [median, range] |

Delete unused participant columns rather than filling them with fabricated values.

#### Finding-to-change table

| Finding | Evidence and coverage | Variation or negative case | Severity | Design or protocol change |
| --- | --- | --- | --- | --- |
| [Analytical finding] | [episodes, quotation IDs, n/N] | [who or when it differed] | [level] | [specific revision] |

### 6.8 Paper-ready reporting template

The following text is a scaffold. Replace every bracketed field with observed evidence.

#### Preliminary Study

**Purpose.** We conducted a preliminary study to assess the feasibility and interpretability of the AI fencing coach workflow before the main evaluation. We examined whether participants could independently record a drill, locate and understand the system's feedback, and translate that feedback into a next action. We did not use this study to test learning effectiveness.

**Participants.** We recruited **[N, between 4 and 6]** fencers through **[recruitment source]**. Participants had **[range]** of fencing experience and practiced independently **[range/frequency]**. **[Describe meaningful variation.]** We compensated each participant **[amount]**. **[Ethics or review statement.]**

**Procedure.** Each **[duration]** session included **[training]**, **[recording tasks]**, feedback review with think-aloud, an explanation-in-own-words probe, **[optional next attempt]**, and a semi-structured interview. We captured **[screen/video/audio/logs/field notes]**. The researcher provided assistance only when **[rule]**, and all interventions were logged.

**Analysis.** We created a participant-by-task matrix covering completion, assistance, technical failure, feedback interpretation, correction choice, and safety or trust concerns. **[Number]** researchers reviewed **[data subset or complete corpus]** using **[inductive/deductive/hybrid]** coding. They developed findings through **[meetings/memos/code refinement]**, retained contradictory cases, and linked each finding to a design or protocol change. Numeric results are reported descriptively as participant-level values, counts, medians, and ranges.

**Feasibility results.** **[n/N]** participants completed the full workflow without assistance. The most common breakdown was **[breakdown]**, observed in **[n/N]** participants and **[number]** task episodes. **[Report technical failures, missing data, discomfort, and researcher interventions.]**

**Interpretation and actionability.** **[n/N]** participants correctly explained **[feedback element]**, while **[n/N]** misinterpreted **[element]** as **[interpretation]**. Participants selected an appropriate next action in **[n/N]** cases. A negative case was **[case]**, which indicates that **[boundary or alternative explanation]**.

**Qualitative findings.** We identified **[number]** findings: **[finding names]**. For each finding, report at least one concrete episode or quotation, participant coverage, variation, and the design consequence. Do not organize this section as a list of interview questions.

**Resulting changes.** Based on the study, we changed **[interface or protocol]** by **[specific revision]**, removed or postponed **[feature]**, and added **[safeguard or measure]** to the main study. The preliminary evidence supports the feasibility of **[bounded capability]**, but does not establish **[learning improvement, superiority, long-term adoption, or other unsupported claim]**.

### 6.9 Wording guide

Use:

- "We observed..."
- "Participants encountered..."
- "The preliminary study identified..."
- "These cases suggest a need to test..."
- "The findings motivated the following revision..."
- "Within this small, purposive sample..."

Avoid:

- "Users generally prefer..."
- "The system significantly improves..."
- "The study proves..."
- "Five users validate the design..."
- "We reached saturation..." without a defensible sampling and analysis account
- "All users..." when a participant, failure, or missing case was excluded

## 7. Recommended Evaluation Path for the AI Fencing Coach

After the 4-6-user preliminary study:

1. Repair critical workflow, interpretation, and safety failures.
2. Freeze the main interface and predefine primary outcomes.
3. Validate coaching correctness with qualified fencing coaches.
4. Run a larger comparison against current practice or a meaningful baseline.
5. Measure understanding and next-attempt correction, not satisfaction alone.
6. Add delayed retention or transfer if claiming learning.
7. Add field use across multiple sessions if claiming integration into practice.

Possible main-study outcomes:

- agreement between coach-prioritized errors and system feedback;
- learner interpretation accuracy;
- time from feedback to an appropriate correction;
- change on the next repetition;
- workload and interruption;
- appropriate reliance under correct and incorrect feedback;
- safety-related overrides; and
- retention or transfer to an unassisted drill.

## 8. Reviewer-Facing Checklist

### Study design

- [ ] The research question and intended claim are explicit.
- [ ] The study type can support that claim.
- [ ] Participant inclusion, recruitment, compensation, and analyzed counts are reported.
- [ ] Sample-size reasoning matches the analytical purpose.
- [ ] Tasks represent the claimed activity.
- [ ] Baselines isolate the proposed contribution.
- [ ] Randomization, counterbalancing, order, learning, and fatigue are addressed.
- [ ] Measures map to named constructs.
- [ ] Ethics, safety, privacy, and stakeholder power are addressed.
- [ ] Technical failures, exclusions, and protocol deviations are visible.

### Analysis

- [ ] The unit of analysis and repeated-measure structure are correct.
- [ ] Descriptive results are reported for all conditions.
- [ ] Quantitative models, assumptions, corrections, effect sizes, and uncertainty are reported.
- [ ] Non-significant and contrary findings are not hidden.
- [ ] The qualitative approach, corpus, analysts, and interpretive process are explained.
- [ ] Findings include variation and negative cases.
- [ ] Mixed-method evidence is integrated rather than merely placed side by side.
- [ ] Every major conclusion traces to evidence.
- [ ] Claims remain within the population, task, setting, and duration studied.

### Preliminary study with 4-6 users

- [ ] The study is labeled preliminary, pilot, or formative.
- [ ] Its purpose is feasibility and design refinement rather than efficacy.
- [ ] Participant-level results, counts, medians, and ranges are shown.
- [ ] No inferential significance claim is built on the tiny sample.
- [ ] Every severe failure and researcher intervention is reported.
- [ ] Findings produce concrete interface or protocol revisions.
- [ ] The paper states what the study cannot establish.

## 9. Bottom Line

A good HCI user study is not defined by having interviews, a usability scale, or a statistically significant result. It is defined by whether the research question, study structure, evidence, analysis, and conclusion form a credible argument.

For a 4-6-user preliminary study, credibility comes from rich task evidence, participant-level transparency, negative cases, visible failures, and concrete design revisions. The result should make the later study better, not imitate the certainty of a larger evaluation.

## References

1. Arakawa et al. [MR.Drum: Designing Mixed Reality Interfaces to Support Structured Learning Micro-Progression in Drumming](https://doi.org/10.1145/3706598.3714156). CHI 2025.
2. Zhang et al. [RoomDreaming: Generative-AI Approach to Facilitating Iterative, Preliminary Interior Design Exploration](https://doi.org/10.1145/3613904.3642901). CHI 2024.
3. Cheng et al. [AirRacket: Perceptual Design of Ungrounded, Directional Force Feedback to Improve Virtual Racket Sports Experiences](https://doi.org/10.1145/3491102.3502034). CHI 2022.
4. Xie et al. [Mobile-Friendly Content Design for MOOCs: Challenges, Requirements, and Design Opportunities](https://doi.org/10.1145/3491102.3502054). CHI 2022.
5. Shen et al. [DataParticles: Block-based and Language-oriented Authoring of Animated Unit Visualizations](https://doi.org/10.1145/3544548.3581472). CHI 2023.
6. Swaminathan et al. [Take My Hand: Automated Hand-Based Spatial Guidance for People with Visual Impairment](https://doi.org/10.1145/3544548.3581415). CHI 2023.
7. Kawakami et al. [Understanding Frontline Workers' and Unhoused Individuals' Perspectives on AI Used in Homeless Services](https://doi.org/10.1145/3544548.3580882). CHI 2023.
8. Joshi et al. [Constrained Highlighting in a Document Reader Can Improve Reading Comprehension](https://doi.org/10.1145/3613904.3642314). CHI 2024.
9. Zhang et al. [Piet: Facilitating Color Authoring for Motion Graphics Video](https://doi.org/10.1145/3613904.3642711). CHI 2024.
10. Kim et al. [Letters from Future Self: Augmenting the Letter-Exchange Exercise with LLM-based Future Self Agents to Enhance Young Adults' Career Exploration](https://doi.org/10.1145/3706598.3714206). CHI 2025.
