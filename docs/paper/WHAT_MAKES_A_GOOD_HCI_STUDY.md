# What Makes a Good HCI Study?

Status: Research synthesis  
Last updated: 2026-06-13  
Scope: Ten recent CHI papers from 2023-2026, including the two supplied papers

## Short Definition

A good HCI study makes a consequential but bounded claim about people and interactive technology, then supports that claim with methods whose participants, setting, comparison, measures, analysis, ethics, and reporting all fit the claim.

Quality is therefore not synonymous with a large sample, a novel AI model, statistical significance, or a polished prototype. The strongest studies create a traceable chain:

> real human problem -> grounded understanding -> design rationale -> appropriate study -> credible evidence -> bounded contribution

## Corpus and Award Verification

Award status was checked against the official SIGCHI program's Best Paper lists for [CHI 2023](https://programs.sigchi.org/chi/2023/awards/best-papers), [CHI 2024](https://programs.sigchi.org/chi/2024/awards/best-papers), [CHI 2025](https://programs.sigchi.org/chi/2025/awards/best-papers), and [CHI 2026](https://programs.sigchi.org/chi/2026/awards/best-papers).

Important correction: the official programs do **not** mark the supplied `MR.Drum` or `RoomDreaming` papers as Best Paper Award recipients. They are included because they were explicitly requested. The corpus therefore contains eight verified Best Papers and two additional recent CHI papers.

| Year | Paper | Status | Main evidence |
|---|---|---|---|
| 2026 | [When Scaffolding Breaks](https://doi.org/10.1145/3772318.3791517) | Best Paper | Six-week deployment with 157 eighth-grade students; logs, assessment, observation, and teacher interview |
| 2026 | [When Workout Buddies Are Virtual](https://doi.org/10.1145/3772318.3790303) | Best Paper | Six-month, four-group randomized trial with 280 participants and 30 follow-up interviews |
| 2025 | [AACessTalk](https://doi.org/10.1145/3706598.3713792) | Best Paper | Formative interviews followed by a two-week home deployment with 11 parent-child dyads |
| 2025 | [A Qualitative Study on How Usable Security and HCI Researchers Judge Effect Sizes](https://doi.org/10.1145/3706598.3714022) | Best Paper | Paper audit, surveys, interviews, and a statistics expert interview |
| 2024 | [Understanding Feedback in Rhythmic Gymnastics Training](https://doi.org/10.1145/3613904.3642434) | Best Paper | Ten training observations and 16 interviews with gymnasts and coaches |
| 2024 | [DynaVis](https://doi.org/10.1145/3613904.3642639) | Best Paper | Counterbalanced within-subject lab study with 24 participants |
| 2023 | [Changes in Research Ethics, Openness, and Transparency](https://doi.org/10.1145/3544548.3580848) | Best Paper | Preregistered analysis of 245 sampled CHI papers using 45 criteria |
| 2023 | [Understanding Frontline Workers' and Unhoused Individuals' Perspectives on AI Used in Homeless Services](https://doi.org/10.1145/3544548.3580882) | Best Paper | AI lifecycle comicboarding with 21 participants across three stakeholder groups |
| 2025 | [MR.Drum](https://doi.org/10.1145/3706598.3714156) | Requested CHI paper, not listed as Best Paper | Two formative studies and a 12-person comparative lab study |
| 2024 | [RoomDreaming](https://doi.org/10.1145/3613904.3642901) | Requested CHI paper, not listed as Best Paper | Five studies with homeowners and professional interior designers |

## What the Ten Papers Teach

### 1. When Scaffolding Breaks

The paper studies an LLM writing assistant in an actual school for six weeks rather than assuming that a successful one-session demo will become a successful classroom intervention. Its strongest move is examining the surrounding classroom: learner proficiency, time pressure, teacher attention, peer learning, and equity. It discovers that scaffolding can improve task completion while frustrating lower-proficiency students, increasing reliance, and hiding student difficulties from teachers.

The study is bounded by one school, limited direct student testimony, analysis of a sample of the interaction logs, and partial reliance on an LLM to judge grammar. These limitations appropriately restrict, rather than erase, the contribution.

**Lesson:** Evaluate how a system reorganizes a social practice, not only whether an individual can operate it.

### 2. When Workout Buddies Are Virtual

The research question concerns sustained motivation, so the authors use a long randomized study with a post-intervention period. Objective step data, validated scales, interviews, mixed-effects models, safety monitoring, and open materials provide complementary evidence. The paper also reports unsupported hypotheses and distinguishes social presence from working alliance instead of reducing both to "users liked the AI."

Attrition, an academic iPhone-owning sample, an underpowered design for smaller effects, and the weak scientific basis of the fixed 10,000-step goal constrain the findings. The authors discuss these directly.

**Lesson:** The duration, comparison, and outcomes must match the phenomenon being claimed.

### 3. AACessTalk

The system follows from formative work with autism professionals and parents, then enters homes for two weeks. Recruitment includes expert screening, child assent considerations, flexible use, remote troubleshooting, and deliberate efforts not to frame the system as compulsory teaching. The evaluation combines logs, repeated surveys, interviews, and qualitative interpretation of how each family appropriated the tool.

The child participants' experience is still largely interpreted through parents. The sample was selected through one expert and excluded families for whom the system might be unsuitable. The authors also identify cultural mismatch and the risk that an LLM could impose neurotypical communication norms.

**Lesson:** With vulnerable or marginalized participants, method design, participant agency, and possible normative harm are part of research validity.

### 4. Effect-Size Judgement

This paper shows why a statistically significant result is not automatically important. Researchers interpreted identical standardized effects differently, sometimes misunderstood the measures, and drew on context, consequences, affected populations, and familiar reference points. The paper recommends reporting standardized and interpretable real-world effects, uncertainty, and a qualitative account of practical importance.

Its participants were not representative of all HCI researchers, and its small surveys were used descriptively rather than for population inference.

**Lesson:** Report how large an effect is, how uncertain it is, and why that magnitude matters in the study's real context.

### 5. Understanding Feedback in Rhythmic Gymnastics

Observation of real training reveals that feedback varies by timing, form, quantity, exercise complexity, group context, and the athlete's skill on the specific exercise. An athlete can be advanced in one movement and a beginner in another. The paper translates these findings into concrete design implications, including progressive difficulty, warm-up, and different feedback timing for fast versus slow movements.

The study covers one competitive academy. Its first author is also a coach, which provides access and expertise but creates interpretive risk; the team explicitly uses reflexivity and challenges insider-derived interpretations.

**Lesson:** Domain expertise should generate situated design rules, while reflexivity prevents expertise from silently becoming evidence.

### 6. DynaVis

DynaVis begins with a precise interaction claim: natural language is useful for broad intent, while persistent widgets may be better for repeated fine adjustment. The evaluation uses a credible NLI baseline, counterbalancing, telemetry, completion outcomes, workload measures, interviews, and failure analysis. The results explain not just preference but the behavior that produced it.

The study is a short lab session with a university-centered sample. It cannot establish long-term usability, whether a changing interface becomes disorienting, or whether the result transfers to expert workflows.

**Lesson:** A good comparison isolates the proposed interaction mechanism and investigates why it changes behavior.

### 7. Changes in Research Ethics, Openness, and Transparency

The authors operationalize broad ideals into 45 inspectable criteria, preregister the study, justify the quantitative sample, use stratified random sampling, document deviations, check coder agreement, and share materials. Particularly useful criteria include consent, compensation, safeguards for vulnerable participants, sample-size justification, demographics, study design, analysis procedures, effect sizes, confidence intervals, and research artifacts.

The audit measures whether practices were reported, not whether the underlying research decisions were correct. The 2017 and 2022 samples also differ in paper-length constraints, and the criteria are less complete for interpretive qualitative methods.

**Lesson:** Good research must be inspectable, but a checklist supports judgment rather than replacing it.

### 8. AI in Homeless Services

The study method is designed around the participation barriers in its context. Comics expose hidden AI lifecycle choices; short text supports varied reading literacy; a neutral persona reduces pressure to disclose personal experience; one-on-one sessions address stigma; and de-identified ideas support deliberation across groups with unequal power. Participants without AI training provide detailed criticism of objectives, proxies, data, overrides, and the surrounding service system.

The work concerns one region and deployed system, and it does not experimentally compare comicboarding with another method. It also cannot guarantee that the resulting feedback changes government policy.

**Lesson:** Meaningful participation requires making technical choices legible and actively designing around power.

### 9. MR.Drum

The paper has a strong formative-to-design chain. Professional instructors provide a domain-grounded micro-progression framework, self-learning drummers reveal current pain points, and the resulting system is compared with the dominant instructional-video practice. The summative study collects performance, preference, and qualitative evidence with counterbalanced conditions.

The evaluation has 12 participants, two 15-minute practice sessions, and no delayed retention or field-use evidence. Novelty remains possible, and results reported at `p < .10` should be treated as suggestive rather than strong confirmation.

**Lesson:** An artifact becomes an HCI contribution when its interaction logic is grounded in human practice and tested on meaningful human outcomes.

### 10. RoomDreaming

RoomDreaming uses multiple studies to move from stakeholder needs to AI-output assessment, self-guided comparison, system revision, and owner-designer collaboration. It includes both professional and non-professional stakeholders and compares the complete interaction approach against current tools and a generative-AI baseline without iterative preference support.

The evidence is distributed across small studies. The final co-design findings cover only three pairs after an API failure, and estimated days saved are expert estimates rather than observed project durations. The work evaluates preliminary exploration, not construction quality or long-term professional adoption.

**Lesson:** Iterative studies can build a persuasive cumulative argument, but each claimed benefit must still be tied to directly observed evidence.

## Ten Criteria for a Good HCI Study

### 1. Start with a specific human problem

The problem should name who is affected, in what activity and context, what currently goes wrong, and why the consequence matters. "Use AI for fencing" is a technology topic. "Help intermediate fencers notice recurring footwork errors during solo practice when a coach is unavailable" is a research problem.

### 2. Make a bounded contribution claim

State whether the contribution is an empirical finding, artifact, interaction technique, method, dataset, theory, framework, or combination. Do not let model accuracy stand in for learning, usefulness, trust, adoption, safety, or social benefit.

### 3. Choose methods from the question

There is no universally superior HCI method.

- Use observation or ethnography to understand situated practice.
- Use interviews or participatory methods to understand experience, meaning, values, and power.
- Use controlled experiments to isolate causal effects.
- Use deployment and longitudinal studies for adoption, adaptation, sustained behavior, and breakdowns.
- Use mixed methods only when each strand answers a necessary part of the question.

### 4. Make design decisions traceable to evidence

Important features should follow from formative findings, domain knowledge, theory, prior work, or clearly identified design hypotheses. A reader should be able to ask "Why is this feature here?" and find an evidence-based answer.

### 5. Use a meaningful comparison

Compare with the actual current practice, a credible state-of-the-art system, or a mechanism-level ablation. A weak or artificial baseline can make any prototype appear successful. When no comparison is appropriate, explain why and use a method suited to exploration rather than causal claims.

### 6. Measure the claimed outcome

Usability, preference, task performance, learning, retention, behavior change, trust, agency, and safety are different outcomes. Select measures that match the claim and combine behavioral, experiential, and contextual evidence when needed. For learning technologies, immediate success is not enough; retention and transfer often matter.

### 7. Match setting and duration to the claim

A lab study can support a narrow interaction claim. It usually cannot support claims about sustained adoption, classroom equity, long-term coaching, or everyday behavior. Field and longitudinal work become necessary when context, novelty, changing routines, or social relationships are central.

### 8. Treat ethics, inclusion, and power as methodological quality

Report consent, compensation, privacy, safety, researcher-participant relationships, and additional safeguards. Adapt participation methods to accessibility, literacy, age, culture, and power differences. Ask whose perspective is missing and whether the system imposes the designer's norms on participants.

### 9. Analyze magnitude, uncertainty, and mechanism

For quantitative work, justify sample size, report attrition and exclusions, check assumptions, report effect sizes and uncertainty, and interpret practical importance in domain terms. For qualitative work, name the analytic approach, explain how interpretations were developed, include positionality where relevant, and connect themes to adequate evidence. In both cases, explain why the observed result may have occurred.

### 10. Be transparent about boundaries and artifacts

Share protocols, interview guides, stimuli, prompts, code, data, and analysis where ethical and feasible. Document deviations and failures. Limitations should identify exactly where the claim may stop transferring. Open artifacts improve inspectability, but participant safety can legitimately constrain openness.

## A Practical Review Rubric

Score each item as `0 = absent`, `1 = partial`, or `2 = strong`. The total is a diagnostic, not a mechanical definition of publishability.

| Criterion | Review question |
|---|---|
| Problem | Is the human problem specific, consequential, and supported by evidence? |
| Claim | Is the contribution clear and no broader than the evidence? |
| Method | Does the study design answer the research question? |
| Participants | Are the relevant people represented, with sampling and sample size justified? |
| Context | Does the setting and duration match the intended use and claim? |
| Design traceability | Can major system features be traced to evidence or explicit hypotheses? |
| Comparison | Is the baseline realistic and fair? |
| Measures | Do the measures capture the claimed human outcomes? |
| Analysis | Are analysis choices, uncertainty, alternative explanations, and mechanisms clear? |
| Ethics and power | Are agency, safety, privacy, inclusion, and researcher positionality addressed? |
| Transparency | Are materials and deviations available where feasible? |
| Contribution | Does the study produce knowledge that travels beyond "participants liked our prototype"? |

## Implications for the AI Fencing Coach

### Recommended central question

> How can AI-generated feedback help beginner and intermediate fencers notice and correct a bounded set of footwork errors during solo practice, while remaining understandable, appropriately timed, safe, and compatible with coach instruction?

This is stronger than asking whether the model can recognize fencing actions. Recognition is enabling infrastructure; the HCI claim concerns whether feedback improves practice.

### Study 1: Understand real coaching and solo practice

Observe practice across more than one context if feasible and interview both fencers and coaches. Document:

- errors learners repeatedly fail to notice;
- how coaches decide what to correct first;
- feedback timing before, during, and after a movement;
- how feedback changes with movement complexity and exercise-specific skill;
- current workarounds such as mirrors, self-video, peers, and delayed coach review;
- safety concerns, trust boundaries, and situations where the system should abstain.

The output should be a grounded feedback taxonomy and explicit design requirements, not only a list of desired features.

### Study 2: Validate the technical foundation

Before making learning claims, evaluate detection by error type, skill level, camera condition, body type, clothing, handedness, and movement speed. Report confusion matrices, latency, confidence calibration, failure cases, and the consequence of false feedback. The system should abstain or request review when confidence is insufficient.

### Study 3: Compare learning support

Use a credible baseline such as the learner's current mirror/self-video workflow, not only a no-feedback condition. A counterbalanced within-subject design may efficiently compare conditions, provided learning carryover is handled with equivalent movement sets.

Useful outcomes include:

- error correction on the next repetition;
- error recurrence over a practice block;
- coach-blinded ratings of technique;
- delayed retention and transfer to an unpracticed drill;
- time required to identify and correct an error;
- comprehension, workload, trust, and perceived actionability;
- over-reliance, disagreement with coaches, and responses to incorrect feedback.

Report effect sizes and confidence intervals, then explain whether the observed magnitude is meaningful in actual training.

### Study 4: Deploy in practice

A multi-week deployment should examine whether novelty fades, which feedback people ignore, how learners adapt the tool, when coaches accept or override it, and whether it changes peer or coach interaction. Logs should be interpreted alongside interviews, observation, and coach assessments rather than treated as self-explanatory.

### Safety and role boundaries

- Require appropriate warm-up before demanding drills.
- Avoid real-time feedback during movements when it would overload or endanger a learner.
- Limit the number of simultaneous corrections according to exercise-specific skill.
- Present the system as a practice aid, not an authority or coach replacement.
- Give coaches a way to inspect, correct, and configure feedback.
- Protect video, pose, identity, and health-related data.
- Include uncertain and incorrect AI feedback in the evaluation.

## Common Failure Modes

- Starting with a technology and inventing a user problem afterward.
- Using convenience participants who do not represent the claimed users.
- Treating preference as proof of learning or effectiveness.
- Treating classification accuracy as proof of HCI value.
- Comparing against an unrealistically weak baseline.
- Running a short lab study while claiming long-term adoption.
- Reporting only statistically significant findings.
- Reporting `p` values without magnitude, uncertainty, or practical meaning.
- Ignoring the social system around the user, such as coaches, peers, classrooms, or institutions.
- Describing limitations generically without narrowing the contribution.
- Calling a study user-centered while participants only evaluate decisions already made.
- Sharing everything without considering participant privacy, or sharing nothing without explaining the ethical constraint.

## Final Definition

A good HCI study is coherent, situated, ethical, and inspectable. It asks a worthwhile question about people and technology; grounds its design in real practices; chooses methods, participants, comparisons, measures, and duration that can answer that question; interprets both effects and mechanisms; and states clearly what the evidence does and does not establish.

The best papers in this corpus differ greatly in method and scale. Their shared strength is not a formula. It is disciplined alignment between the human problem, the contribution, and the evidence.
