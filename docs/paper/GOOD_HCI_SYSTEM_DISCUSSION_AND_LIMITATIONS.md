# What Makes a Good HCI System Discussion and Limitations Section?

Status: Research synthesis and paper-writing guide  
Last updated: 2026-06-14  
Target venue context: ACM CHI  
Project context: AI-assisted fencing coaching

## Executive Definition

A good HCI System Discussion explains what was learned from building and
studying an interactive system, why the observed outcomes occurred, when the
system is useful or harmful, how it changes an existing human practice, and
which design knowledge travels beyond the prototype.

A good Limitations section maps the boundary of every important claim. It
distinguishes uncertainty caused by the study from constraints caused by the
system, explains the consequence of each boundary, and identifies the evidence
needed to address it.

Together, the sections should create this chain:

> result -> answer to the research question -> interaction mechanism ->
> tradeoff -> relationship to prior work -> design implication -> claim
> boundary -> next test

The Discussion is therefore not a second Results section or a celebration of
the artifact. Limitations are not a ritual list of weaknesses. Their shared
purpose is to turn local evidence into credible, reusable HCI knowledge without
claiming more than the evidence supports.

## 1. Review Scope and Corpus

This is a purposive analysis of ten recent CHI system and interaction papers,
not a systematic literature review. The analysis covered each paper's problem
and contribution framing, system design, study findings, Discussion, explicit
Limitations, and Future Work.

Award status was verified through the official SIGCHI conference program data
and the public Best Paper pages for
[CHI 2024](https://programs.sigchi.org/chi/2024/awards/best-papers),
[CHI 2025](https://programs.sigchi.org/chi/2025/awards/best-papers), and
[CHI 2026](https://programs.sigchi.org/chi/2026/awards/best-papers).

Important correction: the supplied `MR.Drum` and `RoomDreaming` papers are CHI
full papers, but the official program records do not mark them as Best Paper
recipients. They remain required corpus members because they were explicitly
requested. The corpus therefore contains eight verified Best Papers and two
requested comparison papers.

| Paper | Year and status | Strong Discussion move | Strong limitation move |
| --- | --- | --- | --- |
| [When Scaffolding Breaks](https://doi.org/10.1145/3772318.3791517) | 2026, Best Paper | Moves from individual task completion to classroom orchestration, equity, teacher awareness, and over-reliance | Bounds transfer by school, culture, language, sampled logs, and limited direct student perspectives |
| [When Workout Buddies Are Virtual](https://doi.org/10.1145/3772318.3790303) | 2026, Best Paper | Develops a "partnership paradox" explaining different mechanisms of human authenticity and AI reliability | Separates population, measurement, duration, attrition, embodiment, self-report, and LLM-safety limits |
| [Code Shaping](https://doi.org/10.1145/3706598.3713822) | 2025, Best Paper | Explains sketch-based code editing through abstraction levels, ambiguity, iteration, and the cost of structure | Distinguishes language, codebase scale, dependency, multi-file, persistence, and long-term code-quality questions |
| [PAIGE](https://doi.org/10.1145/3706598.3713460) | 2025, Best Paper | Separates enjoyment from learning and explains why personalization helps in some subjects but distracts in others | Bounds duration, content, population, culture, and interaction while identifying AI-content ethics |
| [Lost in Magnitudes](https://doi.org/10.1145/3706598.3713487) | 2025, Best Paper | Explains how a systematic design-space method produced guidelines and carefully calibrates novelty against prior designs | States that the design space is consistent but incomplete and names excluded data, tasks, interaction, and experience factors |
| [DynaVis](https://doi.org/10.1145/3613904.3642639) | 2024, Best Paper | Frames dynamic interfaces as a tradeoff between low execution cost and changing-interface burden, discoverability, and trust | Connects implementation limits to grammar expressiveness, long-session use, widget management, and domain transfer |
| [Piet](https://doi.org/10.1145/3613904.3642711) | 2024, Best Paper | Converts observed expert behavior into reusable principles for abstraction, precise control, navigation, and group editing | Separates unexplored design alternatives and workflow integration from sample and task-complexity limits |
| [Debate Chatbots](https://doi.org/10.1145/3613904.3642513) | 2024, Best Paper | Interprets persona effects, nuisance variables, role expectations, and model bias rather than reporting only main effects | Names topic sensitivity, hallucination, identity representation, short duration, artificial content, and ecological validity |
| [MR.Drum](https://doi.org/10.1145/3706598.3714156) | 2025, requested CHI paper | Argues that MR demonstration alone is insufficient and that domain-grounded micro-progression is the operative learning mechanism | Explicitly separates study limitations, hardware limits, and system limits |
| [RoomDreaming](https://doi.org/10.1145/3613904.3642901) | 2024, requested CHI paper | Defines where imperfect generative AI is useful through a speed-and-quantity versus quality tradeoff | Restricts the contribution to preliminary exploration and exposes regional preference, creativity, element control, and spatial-rationality limits |

## 2. What Each Paper Section Must Do

| Section | Primary question | Appropriate content |
| --- | --- | --- |
| Results | What did the study observe? | Measurements, themes, behavior, failures, quotations, effect sizes, uncertainty, and negative cases |
| Discussion | What do those observations mean? | Answers to research questions, mechanisms, tradeoffs, alternative explanations, relation to prior work, and implications |
| System implications | What design knowledge travels? | Evidence-backed principles about interaction, workflow, roles, controls, and sociotechnical integration |
| Limitations | Where does the inference or artifact stop? | Study, system, data, population, setting, duration, measurement, safety, and transfer boundaries |
| Future Work | What specific unresolved question should be tested next? | A study, design change, dataset, comparison, deployment, or safeguard tied to a named limitation |
| Conclusion | What is the bounded contribution? | A concise answer to the paper's central question, without new claims |

A System Discussion should not redescribe the interface. System features belong
in Discussion only when they help explain an observed mechanism, tradeoff,
failure, or design implication.

## 3. What the Ten Papers Show

### 3.1 Answer the research questions before generalizing

The Discussion should begin with a short synthesis of the principal findings
and state what they mean for the research questions or hypotheses. It should
preserve unsupported, partially supported, and contrary results.

`When Workout Buddies Are Virtual` is especially strong here. It does not hide
that several hypotheses were unsupported. It uses the pattern across measures
to formulate a more useful explanation: human peers produced stronger social
presence, while AI peers produced a steadier working alliance. The contribution
comes from explaining the divergence, not from forcing all outcomes into a
single success claim.

For a system paper, the opening should normally answer:

1. What did the system enable?
2. For whom, in which task and setting?
3. What did it fail to enable?
4. What mechanism best explains that pattern?

### 3.2 Explain the mechanism, not only the outcome

"Participants preferred the system" is an outcome. A Discussion must explain
the interaction that produced it.

- `DynaVis` attributes value to a division of labor: natural language expresses
  broad intent, while persistent widgets support repeated, precise adjustment.
- `Code Shaping` explains sketching through movement across abstraction levels,
  spatial constraint, ambiguity, feedforward, and iterative repair.
- `MR.Drum` argues that MR visualization was not sufficient by itself; the
  domain-grounded micro-progression structure controlled complexity and made
  the demonstration learnable.
- `PAIGE` separates the engaging podcast format from the learning mechanism of
  relevant personalization. Enjoyment and comprehension were not treated as
  interchangeable.

A useful mechanism statement has the form:

> The system affected **outcome Y** because **interaction X changed what the
> user could perceive, decide, express, or do**, under **conditions Z**.

### 3.3 Discuss tradeoffs instead of presenting universal benefits

Strong discussions identify what a design gains and what it gives up.

- Dynamic interfaces can reduce the gulf of execution while making the
  interface less stable and its capabilities less discoverable.
- Free-form sketches can increase expressive flexibility while introducing
  ambiguity and interpretation work.
- AI personalization can improve relevance while producing distracting,
  culturally inappropriate, or unwanted references.
- Generative design can make broad exploration inexpensive while remaining
  unsuitable for spatially rational implementation decisions.
- Human peers can provide authenticity and accountability while introducing
  social friction; AI peers can provide reliable support without genuine
  reciprocity.

Tradeoffs make a contribution more credible because they identify design
conditions rather than implying that one interface is best everywhere.

### 3.4 Connect the findings back to prior work

A good Discussion states whether the findings:

- support an established mechanism in a new context;
- refine a theory or split one construct into distinct dimensions;
- contradict an assumption in earlier systems;
- expose a boundary condition;
- combine findings that previous work treated separately; or
- explain why a prior result did not reproduce.

`When Scaffolding Breaks` challenges the assumption that withholding direct
answers is always good scaffolding. In a time-constrained classroom, the same
strategy could demotivate lower-proficiency students. `MR.Drum` uses earlier AR
learning results to argue why visualization without progression can overload
learners. `Lost in Magnitudes` explicitly recognizes similarities with prior
designs and locates novelty in the systematic method and resulting guidelines,
not in an inflated claim of inventing every visual representation.

The literature should be used as an analytical comparison, not as decoration
after the results.

### 3.5 Turn local behavior into reusable design knowledge

The strongest implications are neither feature requests nor universal laws.
They state:

1. the observed evidence;
2. the mechanism or tension;
3. the design principle;
4. the conditions under which it is expected to apply; and
5. the unresolved test.

`Piet`, for example, moves from expert use of linked views to principles about
multiple abstraction levels, direct linkage between an abstract representation
and the edited object, and group-wise manipulation. `DynaVis` moves from widget
use to questions about managing changing interfaces and exposing possible
actions. These implications travel beyond the exact prototype while remaining
traceable to observed use.

### 3.6 Discuss the whole sociotechnical practice

A system changes relationships, attention, authority, and work distribution.
The Discussion should therefore ask not only what happens between one user and
one interface, but also:

- Whose labor is removed, hidden, or created?
- Who becomes more or less visible?
- Who can challenge the system?
- Which existing collaboration is weakened or strengthened?
- Does the system redistribute expertise or authority?
- What happens when the system is wrong?

`When Scaffolding Breaks` shows that private AI help can hide common student
difficulties from a teacher and reduce peer learning. `Debate Chatbots`
examines how generated identities and model bias shape arguments. `MR.Drum`
states where professional instructors remain stronger. `RoomDreaming` defines
AI as rapid preliminary exploration rather than a replacement for professional
spatial judgment.

### 3.7 Treat failures and negative cases as evidence

Breakdowns often reveal the design boundary more clearly than favorable
ratings.

- A changing interface can become cognitively expensive during long sessions.
- A debate agent can induce reflection without changing a user's stance.
- A personalization strategy can help one subject and distract in another.
- A generative design can look attractive while being physically impossible.
- A sketch can communicate high-level intent while failing to preserve a
  low-level coding convention.

The Discussion should explain why the failure occurred and what it teaches
about the interaction. Moving every failure into Limitations wastes evidence
that may be central to the contribution.

### 3.8 Calibrate transfer and novelty

Discussion claims should move outward in controlled steps:

1. tested participants, task, system, and setting;
2. plausible mechanism supported by the evidence;
3. neighboring contexts that share the mechanism;
4. untested contexts that remain hypotheses.

Do not move directly from a short prototype study to "this approach will
transform education, health, creativity, or sport." `Lost in Magnitudes`
provides a useful model: it identifies a method and guidelines that may
generalize, while stating that the explored design space is not complete.

## 4. A Strong System Discussion Structure

### 4.1 Opening synthesis

Use one short paragraph to answer the central question:

> We found **A**, but not **B**. The evidence suggests that **mechanism C**
> explains this pattern under **conditions D**. This reframes the system's
> contribution from **overbroad claim E** to **bounded contribution F**.

Do not repeat every number or theme. Select the findings required for the
argument.

### 4.2 Interpret each research question

For each research question:

1. state the answer;
2. cite the converging and conflicting evidence;
3. explain the mechanism;
4. consider a plausible alternative explanation;
5. compare with prior work; and
6. state the design consequence.

Quantitative and qualitative evidence should do different analytical work.
Numbers can establish the pattern or magnitude; observed behavior and
interviews can explain how the pattern arose.

### 4.3 Explain system tradeoffs and breakdowns

Organize this subsection around tensions such as:

- flexibility versus predictability;
- automation versus control;
- immediacy versus reflection;
- abstraction versus precision;
- personalization versus privacy or unwanted inference;
- support versus dependence;
- scalability versus human awareness;
- consistency versus authenticity; and
- exploration speed versus output validity.

Name which side the current design favors and why. A tradeoff is useful only
when the paper identifies what should change under different conditions.

### 4.4 State design implications

Each implication should pass this test:

> Could a designer of another system identify a concrete decision that would
> change because of this finding?

Use this structure:

> Because participants **observed behavior or failure**, systems for
> **bounded class of contexts** should **design principle**, while preserving
> **countervailing need**. This remains to be tested under **boundary**.

### 4.5 Position human and system roles

State:

- what the system is competent to do;
- what it should abstain from doing;
- which decisions remain with users or experts;
- how uncertainty becomes visible;
- how errors can be challenged or corrected; and
- whether the system supplements, redistributes, or replaces existing work.

This is particularly important for AI, health, learning, accessibility, and
physical-skill systems.

### 4.6 Transition into limitations

The end of Discussion should make the contribution and its boundary legible:

> These findings support **bounded claim** about **tested mechanism and
> context**. They do not yet establish **larger claim**, because **named
> evidence is missing**.

That sentence prevents the Limitations section from feeling disconnected from
the argument.

## 5. What a Good Limitations Section Contains

### 5.1 Separate study limitations from system limitations

**Study limitations** constrain what can be inferred:

- participant selection and sample composition;
- task, material, baseline, and condition design;
- setting and ecological validity;
- duration, novelty, learning, and carryover;
- missing stakeholder perspectives;
- measurement validity and reliability;
- researcher influence;
- exclusions, attrition, and missing data; and
- analysis coverage and uncertainty.

**System limitations** constrain what the artifact can currently do:

- supported inputs, tasks, and workflows;
- recognition, generation, or sensing accuracy;
- latency and failure recovery;
- hardware burden and accessibility;
- output expressiveness and user control;
- interoperability with existing tools;
- scalability and long-term maintenance;
- privacy, bias, safety, and security; and
- uncertainty communication and contestability.

`MR.Drum` makes this separation explicit. Its short, small in-lab comparison is
a study limitation; required progression order, novice-only content, partial
body representation, and manually recorded demonstrations are system
limitations. Combining them into "the sample was small and the prototype can
be improved" would hide their different consequences.

### 5.2 Use a consequence-oriented limitation

A strong limitation contains four parts:

1. **Boundary:** What was not sampled, implemented, measured, or controlled?
2. **Consequence:** Which interpretation or use may change because of it?
3. **Current claim:** What narrower statement remains supported?
4. **Resolution:** What evidence or design change would address the boundary?

Template:

> Because the study included **boundary**, the findings may not transfer to
> **affected population, task, setting, or duration**. The current evidence
> supports **narrow claim**, but not **larger claim**. A **specific next study
> or measure** is needed to test that extension.

### 5.3 Cover the limitation categories that affect the claim

| Category | Review question | Example consequence |
| --- | --- | --- |
| Population | Who was absent or unusually represented? | Expert findings may not describe novice learning |
| Task and materials | Were tasks narrow, artificial, or unusually simple? | Performance may change with complex real projects |
| Setting | Was use controlled, supervised, or unlike real practice? | Breakdowns and workarounds may be underestimated |
| Duration | Could novelty, fatigue, adaptation, or retention change the result? | Immediate performance does not establish learning |
| Comparison | Did the baseline isolate the proposed mechanism? | Preference may reflect novelty or content differences |
| Measurement | Did measures capture the claimed construct? | Enjoyment cannot stand in for comprehension or skill |
| Analysis | Were logs sampled, data missing, or alternatives underpowered? | Important patterns or smaller effects may be absent |
| System | Which inputs, features, and workflows are unsupported? | The artifact may only support a narrow stage of work |
| AI and data | How do errors, bias, hallucination, drift, and opacity matter? | A fluent output may be unsafe or misleading |
| Human practice | Whose role, awareness, or authority changed? | Individual support may reduce teacher or coach visibility |
| Transfer | Which neighboring domains share the mechanism, and which do not? | A principle may transfer while the result does not |

Do not include a category merely to make the list longer. Include it when it
changes how the evidence should be interpreted.

### 5.4 Pair future work with evidence, not aspiration

Weak future work:

> In the future, we will add more features and recruit more users.

Strong future work:

> Because the current study cannot distinguish novelty from sustained use, a
> multi-week deployment should measure feature abandonment, error recovery,
> changing trust, and whether the observed interaction mechanism persists.

Every future direction should name:

- the unresolved question;
- the proposed design or study;
- the comparison or population;
- the outcome to measure; and
- the result that would support or challenge the current interpretation.

## 6. Common Failure Modes

- **Repeating Results:** Restating every result without interpreting mechanism,
  prior work, or consequence.
- **The victory lap:** Discussing only benefits while moving contradictions and
  failures out of sight.
- **Feature-based implications:** Recommending the prototype's feature list
  instead of a reusable interaction principle.
- **Preference inflation:** Treating liking, intention to use, or novelty as
  proof of learning, performance, trust, or adoption.
- **Speculative generalization:** Jumping from one task or population to a
  broad domain without identifying the shared mechanism.
- **Generic limitations:** Writing "small sample" without explaining which
  claim is affected and why.
- **System-study confusion:** Treating an implementation failure as a sampling
  issue, or a weak study design as something future engineering will fix.
- **Future-work wish list:** Listing attractive features that do not address a
  limitation or theoretical question.
- **Hidden sociotechnical costs:** Ignoring displaced labor, reduced expert
  awareness, bias, privacy, authority, and over-reliance.
- **Future tense as evidence:** Implying that planned improvements strengthen
  claims about the system that was actually evaluated.

## 7. Reusable Writing Templates

### 7.1 Discussion outline

```text
9 Discussion

9.1 Answer to RQ1: [human outcome or practice]
- Main answer and supporting evidence
- Mechanism
- Negative or divergent case
- Relationship to prior work
- Bounded interpretation

9.2 Answer to RQ2: [interaction mechanism or tradeoff]
- What the system changed
- Why the change helped or failed
- Alternative explanation
- Design implication

9.3 Broader design implications
- Implication 1: evidence -> principle -> scope
- Implication 2: evidence -> principle -> scope
- Implication 3: evidence -> principle -> scope

9.4 Human roles, ethics, and deployment
- Authority and responsibility
- Uncertainty and contestability
- Workflow and collaboration
- Safety, privacy, inclusion, and bias
```

### 7.2 Limitations outline

```text
10 Limitations and Future Work

10.1 Study limitations
- participants and missing stakeholders
- task, comparison, setting, and duration
- measures, missing data, and researcher influence

10.2 System limitations
- sensing/model accuracy, latency, and unsupported cases
- interaction, accessibility, interoperability, and scale
- uncertainty, safety, privacy, and bias

10.3 Claim boundary
- what the evidence supports
- what it does not support

10.4 Evidence-driven next studies
- technical validation
- controlled comparison
- longitudinal or field deployment
- expert and affected-stakeholder evaluation
```

### 7.3 Discussion paragraph template

```text
Finding:
We found [pattern], including [conflicting or negative evidence].

Interpretation:
This suggests [mechanism], because [behavioral or qualitative evidence].

Prior work:
The result [supports/refines/challenges] prior work on [concept] by showing
[boundary or extension].

Implication:
Systems for [bounded context] should [principle], while preserving [tradeoff].

Boundary:
This interpretation remains limited to [tested conditions].
```

### 7.4 Limitation paragraph template

```text
The study/system was limited by [specific boundary]. This matters because
[mechanism or consequence], so the evidence does not establish [larger claim].
It does support [narrower claim]. A future [study/design change] should test
[comparison, population, duration, or measure] to determine whether [claim]
extends beyond the current setting.
```

## 8. Review Rubric

Score each criterion as `0 = absent`, `1 = partial`, or `2 = strong`. The score
is diagnostic, not a substitute for reviewer judgment.

| Criterion | Review question |
| --- | --- |
| RQ answer | Does the Discussion directly answer each research question or hypothesis? |
| Evidence discipline | Are claims tied to converging, conflicting, and negative evidence? |
| Mechanism | Does the paper explain why the observed interaction produced the result? |
| Prior work | Does it support, refine, challenge, or bound earlier findings? |
| Tradeoffs | Are costs, tensions, and alternative explanations visible? |
| Transfer | Are implications reusable but explicitly bounded? |
| Human practice | Are roles, authority, collaboration, labor, and affected stakeholders considered? |
| Study limits | Are inference limits explained with consequences? |
| System limits | Are technical and interaction boundaries separated from study design? |
| AI risks | Are uncertainty, error, bias, privacy, safety, and contestability addressed where relevant? |
| Future evidence | Does each major future direction answer a named unresolved question? |
| Claim calibration | Is it clear what the paper supports and does not support? |

## 9. Blueprint for the AI Fencing Coach Paper

The current manuscript already has empty headings for Discussion and
Limitations. The following structure fits the evidence currently reported in
`docs/paper/paper.md`.

### 9.1 How AI Feedback Fits Solo Fencing Practice

The central claim should be:

> AI Fencing Coach showed preliminary value as a diagnostic and reflection aid
> that can make some fencing errors more concrete during solo practice. The
> current evidence does not show that it improves fencing skill.

Build the subsection around these points:

- Participants differed in what they could diagnose through unaided video
  self-review. The system's value is therefore added diagnostic structure, not
  the assumption that learners have no self-awareness.
- Brief step-size and center-of-mass cues could support a reported next action,
  while detailed explanation belonged in post-session review.
- The operative mechanism is a feedback loop of **notice -> identify the
  movement referent -> understand -> act -> review**, not action recognition
  alone.
- Delayed cues and action misclassification broke that loop by making otherwise
  plausible advice difficult to attach to a movement.
- High ratings for error awareness, post-review usefulness, and advantage over
  self-review are evidence of perceived early value, not proof of objective
  movement improvement.

### 9.2 Design Implications for AI-Assisted Sports Coaching

Develop four evidence-backed implications:

1. **Ground feedback in a movement episode.** A correction should identify the
   repetition, action, timestamp, or replay segment that produced it. End-to-end
   feedback latency is an interaction-quality measure, not only a performance
   metric.
2. **Layer immediacy and explanation.** Live feedback should contain one
   prioritized next action. Visual and post-session review should preserve the
   evidence, alternatives, and explanation that would overload movement.
3. **Adapt to baseline diagnostic ability.** Measure what a learner can already
   notice, then evaluate which feedback layer adds information. More feedback
   is not automatically more useful.
4. **Design for contestability.** Expose cue-level confidence, support
   dismissal or correction, abstain when action evidence is weak, and preserve
   the learner's and coach's ability to disagree.

Each implication should cite the project finding that produced it and state
which part remains an untested design hypothesis.

### 9.3 Relationship to Coaches and Existing Practice

Position the system as a practice aid when continuous expert attention is
unavailable.

- Coaches remain necessary to validate technical correctness, prioritize errors
  by learner and drill, monitor safety, and interpret movement that a single
  camera cannot capture.
- The system can contribute consistency, session memory, repeated-error
  tracking, and low-level cues without claiming the contextual judgment of a
  coach.
- A coach-facing design should allow review, correction, configuration of cue
  priorities, and inspection of uncertain detections.
- The Discussion should address whether automated feedback changes what
  learners ask coaches, what coaches can observe, and whether system confidence
  could create inappropriate reliance.

### 9.4 Study Limitations

State the consequence of each current boundary:

- Four participants contributed ratings, but detailed interviews were retained
  for only two. This supports feasibility observations and design hypotheses,
  not prevalence or population estimates.
- Participant demographics, recruitment, fencing experience, compensation, and
  some ethics records are missing from the retained artifacts. The sample
  cannot be characterized as representative of a fencing skill level.
- The fixed condition order confounds system exposure with practice, fatigue,
  and learning. The study cannot establish superiority to self-review.
- Researcher explanations were interleaved with at least one interview, which
  may have shaped interpretation.
- No coach-blinded movement rating, kinematic ground truth, delayed retention,
  or transfer measure was collected. Reported action on a cue is not evidence
  of correct or durable skill change.
- The sessions were short and supervised. They do not establish independent,
  safe, or sustained use.

### 9.5 System Limitations

- A single commodity camera and 2D pose estimation are sensitive to viewpoint,
  occlusion, clothing, distance, and movements outside the image plane.
- The action recognizer and biomechanical heuristics cover only a bounded set of
  visible actions and errors; they cannot assess complete fencing technique,
  tactics, blade interaction, opponent response, or all safety concerns.
- Delayed classification can produce feedback without a clear movement
  referent.
- An incorrect action label can make a reasonable correction misleading.
- Cue-level uncertainty, abstention, learner correction, and coach override are
  not yet fully implemented.
- Optional language-model summaries may improve phrasing but cannot repair
  incorrect sensor evidence. They require grounding, output auditing, privacy
  controls, and a deterministic fallback.

### 9.6 Claim Boundary

| The current evidence supports | The current evidence does not support |
| --- | --- |
| The prototype can deliver live and post-session fencing feedback in a short supervised workflow | The system improves fencing skill |
| Participants perceived value in error awareness and review | The system is superior to video self-review |
| Some brief cues were understandable enough to prompt a reported adjustment | The adjustment was biomechanically correct |
| Timing and classification errors are consequential interaction failures | The error rates are acceptable in unsupervised practice |
| Temporal grounding, layered feedback, and contestability are justified design requirements | The design requirements are validated across fencers, coaches, settings, or long-term use |
| The artifact is a plausible practice and reflection aid | The artifact can replace a coach or act as an independent authority |

### 9.7 Evidence-Driven Future Work

1. **Technical validation:** Benchmark action and error detection across skill
   levels, body types, handedness, camera angles, lighting, clothing, distance,
   and movement speed. Report confusion matrices, cue-level precision, false
   feedback, latency distributions, calibration, and abstention performance.
2. **Controlled interaction study:** Compare self-review with AI-assisted
   review using counterbalanced conditions, equivalent drills, baseline
   diagnostic ability, cue-referent identification, next-attempt correction,
   and coach-blinded movement ratings.
3. **Learning study:** Add delayed retention and transfer to an unpracticed
   drill before making a skill-learning claim.
4. **Field deployment:** Study multi-week use in actual practice, including
   feature abandonment, changing trust, incorrect-feedback recovery, coach
   overrides, safety incidents, and effects on coach-learner communication.
5. **Contestability evaluation:** Test whether confidence, abstention, replay,
   correction, and dismissal controls improve appropriate reliance rather than
   merely increasing trust.

## 10. Final Submission Checklist

- The Discussion answers every research question before offering broader
  implications.
- Favorable, null, contrary, and unexpected findings are all accounted for.
- Each major interpretation names an interaction or sociotechnical mechanism.
- Prior work is used to support, refine, challenge, or bound the findings.
- Design implications are traceable to evidence and specify their scope.
- Human expertise, authority, collaboration, and displaced work are discussed.
- Study and system limitations are separated.
- Each limitation explains which claim it affects.
- AI errors, uncertainty, bias, privacy, safety, and contestability are covered
  where relevant.
- Future work addresses named uncertainty rather than listing desired features.
- The final paragraph states what the evidence supports and what it does not.

## Sources

- Official award verification:
  [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers),
  [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers),
  and
  [CHI 2026 Best Papers](https://programs.sigchi.org/chi/2026/awards/best-papers).
- Full paper records and DOIs are linked in the corpus table.
- The supplied local copies of `MR.Drum.pdf` and `RoomDreaming.pdf` were treated
  as required primary sources.
