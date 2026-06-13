# What Makes a Good HCI Related Work Section?

Status: Research synthesis  
Last updated: 2026-06-13  
Scope: Ten CHI 2024-2025 papers, including `MR.Drum` and `RoomDreaming`

## Executive Definition

A good HCI Related Work section is a **selective, evidence-based argument** that explains:

1. which intellectual, empirical, design, and technical conversations the paper belongs to;
2. what those conversations already establish;
3. where their boundaries, tensions, or unresolved questions lie;
4. why the present research question and approach are necessary; and
5. how prior work informs the paper's design, variables, method, and claimed contribution.

It is not an annotated bibliography, a chronological history, or a collection of papers that end with "however, none did exactly what we did." Its purpose is to make the paper's research logic legible.

The shortest practical definition is:

> **Related Work should turn a body of literature into the argument that makes the current study necessary and appropriately designed.**

## Corpus and Method

"Recent years" was operationalized as CHI 2024 and CHI 2025. The corpus contains eight papers listed on the official CHI Best Paper Award pages, plus the two requested CHI papers.

Important scope correction: `MR.Drum` and `RoomDreaming` are CHI full papers, but they do not appear on the official [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers) or [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers) lists. They are therefore treated as required comparison papers, not mislabeled as award winners.

The analysis covered each paper's abstract, introduction and contribution framing, Related Work or equivalent Background sections, and the connection between that literature and the later method or discussion. Word counts are approximate because they were extracted from PDF text.

| Paper | Status | Contribution pattern | Related Work structure | Approx. words |
|---|---|---|---|---:|
| [MR.Drum](https://doi.org/10.1145/3706598.3714156) | Required CHI 2025 paper | Formative studies, learning framework, MR system, comparative study | Learning progression; MR instrument learning | 717 |
| [RoomDreaming](https://doi.org/10.1145/3613904.3642901) | Required CHI 2024 paper | Generative-AI design system and iterative studies | Commercial GenAI tools; CAD; generative design | 1,038 |
| [Code Shaping](https://doi.org/10.1145/3706598.3713822) | CHI 2025 Best Paper | New interaction concept developed through staged studies | Sketch-to-code; annotation and code planning | 1,076 |
| [PAIGE](https://doi.org/10.1145/3706598.3713460) | CHI 2025 Best Paper | Personalized educational content system and experiment | Personalized learning; AI-generated education | 811 |
| [Cripping the Co-Design of Pacing Technologies](https://doi.org/10.1145/3706598.3713990) | CHI 2025 Best Paper | Critical participatory design and methodological contribution | Energy-limiting conditions; crip methodologies | 1,888 |
| [Lost in Magnitudes](https://doi.org/10.1145/3706598.3713487) | CHI 2025 Best Paper | Visualization design space and empirical evaluation | Domain definitions; prior designs; perception theory | 1,938 |
| [Constrained Highlighting](https://doi.org/10.1145/3613904.3642314) | CHI 2024 Best Paper | Theory-grounded controlled experiment | Benefits; pitfalls; HCI applications | 752 |
| [Debate Chatbots](https://doi.org/10.1145/3613904.3642513) | CHI 2024 Best Paper | Social-agent design and factorial experiment | Social identity; critical thinking; supporting systems | 1,681 |
| [A Taxonomy of AI Privacy Risks](https://doi.org/10.1145/3613904.3642116) | CHI 2024 Best Paper | Taxonomy grounded in documented harms | Human-centered AI; prior taxonomies; taxonomy construction | 1,377 |
| [Computing and the Stigmatized](https://doi.org/10.1145/3613904.3642005) | CHI 2024 Best Paper | Critical ethnography and contextual HCI contribution | Computing; geography; spatial theory; legal and local context | 2,005 |

Across this sample, Related Work occupies approximately 6% to 12% of a paper. There is no ideal length independent of the contribution. Narrow experiments can establish their mechanism concisely; critical, theoretical, or design-space papers need more context because the literature also defines their objects, values, and analytical lenses.

## What the Ten Papers Show

### 1. Organize Literature by the Claims the Paper Must Establish

The strongest subsection structure is not simply "topics related to our system." Each strand has a job.

- `Constrained Highlighting` establishes why highlighting can help, why it can fail, and why a user-interface constraint could change the cognitive mechanism.
- `Debate Chatbots` separately grounds social identity, critical thinking, rhetorical style, conversational agents, and social-media interventions because those concepts become experimental factors.
- `Lost in Magnitudes` defines the mathematical object, reviews existing encodings, and introduces perception theory because all three are needed to construct and evaluate its design space.
- `Computing and the Stigmatized` needs computing research, Global South scholarship, spatial theory, law, and local history to make its ethnography interpretable without universalizing a Western account.

A useful test is: **If a subsection were removed, which research question, design choice, variable, method, or interpretation would become unsupported?** If the answer is "none," that subsection may not belong.

### 2. Synthesize by Dimensions, Not by Paper

Weak writing produces a sequence such as "A did X. B did Y. C did Z." Strong writing groups work by a meaningful comparison:

- kind of personalization;
- interaction modality;
- stage of a workflow;
- user role and agency;
- theory or mechanism;
- context and population;
- input, output, and feedback;
- evaluation task and outcome;
- assumptions, tradeoffs, or values.

`RoomDreaming` is strongest when it compares tools by whether users can state preferences, iterate from prior outputs, explore alternatives, and control divergence. `Code Shaping` distinguishes sketches of program output from sketches used directly on code. These comparisons expose a consequential interaction difference, rather than only a missing feature.

Tables can help when the dimensions are stable and important. `Lost in Magnitudes` uses a table of marks, channels, ranges, tasks, and evaluations as evidence for an incompletely explored design space.

### 3. Define Concepts Before Using Them as Contributions

HCI combines theories and evidence from psychology, education, design, sociology, disability studies, law, visualization, and computing. A good review does not assume that a term has one obvious meaning.

- `Debate Chatbots` defines the form of critical thinking it measures and the rhetorical styles it manipulates.
- `A Taxonomy of AI Privacy Risks` distinguishes legal privacy taxonomies, AI-specific frameworks, perceived risks, and documented harms.
- `Cripping the Co-Design` explains crip theory and then uses it to question productivity, participation, pacing, and even the study's own methods.
- `Computing and the Stigmatized` defines sex work and digital sex work before connecting technology to social and spatial conditions.

Theory is useful only when it changes what the researchers design, measure, ask, or interpret. A decorative theory citation that disappears after Related Work is not conceptual grounding.

### 4. Express the Gap as a Consequence, Not an Empty Space

"No one has combined A, B, and C" is not yet a research gap. The paper must explain why that absence matters.

Strong gaps in the corpus take several forms:

- **Empirical gap:** evidence is absent, limited, or contradictory.
- **Design gap:** existing systems support the wrong interaction loop, stage, user role, or value.
- **Contextual gap:** knowledge from one population or geography cannot safely be generalized.
- **Theoretical gap:** existing concepts do not explain an observed phenomenon.
- **Methodological gap:** current methods exclude participants or cannot reveal the relevant experience.
- **Coverage gap:** a taxonomy or design space omits consequential categories or combinations.

Examples:

- `PAIGE` distinguishes selecting prewritten personalized material from generating new interest-based content, then identifies the missing evidence around generative audio learning.
- `Cripping the Co-Design` argues that existing pacing studies provide design implications without fully involving people with energy-limiting conditions in imagining technologies through accessible crip methods.
- `A Taxonomy of AI Privacy Risks` identifies a divide between broad legal taxonomies and AI-specific frameworks not grounded in documented, realized harms.
- `Computing and the Stigmatized` argues that separating technological and sociological readings produces an incomplete account of physical and digital space.

The gap should end with a consequence: what cannot currently be designed, explained, evaluated, or decided?

### 5. Make Related Work Predict the Method

After reading Related Work, a reader should understand why the study takes its particular shape.

- Psychological evidence about selectivity and self-regulation leads `Constrained Highlighting` to test a highlight limit.
- Social identity and rhetorical-style research lead `Debate Chatbots` to manipulate chatbot identity and dialogue style.
- Graphical perception research gives `Lost in Magnitudes` its encoding dimensions and evaluation tasks.
- Crip theory leads `Cripping the Co-Design` to use an asynchronous remote community and to evaluate the cost and accessibility of participation.
- Professional teaching practice and instrument-learning systems lead `MR.Drum` toward micro-progression and first-person mixed-reality guidance.

If the method could have been designed without the literature presented, the review is probably descriptive rather than generative.

### 6. Calibrate Novelty Claims

Claims such as "the first," "no prior work," and "has not been explored" are fragile because they require near-exhaustive evidence.

Prefer bounded language:

- "We found no prior HCI system that..."
- "Prior work has primarily focused on..."
- "Existing studies establish X, but evidence remains limited for Y..."
- "Within the literature on [defined scope], we extend..."
- "Unlike the systems reviewed above, our work..."

`MR.Drum` makes a clear distinction between whole-piece support and structured rhythm/limb micro-progression, but its "first" claim would be safer if explicitly bounded to the reviewed interactive music-learning literature. Novelty is more credible when expressed as a precise difference and consequence than as absolute priority.

### 7. Treat Products, Scholarship, and Theory as Different Evidence

Commercial products show the current design landscape, but they are not substitutes for peer-reviewed evidence about behavior, learning, effectiveness, or social impact.

`RoomDreaming` usefully includes commercial GenAI tools because the contribution concerns a fast-changing product workflow. A rigorous version of this move should:

- separate products from research systems;
- record the access date because features change;
- compare observable capabilities without inferring untested user outcomes;
- use scholarship or original studies for theoretical and empirical claims.

Similarly, textbooks, standards, policy, law, and community terminology can be essential sources, but the paper should state the role each type of source plays.

### 8. Give the Reader an Explicit Route Through the Argument

The reader should not have to infer why a subsection exists. Effective sections use short roadmaps and synthesis sentences:

- what this subsection reviews;
- why it matters to the paper;
- what the literature collectively shows;
- what remains unresolved;
- how the paper responds.

`Debate Chatbots` does this particularly clearly by stating the purpose of each major subsection before reviewing its literature. Repetition is acceptable when it keeps a long, interdisciplinary argument navigable.

## Four Related Work Archetypes

Different HCI contributions need different review structures.

### System or Interaction Technique Paper

Examples: `MR.Drum`, `RoomDreaming`, `Code Shaping`.

The review should establish:

1. the human activity and its current practice;
2. existing interactive approaches;
3. the dimensions on which those approaches differ;
4. the consequential limitation in the current interaction or workflow; and
5. how the proposed system changes that relationship.

The contribution should not be framed as a feature bundle. Explain the new capability, interaction loop, division of labor, or form of user agency.

### Controlled Experiment or Behavioral Study

Examples: `Constrained Highlighting`, `PAIGE`, `Debate Chatbots`.

The review should establish:

1. the underlying cognitive, social, or learning mechanism;
2. what prior evidence predicts, including conflicting results;
3. the intervention or manipulation;
4. the dependent outcomes and why they matter; and
5. the hypotheses or research questions.

The literature should make the experimental design feel derived rather than arbitrary.

### Taxonomy, Framework, or Design-Space Paper

Examples: `A Taxonomy of AI Privacy Risks`, `Lost in Magnitudes`.

The review should establish:

1. definitions and boundaries;
2. prior classification schemes or design alternatives;
3. explicit comparison dimensions;
4. missing categories, combinations, evidence, or users; and
5. the intended analytical or practical use of the new framework.

The new structure must do more than rename existing categories. It should enable analysis, design, evaluation, or decision-making that was previously difficult.

### Critical, Qualitative, or Participatory Paper

Examples: `Computing and the Stigmatized`, `Cripping the Co-Design`.

The review should establish:

1. historical and sociotechnical context;
2. whose knowledge dominates the existing literature;
3. population, geography, institutions, law, and power;
4. the theoretical lens and its consequences;
5. methodological and ethical implications; and
6. how the work avoids universalizing participants or treating them as a deficit.

In this archetype, context is not background decoration. It is part of the phenomenon and often part of the contribution.

## A Strong Paragraph Pattern

A reliable Related Work paragraph usually performs five moves:

1. **Claim:** State the idea the paragraph will establish.
2. **Synthesis:** Combine several relevant findings or approaches.
3. **Comparison:** Explain a pattern, conflict, boundary, or tradeoff.
4. **Consequence:** State why that pattern matters to the present problem.
5. **Bridge:** Connect it to the paper's question, design, method, or claim.

Template:

> Prior research on **[construct or approach]** shows **[collective finding]**. Studies have examined **[dimension A]** and **[dimension B]**, generally finding **[pattern]**, although evidence differs when **[condition or context]**. These studies establish **[what is known]**, but they leave **[specific consequential uncertainty]** unresolved. We therefore examine/design **[response]**, focusing on **[bounded contribution]**.

Not every paragraph needs all five moves, but each subsection should complete the sequence.

## Recommended Section Template

```text
2 RELATED WORK

Opening roadmap
- Name the two to four literature strands.
- Explain the role each strand plays in the paper.

2.1 Human activity, construct, or domain
- Define the central phenomenon.
- Establish user needs, mechanisms, or current practice.
- Identify relevant tensions or conflicting evidence.

2.2 Interactive and technical approaches
- Group systems by meaningful design dimensions.
- Compare capabilities, assumptions, users, and evaluations.
- Avoid a one-paper-per-sentence catalog.

2.3 Context, theory, or method
- Add the lens needed to interpret the problem responsibly.
- Explain how it changes the research design or analysis.

Closing synthesis
- State what is established.
- State the precise unresolved problem.
- Explain why the proposed research question and approach follow.
```

The headings should reflect the actual argument. Not every paper needs exactly three subsections.

## Common Failure Modes

- **Bibliography dump:** many citations, little comparison or synthesis.
- **Topic matching:** papers are included because they mention the same technology, not because they inform a claim.
- **Feature-gap novelty:** the contribution is framed as an unmotivated combination of features.
- **Universal "however":** every cited paper is reduced to a deficiency.
- **Unsupported absolute novelty:** "first" or "no work" without a bounded search scope.
- **Theory decoration:** concepts appear in Related Work but do not affect the study.
- **Method disconnect:** reviewed evidence does not explain the research questions or method.
- **Context erasure:** findings from one population, culture, or institution are treated as universal.
- **Evidence mixing:** product descriptions, blog posts, theory, and empirical studies are treated as equivalent.
- **Contradiction avoidance:** mixed findings are hidden instead of used to motivate the study.
- **Citation ambiguity:** a citation is attached to a sentence containing several claims without showing which source supports which claim.
- **Gap repetition:** the Introduction and every subsection repeat the same novelty sentence without adding precision.

## Review Rubric

Score each dimension from 0 to 2:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Selection | Broad or arbitrary | Mostly relevant | Every strand supports a necessary claim |
| Synthesis | Paper-by-paper list | Some grouping | Clear patterns, dimensions, and tensions |
| Concepts | Undefined or decorative | Partly defined | Defined and used in design or analysis |
| Gap | Generic absence | Specific but weakly motivated | Specific, consequential, and bounded |
| Method bridge | No connection | Implicit connection | Literature clearly generates the method |
| Context and power | Ignored | Mentioned | Integrated where relevant to claims and ethics |
| Evidence quality | Unclear or mixed | Generally credible | Source roles, recency, and conflicts handled explicitly |
| Navigation | Hard to follow | Understandable | Roadmaps and synthesis make the argument easy to audit |

Interpretation:

- **13-16:** strong submission-ready argument;
- **9-12:** sound foundation, but some strands remain descriptive;
- **5-8:** substantial restructuring needed;
- **0-4:** literature list rather than Related Work analysis.

The score is diagnostic, not a substitute for expert judgment.

## Application to the AI Fencing Coach Paper

For this project, a defensible Related Work section would likely need four strands:

1. **Motor learning and feedback:** what makes feedback actionable, correctly timed, interpretable, and useful for skill acquisition.
2. **Video and pose-based movement analysis:** what can be sensed or recognized, under which camera and data constraints, and with what reliability.
3. **Interactive sports coaching systems:** how systems represent errors, comparisons, progress, uncertainty, and the roles of learner and coach.
4. **Fencing-specific research:** footwork, actions, tactics, assessment, terminology, and domain constraints that generic exercise systems do not capture.

A synthesis matrix could compare prior systems by:

- target sport and skill;
- novice, athlete, or coach user;
- sensing hardware;
- action or technique granularity;
- real-time or post-session feedback;
- visual, verbal, haptic, or multimodal feedback;
- personalization and progression;
- error explanation and uncertainty;
- coach involvement;
- learning, usability, recognition, or performance evaluation.

The central argument should not stop at "few systems study fencing." It should distinguish:

- recognition from coaching;
- detected difference from actionable feedback;
- generic pose correctness from fencing-specific technique and tactics;
- model output from a feedback interaction that supports learning;
- laboratory sensing from realistic training with commodity video.

A calibrated closing synthesis might take this form:

> Fencing research has developed methods for recognizing, segmenting, and visualizing actions, while HCI research in other physical skills has studied feedback representations and movement learning. However, recognition accuracy alone does not establish how video-derived observations should be converted into fencing-specific, interpretable, and actionable coaching under realistic training constraints. This motivates an HCI investigation of how such feedback should be designed, understood, and incorporated into practice by fencers and coaches.

That claim must ultimately be revised to match the actual evidence and contribution of the completed system and study.

## Final Submission Checklist

- Can every subsection be tied to a research question, design choice, method, or contribution?
- Are papers grouped by meaningful dimensions rather than summarized one at a time?
- Are central concepts defined and used later?
- Does the review include contradictory or negative evidence where it exists?
- Is the gap specific, consequential, and bounded?
- Does the gap explain why the chosen method is appropriate?
- Are products, research systems, theory, and empirical evidence distinguished?
- Are context, geography, institutions, accessibility, and power addressed where relevant?
- Are strong novelty claims supported or softened?
- Does each subsection end with a synthesis, not merely its final citation?
- Does the section complement rather than duplicate the Introduction?
- After reading it, can a reviewer predict what the paper will do and why?

## Sources

- [Official CHI 2025 Best Paper Award list](https://programs.sigchi.org/chi/2025/awards/best-papers)
- [Official CHI 2024 Best Paper Award list](https://programs.sigchi.org/chi/2024/awards/best-papers)
- The ten DOI-linked papers listed in the corpus table
- User-provided PDFs: `MR.Drum.pdf` and `RoomDreaming.pdf`
