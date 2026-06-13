# What Makes a Good HCI Paper Introduction?

## Purpose

This guide defines a strong HCI paper introduction based on a close reading of ten recent CHI papers. It focuses on one central standard:

> A strong introduction establishes a consequential human problem, explains precisely why current practice and key prior work do not resolve it, and makes the paper's response and contribution easy to understand.

The introduction is not a compressed Related Work section or a long description of the system. It is a compact research argument.

## Corpus and Scope

This review includes eight papers officially listed as CHI Best Papers in 2024 or 2025, plus the two CHI papers supplied for this review.

Important accuracy note: as of June 13, 2026, the official CHI 2024 and CHI 2025 award pages do not list **MR.Drum** or **RoomDreaming** as Best Papers. They are included because they were explicitly requested, but they should not be described as Best Paper Award recipients.

| Paper | Year | Status in this review | Introduction strategy |
| --- | --- | --- | --- |
| [RoomDreaming: Generative-AI Approach to Facilitating Iterative, Preliminary Interior Design Exploration](https://doi.org/10.1145/3613904.3642901) | 2024 | Requested CHI paper | Explains the real interior-design workflow, its time and iteration constraints, and how existing CAD and AI tools only partially support exploration. |
| [Understanding Feedback in Rhythmic Gymnastics Training](https://doi.org/10.1145/3613904.3642434) | 2024 | Best Paper | Establishes feedback as essential, shows that technology lacks an understanding of coaching nuances, and motivates an empirical study before system design. |
| [The Metacognitive Demands and Opportunities of Generative AI](https://doi.org/10.1145/3613904.3642902) | 2024 | Best Paper | Connects known GenAI usability problems to the absence of a coherent cognitive account, then introduces metacognition as a productive theoretical lens. |
| [Generative Echo Chamber? Effect of LLM-Powered Search Systems on Diverse Information Seeking](https://doi.org/10.1145/3613904.3642459) | 2024 | Best Paper | Starts with societal stakes, explains known selective-exposure mechanisms, then isolates what is newly uncertain about LLM-powered search. |
| [Time-Turner: A Bichronous Learning Environment to Support Positive In-class Multitasking of Online Learners](https://doi.org/10.1145/3613904.3641985) | 2024 | Best Paper | Reframes multitasking as a persistent and sometimes strategic practice, motivating support for recovery rather than simply trying to eliminate it. |
| [MR.Drum: Designing Mixed Reality Interfaces to Support Structured Learning Micro-Progression in Drumming](https://doi.org/10.1145/3706598.3714156) | 2025 | Requested CHI paper | Connects the distinctive coordination difficulty of drumming to teaching practice, then shows that prior music-learning systems omit structured micro-progression. |
| [Amuse: Human-AI Collaborative Songwriting with Multimodal Inspirations](https://doi.org/10.1145/3706598.3713818) | 2025 | Best Paper | Compares two close technical traditions and shows that neither combines multimodal inspiration with reusable, iterative musical material. |
| [PAIGE: Examining Learning Outcomes and Experiences with Personalized AI-Generated Educational Podcasts](https://doi.org/10.1145/3706598.3713460) | 2025 | Best Paper | Contrasts students' engagement with podcasts and textbooks, identifies the production barrier, and motivates real-time AI conversion and personalization. |
| [Prototyping with Prompts: Emerging Approaches and Challenges in Generative AI Design for Collaborative Software Teams](https://doi.org/10.1145/3706598.3713166) | 2025 | Best Paper | Explains how prompt-based prototyping changes multidisciplinary work, then identifies missing knowledge about collaboration, iteration, and evaluation. |
| [Supporting Co-Adaptive Machine Teaching through Human Concept Learning and Cognitive Theories](https://doi.org/10.1145/3706598.3713708) | 2025 | Best Paper | Shows that interactive ML optimizes model learning while under-supporting human learning, then derives a system from two cognitive theories. |

Official award records:

- [CHI 2024 Best Papers](https://programs.sigchi.org/chi/2024/awards/best-papers)
- [CHI 2025 Best Papers](https://programs.sigchi.org/chi/2025/awards/best-papers)

## Core Definition

A good HCI introduction is a sequence of justified claims:

1. **A human activity, need, or risk matters.**
2. **A specific group encounters a specific difficulty in a specific context.**
3. **Current practices or workarounds leave consequential limitations.**
4. **Key prior research addresses parts of the problem but leaves a precise opening.**
5. **The paper takes an approach that logically follows from that opening.**
6. **The study provides evidence appropriate to the claimed contribution.**
7. **The contribution advances HCI knowledge, design, method, or theory beyond the artifact alone.**

Each step should make the next step feel necessary. If the system appears before the reader understands the human problem and research opening, the introduction is probably technology-led rather than HCI-led.

## The Recurring Structure

The ten papers use different styles, but most can be understood through the following eight-part structure.

### 1. Begin with the human practice or consequence

Start with what people are trying to do and why it matters. Name the activity, users, and stakes.

Strong openings in the corpus focus on:

- mastering coordinated movement;
- receiving effective coaching feedback;
- exploring a design space;
- learning while managing competing demands;
- encountering diverse information;
- creating music from inspiration;
- learning from educational materials; or
- collaboratively designing AI products.

Do not begin with a generic statement such as "AI is developing rapidly." Technology is context, not motivation.

### 2. Narrow to a concrete breakdown

Describe the moment where the current experience fails. A useful problem statement specifies:

- **who** experiences the problem;
- **what** they are trying to accomplish;
- **when and where** the breakdown occurs;
- **what they currently do instead**; and
- **what the consequence is**.

For example, the relevant problem is not merely "drumming is hard." It is that novices must coordinate independent rhythms across limbs, while common learning tools do not break this coordination into the progression used by instructors.

### 3. Establish importance with evidence

Motivation becomes credible when supported by evidence rather than adjectives.

Useful evidence includes:

- prior empirical findings;
- prevalence or adoption data;
- observed professional workflows;
- formative interviews or fieldwork;
- time, cost, error, or access constraints;
- known cognitive, social, or learning consequences; and
- concrete limitations of current tools and workarounds.

Use words such as "important," "critical," and "challenging" only after showing why.

### 4. Organize key prior work into a small comparison

The introduction should compare the closest work, not summarize an entire field. Strong papers often organize prior work into two or three capability clusters.

Examples from the corpus include:

- systems that generate finished music versus systems that provide editable musical elements;
- fencing or sports analysis systems versus learner-facing feedback systems;
- CAD tools for individual design elements versus generative systems for complete images;
- interactive ML methods that improve models versus interfaces that also support human concept learning; and
- conventional search research versus unresolved interaction effects of LLM-powered search.

This comparison gives the reader a map of what is already possible and where the paper sits.

### 5. State the gap as a precise mismatch

A strong gap is rarely "nobody has studied X." It is usually a mismatch between an important requirement and what current practice or prior work provides.

Common gap forms are:

- **Experience gap:** a capability exists, but the user experience is delayed, difficult, fragmented, or not actionable.
- **Context gap:** an approach works elsewhere but is not grounded in this population, setting, or practice.
- **Capability gap:** existing tools support A or B, but not the combination required by the activity.
- **Knowledge gap:** technology is being proposed without sufficient understanding of the underlying human practice.
- **Theory gap:** known problems lack a coherent explanatory framework.
- **Evidence gap:** a promising design direction lacks evaluation of the outcome that matters.

A useful gap sentence has this structure:

> Prior work supports **A** and adjacent work demonstrates **B**; however, neither resolves **C** for **users in context D**, where **consequence E** matters.

### 6. Present the response and explain why it fits

Introduce the research response only after the opening is clear. State:

- what was studied, designed, or built;
- why this approach follows from the gap;
- the main research questions or design requirements; and
- the scope of the claim.

The best introductions do not merely announce a system. They connect design choices to the earlier argument. MR.Drum derives micro-progression from instructor practice. Mocha derives interactions from cognitive theories. Time-Turner derives recovery and monitoring requirements from how students actually multitask.

### 7. Preview the evidence and principal result

Tell the reader how the claim was investigated:

- formative study;
- fieldwork or interviews;
- controlled experiment;
- deployment;
- design study;
- comparative evaluation; or
- theoretical synthesis.

Include sample sizes and conditions when they help establish the scale and type of evidence. Summarize only the results needed to show that the paper answers its motivating question.

### 8. End with contributions that answer the gap

Contribution bullets should complete the argument introduced earlier. They should not be a list of project activities.

Strong contribution categories include:

- new empirical knowledge about a human practice;
- a design framework or set of implications;
- an interactive system embodying those insights;
- a technical method required to produce the interaction;
- empirical evidence about user experience or outcomes; and
- a theoretical framing that changes how a problem can be understood.

"We built a system" is usually insufficient by itself. State what HCI can know, design, or evaluate differently because the system was built and studied.

## How to Build Strong Motivation

Strong motivation is not excitement about a technology. It is a defensible account of why a human problem deserves research attention.

### The five-part motivation test

Before drafting, answer these questions in one or two sentences each:

1. **Who is affected?** Avoid "users" when a more specific group is known.
2. **What are they trying to accomplish?** Describe the activity, not the interface.
3. **Where does the current practice break down?** Name the moment or workflow.
4. **What is the consequence?** Explain the effect on learning, work, agency, safety, access, creativity, or wellbeing.
5. **Why is HCI needed?** Explain why the issue concerns interaction, experience, collaboration, interpretation, or design rather than only model performance.

### Motivation should include current workarounds

Existing workarounds prove that people already expend effort on the problem. They also provide realistic comparison points.

For an HCI system, ask:

- What do people do today?
- What is useful about that practice?
- Where is it delayed, expensive, inaccessible, cognitively demanding, or unreliable?
- Which parts should technology preserve rather than replace?

This prevents the paper from inventing a problem around a new technology.

### Motivation should explain why now

"Why now" can come from:

- a new technical capability;
- a newly widespread practice;
- a change in policy or infrastructure;
- an emerging risk;
- a newly accessible population or setting; or
- evidence that an old design assumption no longer fits.

The introduction should connect the new opportunity to the human need without implying that novelty alone creates value.

## How to Compare Key Prior Work

The goal is not to defeat prior work. The goal is to identify the exact research opening.

### Compare on dimensions that matter to the contribution

Useful dimensions include:

| Dimension | Questions |
| --- | --- |
| User and setting | Is the work designed for the same population and real practice context? |
| Goal | Does it recognize, describe, explain, teach, support reflection, or change action? |
| Timing | Is support real-time, post-hoc, asynchronous, or detached from the relevant event? |
| Input and hardware | Does it require specialized sensing, curated data, or commodity devices? |
| Output | Does it return a label, metric, visualization, recommendation, or actionable instruction? |
| Adaptation | Does it respond to skill, preference, context, history, or changing user understanding? |
| Human role | Is the person a data source, operator, learner, collaborator, decision-maker, or expert? |
| Evidence | Was the work technically evaluated, studied with users, deployed, or validated in practice? |

Select only the dimensions that are necessary to justify the paper.

### Use the two-neighbor comparison

Most HCI contributions should compare against at least two kinds of neighbor:

1. **Same or similar problem, different approach.**
2. **Similar approach, different problem or context.**

For AI-assisted fencing coaching:

- fencing recognition and tactical-analysis papers address the domain but often stop before learner-facing coaching;
- pose-based coaching papers address actionable feedback but are not grounded in fencing technique and practice.

The paper's opening exists at the intersection of those two limitations.

### Write bounded comparison claims

Prefer:

> Existing fencing systems demonstrate action recognition and tactical analysis, but their primary output supports classification or expert interpretation rather than immediate learner-facing correction.

Avoid:

> Existing fencing systems do not help users.

The first claim is specific, testable, and respectful. The second is broad and likely false.

### Separate academic evidence from products

Commercial tools can demonstrate current workflows, expectations, and alternatives. They do not replace academic evidence about learning or behavior.

Use products to explain:

- what people can already buy or use;
- common interaction patterns;
- setup and hardware expectations; and
- practical comparison baselines.

Use research literature to justify:

- empirical effects;
- theoretical claims;
- design knowledge; and
- the scholarly gap.

## A Drafting Template

The following template is a scaffold, not a requirement to force every paper into identical paragraphs.

### Paragraph 1: Human activity and stakes

> **[Population]** rely on **[practice/resource]** to accomplish **[important goal]**. During **[specific context]**, they must **[difficult action or decision]**. When **[breakdown]** occurs, **[meaningful consequence]** follows.

### Paragraph 2: Current practice and workarounds

> People currently use **[workaround A]**, **[workaround B]**, and **[workaround C]**. These practices provide **[real value]**, but are limited by **[timing, access, specificity, effort, reliability, or other constraint]**.

### Paragraph 3: Closest prior work

> Prior **[domain-specific]** research demonstrates **[capability A]**. In parallel, **[approach-specific HCI]** research shows **[capability B]**. Commercial or professional tools support **[workflow C]**.

### Paragraph 4: Research opening

> However, these approaches stop short in different ways: **[A lacks requirement X]**, while **[B lacks context Y]**. Consequently, we still lack **[precise missing knowledge or experience]** for **[population and setting]**.

### Paragraph 5: Research response

> We investigate **[research question]** through **[system/study/theory]**. Our approach uses **[key mechanism]** because **[connection to earlier evidence or practice]**.

### Paragraph 6: Method and evidence

> We conducted **[formative work]**, developed **[artifact/framework]**, and evaluated it through **[study design and participants]**. We found **[principal result, with careful scope]**.

### Paragraph 7: Contributions

> This work contributes **[empirical knowledge]**, **[design/system/method]**, and **[validated implications or theoretical advance]**.

## Common Failure Modes

### Technology-first motivation

Weak:

> Recent advances in LLMs create many opportunities for sports.

Stronger:

> During independent practice, fencers can repeat timing and distance errors without recognizing them at the moment of performance; current feedback is often delayed or dependent on coach availability.

### A gap defined only by absence

Weak:

> No prior system combines AI and fencing.

Stronger:

> Fencing research supports recognition and tactical analysis, while adjacent sports systems support learner-facing feedback; neither establishes how fencing-specific, timely feedback should fit into solo practice.

### Listing prior work without comparing it

A citation inventory does not establish a gap. Group papers by what they enable, then compare those capabilities with the requirements of the motivating practice.

### Overclaiming novelty

Avoid "first," "never," and "no work" unless supported by a credible search. A contribution can be important because it combines requirements, reframes a problem, studies a neglected context, or supplies stronger evidence.

### Confusing model output with an HCI contribution

Classification accuracy may enable the interaction, but the introduction must explain why the output is understandable, timely, actionable, trustworthy, or compatible with real practice.

### Claiming effectiveness before the evidence permits it

Match the verb to the study:

- an interview study can **characterize**, **identify**, or **reveal**;
- a controlled study can **compare** effects under tested conditions;
- a deployment can **examine use over time**;
- a prototype study can **demonstrate feasibility**;
- only suitable learning or performance evidence should support **improves skill**.

### Contributions that repeat the method

Weak:

> We conducted interviews, built a system, and ran a study.

Stronger:

> We contribute an account of how feedback timing and specificity shape solo practice, a coaching interface derived from those requirements, and evidence about whether learners can act on its feedback.

## Introduction Review Checklist

### Motivation

- [ ] The opening names a human activity, population, and consequence.
- [ ] The problem is specific to a context or moment, not merely a broad topic.
- [ ] Claims of importance are supported by evidence.
- [ ] Current practices and workarounds are described fairly.
- [ ] The paper explains why this is an HCI problem.
- [ ] The introduction explains why the question is timely.

### Prior work and positioning

- [ ] The closest prior work appears in the introduction.
- [ ] Prior work is grouped by capability or approach rather than listed.
- [ ] The comparison uses dimensions relevant to the claimed contribution.
- [ ] The gap is a precise mismatch, not an unsupported absence claim.
- [ ] The introduction includes both same-problem and same-approach neighbors.
- [ ] Limitations of prior work are stated narrowly and respectfully.
- [ ] Products and academic findings are not treated as equivalent evidence.

### Research response

- [ ] The proposed study or system logically follows from the gap.
- [ ] Research questions or design requirements are explicit.
- [ ] The method and evidence type are previewed.
- [ ] Main findings are summarized without exceeding the study's scope.
- [ ] Contributions state reusable knowledge, not only project activities.
- [ ] The paper clearly states what it is not claiming when confusion is likely.

### Argument quality

- [ ] Every paragraph advances the same central argument.
- [ ] Important terms are defined or made concrete.
- [ ] The system is introduced only after the problem and opening are clear.
- [ ] The gap sentence could be understood without reading the entire Related Work section.
- [ ] The contribution bullets answer the motivation established at the beginning.

## Application to the AI Fencing Coach Paper

The current project already has the right high-level direction: frame the contribution as an improvement to the feedback experience of solo fencing practice, not as "AI for fencing" or as action recognition alone.

A strong introduction should use this argument:

1. **Practice and stakes:** beginner and intermediate fencers need timely correction to avoid repeating weak footwork, recovery, spacing, or timing.
2. **Breakdown:** coaches cannot continuously observe every repetition, especially in solo or lightly supervised practice.
3. **Current workarounds:** mirrors, self-video, teammate comments, and delayed coach questions help, but differ in immediacy, expertise, and alignment with the exact error.
4. **Domain-specific prior work:** fencing systems recognize, segment, or visualize actions and tactics, establishing technical feasibility and domain knowledge.
5. **Approach-specific prior work:** sports-HCI systems show that pose-based and AI-supported feedback can support movement learning.
6. **Opening:** domain-specific work is less learner-facing, while learner-facing work is less fencing-specific. The unresolved question is how to make feedback timely, understandable, and actionable within real fencing practice.
7. **Response:** investigate a commodity-camera coaching system whose feedback is grounded in fencing concepts and validated with learners and coaches.
8. **Evidence:** evaluate whether users notice the relevant error, understand the explanation, and change the next action. Model accuracy is supporting evidence, not the final HCI outcome.

Suggested gap paragraph:

> Prior computing research in fencing demonstrates that footwork and tactical patterns can be recognized, segmented, and visualized. In parallel, HCI systems for movement learning show that pose-based interfaces and automated feedback can support self-correction. These lines of work stop short in complementary ways: fencing-specific systems primarily support recognition or expert analysis, whereas learner-facing coaching systems are not grounded in fencing technique, vocabulary, and practice structure. We therefore lack evidence about how commodity-camera AI feedback can become timely, understandable, and actionable during solo fencing practice.

Suggested scope statement:

> Our goal is not to replace coaches or to treat action recognition as the contribution by itself. We investigate how automated analysis can support the practice moments in which expert feedback is unavailable, and how its feedback can prepare learners for more focused reflection and coach-student discussion.

## Final Standard

The best test of an HCI introduction is whether a reader can answer these questions before reaching Related Work:

1. What human problem matters?
2. Who experiences it, and in what context?
3. What do people do today?
4. Why are those practices insufficient?
5. What has the closest research already achieved?
6. What precise opening remains?
7. Why does this paper's approach fit that opening?
8. What evidence and contribution will the paper provide?

If these answers are clear, well-evidenced, and logically connected, the introduction is doing its job.
