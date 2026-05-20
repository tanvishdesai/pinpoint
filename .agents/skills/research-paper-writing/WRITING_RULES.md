# Writing rules for research papers

This document contains every rule, test, and anti-pattern relevant to writing
research papers well. It is organized from structural rules (whole-paper level) down
to sentence-level rules. Read once at the start of any paper project. Return to the
relevant section during each revision pass.

---

## 1. Structural rules

### 1.1 Write the story before writing the paper

Before drafting any section, write two paragraphs in plain conversational English —
no LaTeX, no citations, no technical vocabulary. These paragraphs should describe:

Paragraph 1: The problem and why it matters. What happens in the world because this
problem is unsolved? Who is affected?

Paragraph 2: What you did differently, what you found, and what it means. Explain it
as you would to a smart colleague who is not in your specific subfield.

If these two paragraphs feel unclear or hard to write, the contribution framing is not
yet sharp enough to write a paper. Resolve the framing first.

**This is not a section of the paper. These paragraphs are for you only.**

### 1.2 Write the introduction last

The introduction's job is to promise exactly what the paper delivers. If you write the
introduction before the rest of the paper, you are writing promises before knowing
what you can keep. Write the introduction after the method, experiments, and related
work sections are complete. At that point it writes itself: you already have all the
language, claims, and structure you need.

The abstract is a compression of the introduction. Write it after the introduction.

### 1.3 Contributions must be specific, verifiable, and numbered

Every contribution listed in the introduction (or abstract) must satisfy three tests:

**The specificity test:** Does the contribution make a claim specific enough that
someone could verify it? "We propose a novel framework" fails. "We introduce a
three-component synchronization loss that constrains cross-attention maps to
produce diagonal structure in synchronized samples, making explanations architecturally
guaranteed rather than post-hoc" passes.

**The verifiability test:** Does the paper contain a specific result (table row,
figure, ablation) that directly demonstrates this contribution? If not, either the
contribution is not demonstrated or it needs to be removed.

**The match test:** Are the words used in the contribution statement matched by the
same words used in the relevant section of the paper? A contribution that says
"synchronization-aware loss" must have a section named or indexed to that loss. The
reader should be able to look up every contribution statement directly.

Number the contributions. Readers scan for them. Prose contributions buried in
paragraphs are harder to find and harder to remember.

### 1.4 The abstract must contain five things

In order: (1) the problem and its significance, (2) the limitation of existing work
that motivates this paper, (3) what you propose, (4) your key result(s) with numbers,
(5) the significance of the result. An abstract missing any of these five leaves the
reviewer without the information they need to assess the paper's value.

### 1.5 The introduction must not repeat the abstract word for word

The abstract and introduction cover the same ground but at different depths. The
abstract is 150–250 words of pure signal. The introduction is 400–700 words that
motivate the problem more fully, survey the landscape of failure modes in prior work,
and end with the contribution list. They should share themes, not sentences.

### 1.6 The conclusion must not just repeat the abstract

The conclusion adds: (1) a brief reflection on what the results mean beyond the
numbers, (2) limitations stated honestly, (3) specific future directions. A conclusion
that is a paragraph-for-paragraph restatement of the abstract wastes a section.

---

## 2. Section-level rules

### 2.1 Related work

**Organize by research direction, not by paper.**
Do not write one paragraph per paper. Write one paragraph per direction or theme,
synthesizing multiple papers. The reader is building a map of the field, not reading
a list.

**End every subsection with a gap statement.**
Each subsection should close with one or two sentences naming what the papers in this
direction collectively fail to address. That gap is what makes your paper necessary.
If you cannot write a gap statement, the subsection does not belong in your related
work.

**Summarize, do not just describe.**
For each paper, say: what they do, what they achieve, and what limits them — in
2–3 sentences. Do not summarize only the method without naming the limitation. Do not
name the limitation without identifying the method that causes it.

**Avoid the laundry list.**
Do not write: "A did X. B did Y. C did Z. D did W." This is a list of facts, not a
literature review. Connect adjacent papers: "While A and B address X with CNN-based
approaches, neither accounts for Y, a limitation shared by C's later work despite its
improved Z."

**Cite the right number of papers per claim.**
One claim, one or two citations. Do not stack five citations on a single sentence
unless the sentence is genuinely a field-wide consensus claim. Over-citing makes the
writing feel insecure.

### 2.2 Methodology

**One idea, one subsection.**
Each subsection of the methodology introduces exactly one component of your approach.
If a subsection contains two ideas that cannot be explained without explaining each
other, try to split them and add a brief "these two components interact as follows"
bridge paragraph.

**Explain the why before the what.**
Before describing a component mechanically, explain why it is needed. "Standard cross-
attention concatenates features without dynamic filtering. This causes noisy fusion
when one modality contains irrelevant information. To address this, we introduce a
gating layer that..." makes the design choice legible. "We introduce a gating layer
that computes per-dimension relevance scores..." does not.

**Every hyperparameter must appear somewhere.**
The methodology section or an accompanying implementation details paragraph must
contain every hyperparameter value. If a reviewer tried to reproduce your work from
the paper alone, they should not need to email you for a learning rate.

**Equations must be explained in words.**
Every equation must be followed by a sentence that explains it in English. Define every
symbol the first time it appears. The explanation is not optional for complex equations
— it is the sentence that tells the reader what the equation is computing, not just
how.

### 2.3 Experiments

**Lead with the main comparison, not setup.**
The experimental setup (datasets, metrics, baselines) must be described, but
concisely. Readers want results. Put setup details in a subsection that can be skimmed
and put the main results table first or second.

**Every table needs a takeaway sentence.**
After every table, write one sentence that tells the reader what to conclude from it.
Do not write "Table 2 shows the results." Write "Table 2 shows that removing the
synchronization loss reduces accuracy by 15%, confirming that temporal alignment
supervision is the primary contributor to performance."

**Ablation studies must isolate one variable at a time.**
Each row of an ablation table should differ from the full model by exactly one
component. If two components are removed together, you cannot attribute the performance
drop. If you cannot ablate cleanly, explain why in the text.

**Report the right metrics for the task.**
Use the metrics that the field uses, not the metrics that make your results look best.
If your field uses F1 for imbalanced datasets and you report only accuracy, reviewers
will notice. If there is a standard benchmark, report on it even if your numbers are
not your strongest.

**Statistical claims require statistical evidence.**
"Model A significantly outperforms Model B" requires a significance test. If you
cannot run significance tests, change "significantly" to "consistently" or "notably"
and report error bars or confidence intervals where possible.

---

## 3. Paragraph-level rules

### 3.1 One paragraph, one idea — no exceptions

Every paragraph must be summarizable in one sentence. Before writing a paragraph,
state that sentence to yourself. Write the paragraph to develop and support that single
idea. If the summary sentence requires "and" to join two ideas, split the paragraph.

This rule has no exceptions. Short paragraphs are fine. Two-sentence paragraphs are
acceptable. Paragraphs that jump between ideas are not.

### 3.2 The "so what" test on every claim

After every factual claim or result statement, ask "so what?" out loud. If the answer
is a new piece of information, that new piece of information is your next sentence.

Wrong: "The model achieves 97.47% accuracy on the unified benchmark."

Right: "The model achieves 97.47% accuracy on the unified benchmark, representing a
15% relative improvement over the next-best baseline and confirming that the
synchronization loss — not simply the gated attention mechanism — is the primary
driver of performance."

The "so what" answer is almost always the more important sentence. Lead with the
interpretation, not just the number.

### 3.3 The first sentence of every paragraph must earn its position

The first sentence of a paragraph tells the reader what the paragraph is about and
why they should keep reading. If the first sentence is a weak transition ("In
addition," "Furthermore," "Another approach is") the paragraph will feel structureless.
Start with the claim or the main idea, then develop it.

### 3.4 Transitions between paragraphs must be logical, not decorative

"Furthermore," "Moreover," "Additionally," and "In addition" are decoration. They
say "here is another thing" without explaining the relationship. Replace them with
transitions that carry meaning: "This limitation motivates our design of...", "In
contrast to A, B...", "Building on this insight, we..."

### 3.5 Never state a limitation without following it with your solution

In the related work and introduction, every limitation you name in prior work sets up
an implicit promise to the reader: you will address this. If you name a limitation and
never return to it with your solution, the reader will look for the resolution and
not find it. Either address every limitation you name, or do not name it.

---

## 4. Sentence-level rules

### 4.1 Prefer active voice

Passive voice obscures agency and makes writing feel distant. Most sentences in a
research paper should identify who or what is doing the action.

Passive: "The results were evaluated using accuracy and F1 score."
Active: "We evaluate results using accuracy and F1 score."

Passive is acceptable when the actor genuinely does not matter or is unknown. It is
not acceptable as a default style.

### 4.2 Avoid throat-clearing openers

These sentences add no information and should be deleted:
- "It is worth noting that..."
- "It should be mentioned that..."
- "As mentioned above..."
- "Needless to say..."
- "To summarize the above..."

Delete them. Start with the actual content.

### 4.3 Do not overhedge

Research papers must be honest about uncertainty, but systematic overhedging makes
writing feel unconvinced of its own claims.

Overhedged: "Our results seem to suggest that there may potentially be some benefit
to using synchronization constraints."
Honest and direct: "Our ablation study shows that removing synchronization constraints
reduces accuracy by 15.16%, suggesting it is the primary architectural driver."

Reserve hedging language ("suggest," "indicate," "appear to") for genuinely uncertain
conclusions. Use direct language ("demonstrate," "show," "confirm") for conclusions
with direct evidence.

### 4.4 Do not undersell or oversell

**Underselling:** Describing a genuine novel contribution as "a slight modification"
or "a simple extension." If it is novel and it works, say it is novel and it works.

**Overselling:** Claiming broad impact beyond what the evidence supports.
"This framework solves the deepfake detection problem" when you have tested on two
datasets is overselling. "This framework advances multimodal deepfake detection
by addressing the open problem of synchronization-aware explainability" is honest.

Match the strength of language to the strength of evidence.

### 4.5 Numbers must be precise and consistently formatted

Inconsistent: "the model achieves approximately 97% accuracy in most cases"
Precise: "the model achieves 97.47% accuracy on the unified 2500-sample test set"

Every number in the paper must be traceable to a specific experiment and dataset. If
a number is approximate, say why ("we report approximate values since the baselines
did not provide standard deviations").

### 4.6 Acronyms must be defined on first use

Every acronym must be written out in full the first time it appears, with the acronym
in parentheses. After that, use the acronym. Do this even for common acronyms like
CNN, LSTM, XAI — not every reviewer is a specialist.

---

## 5. Avoiding AI-generation artifacts

If any part of the draft was produced with AI assistance, every sentence requires
active auditing. The following failure modes appear consistently in AI-assisted
academic writing and will cause rejection or embarrassment:

### 5.1 Phantom citations

AI models invent plausible-sounding author names, paper titles, and venue names with
high confidence. Never include a citation without having read (or at minimum opened
and verified the existence of) the actual paper. Every reference must have a real DOI,
arXiv link, or proceedings entry you can open.

### 5.2 Cross-contamination from other documents

Text generated for one context gets blended into another. The PinPoint paper contained
a Brian Krebs blog attribution and an environmental science passage in its related
work — both clearly from unrelated source material processed by the AI. Every
paragraph must be read as if you are a suspicious reviewer who has never heard of your
project. Any sentence that could have come from a different paper, a blog post, or a
news article needs to be deleted and rewritten from scratch.

### 5.3 Generic structure without substance

AI-generated text tends toward: "Many studies have investigated X. However, these
approaches have limitations. In this paper, we propose Y." This is a structural shell
with no actual content. Every sentence in the related work must name a specific paper
or specific result. Generalizations without citations are not literature review.

### 5.4 Confident vagueness

Phrases like "recent advances in deep learning have shown promising results" are
technically not wrong but carry zero information. Every general claim must be
immediately followed by a specific example: which method, which paper, which result.

### 5.5 Inconsistent notation

AI-assisted writing often introduces the same variable with different symbols in
different sections (e.g., `N` for batch size in section 3 and `B` in section 4).
Do a dedicated notation consistency pass: extract every symbol defined in the paper
into a list and verify it is used consistently everywhere.

### 5.6 The read-aloud test

Read the entire paper out loud before submitting. Every sentence that sounds wrong
when spoken needs to be fixed. This test catches run-on sentences, phantom citations,
broken transitions, sentences borrowed from other documents, and passages that make
sense visually but fail logically. Nobody does this. The papers that do it are
noticeably better.

---

## 6. Figure and table rules

### 6.1 Every figure must be interpretable without reading the main text

The caption must identify: what is being shown, what the axes or labels mean, and
what the reader should conclude. A reviewer who reads only the figures and captions
must be able to understand the paper's core claims.

### 6.2 Every table must have a header row, consistent column alignment, and a caption

Use the formatting style of your target venue. In most IEEE/ACM venues, tables use
booktabs-style horizontal rules (toprule, midrule, bottomrule) and no vertical rules.
Never use a different style without checking.

### 6.3 Bold the best number in each column of a results table

This is a universal convention. Bold the best-performing method for each metric. If
your method is not the best on a particular metric, bold the actual best-performing
method and note the discrepancy in the analysis.

### 6.4 Figures must appear near their first reference in text

A figure referenced in section 3 should not appear in section 5. LaTeX float placement
is unreliable — use `[h]` or `[t]` placement hints and check the final PDF carefully.

### 6.5 Do not use a figure just to fill space

Every figure must carry information that prose cannot convey as effectively. An
architecture diagram is essential. A bar chart showing one number per method is often
better as a table. A confusion matrix is informative. A screenshot of a user interface
is only needed if the paper is specifically about the interface. When in doubt, ask:
does this figure help a reviewer understand or believe a claim? If not, cut it.

---

## 7. Final checklist before submission

Run through every item below. Do not submit until every item is checked.

**Content:**
- [ ] Core contribution is stated in ≤2 sentences in the abstract
- [ ] All contributions in the introduction are demonstrated by a specific result
- [ ] Related work ends with an explicit gap statement
- [ ] Every claim in the methodology is explained before being described mechanically
- [ ] Every hyperparameter is reported
- [ ] Every baseline is cited with its original source
- [ ] Ablations isolate one variable at a time
- [ ] Every "significantly" claim has a statistical test behind it

**Writing:**
- [ ] Introduction was written after method and experiments
- [ ] Every paragraph has one idea and can be summarized in one sentence
- [ ] Every table result has an interpretation sentence in the text
- [ ] No phantom citations — every reference has been personally verified
- [ ] No throat-clearing openers
- [ ] Notation is consistent across all sections

**Format:**
- [ ] Page limit respected
- [ ] Figures are near their first text reference
- [ ] Best number in each table column is bolded
- [ ] Every acronym is defined on first use
- [ ] Figure captions are self-contained
- [ ] References are complete (no missing venue, year, or page numbers)

**Final test:**
- [ ] Paper has been read aloud, start to finish, at least once
