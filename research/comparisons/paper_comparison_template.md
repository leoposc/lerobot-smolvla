# {{compare_key}}: Comparison of {{paper_a citekey}} and {{paper_b citekey}}

---

## Papers Being Compared

| Paper | Key Citation Details | Primary Contribution |
|-------|---------------------|----------------------|
| A | {{paper_a}} – authors, year, venue | {{one-sentence core contribution}} |
| B | {{paper_b}} – authors, year, venue | {{one-sentence core contribution}} |

---

## Methodological Alignment

### Shared Frameworks
{{List common foundations: e.g., "Both use LoRA-adapted transformer heads," or "Both evaluate via downstream task accuracy"}}

**Specific shared elements:**
1. **Adaptation mechanism**: Both employ [LoRA/adapter-free/fixed-base-tuning]
2. **Evaluation protocol**: {{dataset}} with tasks from similar/dissimilar pretraining distributions
3. **Performance axis**: Both report {{metric}}, delta = +X / -Y compared to baseline

### Divergent Choices
{{Explicitly list where papers differ: e.g., "Paper A freezes all but output heads; Paper B uses low-rank adapters in encoder layers"}}

**Fundamental design choice:**
1. **Parameter budget allocation**: Paper A targets {{component}}, Paper B focuses on {{component}}
2. **Training objective**: Paper A uses [loss], Paper B optimises for [different loss]
3. **Task similiarity emphasis**: Paper A evaluates {{task_similarity_level}}, Paper B emphasises {{other_level}}

---

## Empirical Evidence Comparison

| Aspect | {{paper_a citekey}} – reported results | {{paper_b citekey}} – reported results | Verdict (which aligns, which conflicts with your thesis) |
|--------|----------------------------------------|----------------------------------------|----------------------------------------------------------|
| Similar-task adaptation gain | +{{X}}% (+/- SE) | +{{Y}}% (+/- SE) | {{alignment: "A matches my hypothesis; B contradicts it"}} |
| Dissimilar-task transferability | -{{X}}% | -{{Y}}% | {{which shows better zero-shot capability?}} |
| Parameter efficiency (efficiency metric, e.g., FLOPs saved/accuracy retained) | {{metric_value_A}} | {{metric_value_B}} | Which is more efficient for limited compute? |
| Adaptation time / throughput | {{time_metric_A_sec}} | {{time_metric_B_sec}} | Which is faster/practical given your budget? |

*(Only cite numbers actually reported in the papers. No extrapolation beyond measured results.)*

---

## Theoretical vs Empirical Focus

### {{paper_a citekey}} – Theoretical angle
{{Did it propose a new theoretical bound? derive an exact characterization or provide empirical evidence for a claim that would be hard to prove theoretically?}}

**Theoretical contribution:** [e.g., "Proved representation orthogonality guarantees"]
**(If purely empirical, note: *This paper makes no theoretical claims*)**

### {{paper_b citekey}} – Empirical angle
{{Is it demonstrating an experimental pattern rather than proposing a formal justification?}}

**Empirical focus:** [e.g., "Systematic benchmark across 12 task sets"]
**(If has theory: note that the main claim is justified empirically)**

---

## Implications for My Thesis on Parameter-Efficient VLA Adaptation

{{Select one path: either A, B, both, or neither as primary guidance – state it explicitly.}}

- [[primary_guidance]]: {{name of preferred paper}}, because [specific reason tied to your research design]
- [[secondary_guidance]]: [[alternative paper]] is more relevant if you need to address [specific gap]

**Direct implications for my experimental plan:**
1. **Which adaptation strategy should I test?** {{strategy_from_preferred_paper}}
2. **What baseline configs do I need to establish?** [[specific_config_justified_by_paper]]
3. **Do I expect positive/negative transfer on similar/dissimilar tasks?** {{prediction grounded in cited experiment}}

---

## Critical Synthesis

{{One paragraph synthesising: if papers disagree on something fundamental or complementary, state the unresolved tension and how your research design addresses it.}}

**Unresolved question:** Can we combine [feature_A from paper A] with [feature_B from paper B] to achieve [desired outcome]?


**My answer hypothesis:** {{clear statement of what you expect based on synthesis}}

---

## Notes for Further Research

{{What aspects remain unclear or would deepen the comparison? Only list questions that arise directly from these two.}}

1. **Clarify X**: Paper A does not specify [detail], but implies Y...
2. **Extend Z**: If we replicate B's method on C's dataset, what delta do we expect based on their reported +X/+Y performance gap?
