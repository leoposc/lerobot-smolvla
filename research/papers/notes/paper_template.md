# {{citekey}}

## Bibliographic Information

{{authors}}, "{{title}}" ({{year}}). {{venue}}. DOI: {{doi}}. arXiv: {{arxiv_id}}. URL: {{url}}
PDF: {{pdf_path}}

---

## Overview

*Write a concise 150–200 word summary capturing the paper's core argument, problem setting, and main finding in your own words.*

**Core claim:** [one sentence summarising the central thesis]


**Key finding:** [one sentence summarising the primary empirical or theoretical result]


**Why it matters for my thesis:** [1–2 sentences connecting to parameter-efficient VLA adaptation at scale]

---

## Main Contributions

### {{contribution_1}}
{{brief description}}

### {{contribution_2}}
{{brief description}}

*(Add more bullet points as needed)*

---

## Research Problem and Motivation

**Problem statement:** Clearly articulate the research gap or problem addressed.

**Motivation:** Why is this problem important? What has prior work overlooked?

---

## Technical Methodology and Architecture

**Core approach:** {{high-level description}}

### Model Architecture
{{describe architectures, modules, special components like action tokenisation, flow matching layers, etc.}}

### Parameter-Efficient Adaptation Strategy
{{explain freezing strategies, LoRA/RANK dimensions, which layers/tokens are adapted vs frozen}}

### Training Procedure
{{data regime, optimiser, learning rate schedule, duration, loss functions}}

---

## Important Equations

**{{equation_1}}**  
$$[your latex equation]$$  
*Interpretation: ...*  

*(Include equations that clarify methodology or key results)*

---

## Experimental Setup and Results

### Datasets
- {{dataset}} (N, distribution relevant to VLA tasks)
- ...

### Tasks Evaluated
- {{task_a}} (similarity/dissimilarity to pretraining distribution)
- {{task_b}}
- ...

### Adaptation Strategies Compared
1. {{strategy_1}: {{freeze/LoRA/specific_pattern}}} – {{results with numbers}}
2. {{strategy_2}} – {{results with numbers}}
3. {{strategy_3}} – {{results with numbers}}

*(Always cite numerical results from the paper. Do not generalise beyond explicitly reported findings.)*

### Key Performance Metrics
{{performance metric (e.g., task success rate, FLOPs, adaptation time), value, baseline comparison}}

---

## Prerequisite and Foundational Literature

**Essential prior work:**
- {{citation_key_1}}: [short reason why this paper's method is relevant]
- {{citation_key_2}}: [why needed for understanding VLA pretraining/architecture]
- ...

**Technical dependencies:** Understanding of [transformers/flow-matching/vision-language-action models/LoRA] is required.

---

## Limitations and Critical Assessment

{{What does the paper admit as limitations? What are objective weaknesses in the design/evaluation?}}

*(Separate authors' claims from your own analysis. Do not amplify beyond what is stated or what you can independently verify.)*

---

## Relevance to My Thesis

**Direct relevance:** {{explicitly state how this work addresses parameter-efficient VLA adaptation}}

{{How does the paper's model of layer/task representation reuse inform whether pretrained representations transfer?}}

{{Which aspects (e.g., freezing policy, adapter design, evaluation protocol) align or diverge from your own experimental plan?}}

---

## Comparison with Papers Already Read


| Aspect | {{current_paper citekey}} | {{other_paper citekey}} | Shared pattern? Divergence? |
|--------|--------------------------|-------------------------|----------------------------|
| Adaptation method | ... | ... | ... |
| Strategies compared | ... | ... | ... |
| Performance delta on similar tasks | +X / -Y | ... | ... |
| Parameter budget | {{N_params}} | {{M_params}} | ... |

**Direct comparison:** Write 3–5 points explicitly contrasting with papers below. Cite the other paper(s).

{{citation_key_3}}, "{{title}}" ({{year}}): [comparison point]

*(Continue as needed)*

---

## Recommended Next Reading

Based on this paper, read next:
1. {{citekey}} ({{reason}})
2. {{citekey}} ({{reason}})
3. ...

---

## Personal Reading Notes

**My own observations:** Write here your critical engagement with the material and any connections to other papers or your own research questions.

*(This section belongs in the log — only populate when you explicitly confirm reading.)*

---

## Source References

> *"Exact quotation"* (authors, p. {{p}}, s. {{section}})

*(Add page/section references where you have access to the full text and can verify location. Do not cite sources you cannot locate.)*
