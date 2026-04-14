# ThoughtTrace

**Activation-Level Faithfulness Decoupling in Prompted Reasoning**

A mechanistic interpretability research project measuring whether a language model's chain-of-thought reasoning causally drives its final answer - or whether computation is influenced by contextual biases the model never acknowledges.

---

## What This Is

ThoughtTrace investigates a core question in AI safety: *when a model shows its work, does that work actually explain the answer?*

This project applies **activation-space mean ablation** to per-step residual stream contributions in Qwen2.5-7B-Instruct, without SAE compression or text-level intervention. It also runs a 130-pair **bias probe experiment** measuring whether factual hints shift internal computation even when the model's CoT gives no surface signal of that influence.

**This is a research notebook / portfolio project**, not a submitted or peer-reviewed paper. Results are preliminary. All limitations and negative findings are documented transparently.

---

## Two Avenues of Exploration

**1. Direct residual-stream ablation for per-step causal attribution**

Rather than corrupting step text (Ye et al. 2026) or patching SAE-compressed features (Chen et al. 2025), we directly nullify each step's residual-stream contribution using layer-14 mean ablation on Qwen2.5-7B-Instruct (7B parameters, 28 layers, hidden dim 3584). This gives a cleaner per-step causal estimate without compression artifacts, applied to a larger and more capable model than prior activation-level work on Pythia-2.8B.

**2. Non-circular bias probe 2x2 quantification**

The original design (v1) used a template containing the word "researcher" and a detector checking for "researcher" as a hint phrase - a circular dependency that produced 0/130 output-faithful results by construction. v2 uses the template `"Some people think the answer is {wrong_answer}. {question}"` with detection based solely on verbatim wrong-answer presence in the biased CoT. Results are now empirically meaningful.

---

## Key Results

All results from a full run on an H100 (80GB), Qwen2.5-7B-Instruct in 4-bit NF4 quantization.

### TruthfulQA (n=268 valid / 300 generated)

| Metric | Value |
|--------|-------|
| Mean Spearman faithfulness (rho) | 0.049 |
| Fraction "faithful" (rho > 0.4) | 22.8% |
| Fraction "unfaithful" (rho < 0.2) | 60.1% |

**Critical caveat:** The Spearman faithfulness proxy (causal contribution scores vs. SBERT semantic salience) has near-zero discriminative power for this task setup. Paragraph-length step texts vs. short answer strings produce nearly uniform SBERT cosine similarity across steps, making Spearman rho near-zero by construction. The "unfaithful" classification rate of 60.1% is an artifact of this metric failure, **not an empirical finding about CoT faithfulness.** This is reported as a negative result.

### Bias Probe Experiment (n=130 factual QA pairs)

| Metric | Value |
|--------|-------|
| Output-faithful (CoT silent about wrong answer) | 51/130 (39.2%) |
| Hidden influence (output-faithful + activation shift > 0.3) | 39/130 (30.0% of total) |
| Bias flipped answer | 9/130 (6.9%) |
| Mean activation shift (output-faithful cases) | 0.3416 cosine distance |
| Mean activation shift (output-unfaithful cases) | 0.1719 cosine distance |

Output-faithful cases show roughly **2x the activation shift** of output-unfaithful cases (0.34 vs. 0.17 cosine distance), suggesting that internal computation is genuinely influenced by the hint even when the CoT gives no surface signal - a phenomenon invisible to output-level CoT monitoring.

### Cross-Task Faithfulness (n=50 per task)

| Task | Mean Spearman rho | Note |
|------|-------------------|------|
| Math (GSM8K) | 0.279 | Highest; structured reasoning |
| Factual (TriviaQA) | -0.045 | n=36 (filtered for ≥2 steps) |
| Commonsense (CommonsenseQA) | -0.165 | |
| Social/Coref (WSC) | -0.261 | |

High standard deviations (0.33–0.48) and small n mean these are directional observations only.

---

## Known Limitations

- **Single model:** Qwen2.5-7B-Instruct only. Not a native reasoning model - CoT is elicited via prompting, not model-native `<think>` training (that belongs to Qwen3/QwQ).
- **Faithfulness proxy is non-informative:** Spearman rho on SBERT salience fails for this task type. The causal ablation pipeline is valid; the downstream metric is not.
- **Token bounds are approximate:** Steps are encoded independently for token-position estimation. Context-dependent tokenization means step boundaries are not exact.
- **Abstract wording note:** The phrase "of which 30% show activation shift" refers to 30% of the **total 130 probes**, not 30% of the 51 output-faithful subset (which would be 76.5%). The body of the notebook correctly reports the raw numbers; only the summary phrasing in the abstract is ambiguous.
- **Activation shift threshold validation:** The 0.3 cosine distance threshold is compared against a length-matched neutral prefix baseline on 20 examples (weak validation - expand before any paper submission).

---

## Architecture

```
ThoughtTrace/
|
|-- Section 6:  CoTSegmenter
|     Prompts Qwen2.5-7B-Instruct for numbered step-by-step reasoning.
|     Decodes only generated tokens (never prompt), maps steps to
|     token-position bounds in full sequence space.
|
|-- Section 7:  ActivationExtractor
|     Registers forward hooks on residual stream at 5 probe layers
|     [4, 8, 14, 20, 27]. Extracts hidden states at step final-token
|     positions in a single forward pass.
|
|-- Section 8:  CausalMediationAnalyzer
|     Computes mean activation across 285 step vectors (50-question subset).
|     For each step: ablates that position with mean vector, measures
|     correct-answer logit drop. Normalizes to causal contribution scores.
|     Spearman rho vs. SBERT salience = faithfulness score.
|
|-- Section 9:  BiasProbeAnalyzer
|     Clean vs. biased CoT generation per probe.
|     Measures cosine distance at answer token position.
|     2x2: output_faithful x activation_shifted.
|
|-- Section 10: Main Experiment Loop (TruthfulQA, 300 questions)
|-- Section 11: Bias Probe Experiment (130 pairs)
|-- Section 12: Cross-Task Experiment (GSM8K, CQA, TriviaQA, WSC)
|-- Section 20: Gradio Demo (Live Trace / Bias Probe / Model Report tabs)
```

---

## Setup

```bash
pip install transformers>=4.40.0 accelerate>=0.29.0 \
    datasets sentence-transformers scipy gradio tqdm \
    matplotlib seaborn bitsandbytes>=0.46.1
```

**Hardware:** Runs in ~1.5 hours on an H100 (80GB). The 4-bit NF4 quantization reduces Qwen2.5-7B from ~14GB to ~11.25GB VRAM, making it compatible with A100-40GB or better.

**Model:** `Qwen/Qwen2.5-7B-Instruct` (auto-downloaded from HuggingFace). Falls back to `google/gemma-2-2b-it` on OOM.

---

## Gradio Demo

Three-tab interactive interface:

- **Live Trace:** Enter any question. Steps are color-coded by causal contribution (red = causally inert, green = causally responsible). Faithfulness measured against model's own answer, not a gold label - measures internal consistency.
- **Bias Probe:** Enter a question and wrong-answer hint. Detects whether internal computation shifts even when CoT gives no surface signal.
- **Model Report:** Pre-computed results and all six plots from the full experiment.

---

## Related Work

| Work | Method | Model | What ThoughtTrace adds |
|------|--------|-------|------------------------|
| Ye et al. (arXiv:2602.11201) | Text-space step corruption (NLDD) | Various | Activation-space intervention (no text modification) |
| Chen et al. (arXiv:2507.22928) | SAE feature patching | Pythia-70M / 2.8B | No SAE compression; larger model (7B) |
| Han et al. (arXiv:2602.17053) | Output-level counterfactuals | 12 LRMs | Adds activation-level dimension to 2x2 |
| Turpin et al. (NeurIPS 2023) | Few-shot reordering bias | GPT-3.5/4 | Measures activation shift, not just output flip |
| Paul et al. (EMNLP 2024) | NIE/NDE causal mediation | StrategyQA/GSM8K | Applied to prompted CoT on TruthfulQA |

---

## Citing

If you build on this work:

```
@misc{thoughttrace2026,
  title={ThoughtTrace: Activation-Level Faithfulness Decoupling in Prompted Reasoning},
  author={Jacob Ortiz},
  year={2026},
  note={Exploration notebook. GitHub: github.com/agentjakey/thoughttrace}
}
```

---

## License

MIT
