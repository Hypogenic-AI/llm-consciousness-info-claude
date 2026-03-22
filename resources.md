# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: "Is Linear/Nonlinear Information in LLMs Similar to Conscious/Unconscious Information in Humans?"

## Papers
Total papers downloaded: 16

| # | Title | Authors | Year | Citations | File | Key Info |
|---|-------|---------|------|-----------|------|----------|
| 1 | Refusal Behavior in LLMs: A Nonlinear Perspective | Hildebrandt et al. | 2025 | - | papers/2501.08145_*.pdf | **PRIMARY** — refusal is nonlinear |
| 2 | Refusal Is Mediated by a Single Direction | Arditi et al. | 2024 | 519 | papers/2406.11717_*.pdf | Refusal as linear direction |
| 3 | Not All Features Are One-Dimensionally Linear | Engels et al. | 2025 | - | papers/2405.14860_*.pdf | Multi-dimensional features theory |
| 4 | The Geometry of Truth | Marks & Tegmark | 2023 | 427 | papers/2310.06824_*.pdf | Linear truth representations |
| 5 | The Linear Representation Hypothesis | Park et al. | 2023 | 408 | papers/2311.03658_*.pdf | Formalizes linear representations |
| 6 | Discovering Latent Knowledge | Burns et al. | 2022 | 599 | papers/2212.03827_*.pdf | Unsupervised knowledge discovery |
| 7 | Inference-Time Intervention | Li et al. | 2023 | 951 | papers/2306.03341_*.pdf | Truthfulness steering |
| 8 | Representation Engineering | Zou et al. | 2023 | - | papers/2310.01405_*.pdf | Top-down interpretability |
| 9 | Factual Self-Awareness in LMs | Tamoyan et al. | 2025 | 1 | papers/2501.03489_*.pdf | Linear self-awareness features |
| 10 | When Do LLMs Admit Mistakes | Yang & Jia | 2025 | 2 | papers/2501.04648_*.pdf | Belief-driven retraction |
| 11 | Activation Engineering | Turner et al. | 2024 | - | papers/2308.10248_*.pdf | Steering via activation addition |
| 12 | Steering Llama 2 via CAA | Panickssery et al. | 2024 | - | papers/2312.06681_*.pdf | Contrastive activation addition |
| 13 | Emergent Linear Representations | Nanda et al. | 2023 | 290 | papers/2309.00941_*.pdf | Linear world models (Othello) |
| 14 | Probing Classifiers Review | Belinkov | 2021 | 666 | papers/2102.12452_*.pdf | Probing methodology |
| 15 | Jailbreak Features | Kirch et al. | 2024 | - | papers/2411.03343_*.pdf | Nonlinear jailbreak features |
| 16 | Alignment Faking | Greenblatt et al. | 2024 | - | papers/2412.14093_*.pdf | Resistance to alignment |

See [papers/README.md](papers/README.md) for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| harmless_alpaca | HuggingFace (mlabonne) | 25K train + 6K test | Harmless instructions | datasets/harmless_alpaca/ | From Stanford ALPACA |
| harmful_behaviors | HuggingFace (mlabonne) | 416 train + 104 test | Harmful instructions | datasets/harmful_behaviors/ | From LLM Attacks |

See [datasets/README.md](datasets/README.md) for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Refusal-LLMs | github.com/FabianHildebrandt/Refusal-LLMs | Primary paper implementation | code/Refusal-LLMs/ | Extraction + analysis notebooks |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability library | code/TransformerLens/ | Core tool for activation caching |

See [code/README.md](code/README.md) for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with diligent mode for comprehensive search
- Search queries: "linear probing LLM representations introspection self-knowledge", "nonlinear representations neural networks probing consciousness awareness LLM"
- Cross-referenced citations from primary paper (Hildebrandt et al.)
- Focused on papers at the intersection of: (a) linear/nonlinear representations, (b) LLM self-awareness/introspection, (c) refusal behavior and alignment

### Selection Criteria
- Prioritized papers that directly address the linear vs nonlinear representation distinction
- Included foundational interpretability papers (probing, linear representation hypothesis)
- Included self-awareness/introspection papers as evidence for the "conscious" side of the analogy
- Selected papers with code availability when possible
- Covered both theoretical frameworks and empirical studies

### Challenges Encountered
- Some GitHub repos referenced in papers could not be found (refusal-direction by Arditi et al.)
- The connection between neuroscience consciousness theories and LLM interpretability is largely implicit in the literature — few papers make the analogy explicit
- Multi-dimensional/nonlinear features in LLMs is still an emerging area with limited prior work

### Gaps and Workarounds
- No existing dataset specifically designed for testing LLM introspection on linear vs nonlinear features → experiment will need to construct test prompts
- No direct comparison study of linear vs nonlinear probe accuracy for refusal exists → this is the core experimental gap to fill
- Limited work on LLM "self-report" accuracy for different feature types → novel contribution area

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **mlabonne/harmless_alpaca + mlabonne/harmful_behaviors**: For refusal activation extraction (replicating Hildebrandt et al.)
- Construct additional prompt sets for testing introspection on known-linear features (truth, sentiment)

### 2. Baseline Methods
- **Linear probes** (logistic regression) on residual stream activations
- **Nonlinear probes** (2-layer MLP) on same activations
- **PCA** vs **UMAP** separability analysis (GDV metric)
- **Difference-in-means** direction for refusal

### 3. Evaluation Metrics
- **GDV** for cluster separability (linear vs nonlinear reduction)
- **Probe accuracy** gap: (nonlinear probe accuracy) - (linear probe accuracy) as measure of nonlinearity
- **Self-report correlation**: correlation between model's verbal reports about its features and actual activation patterns
- **Introspection accuracy**: whether models can identify when they are about to refuse (linear feature) vs why they refuse in specific ways (nonlinear sub-cluster structure)

### 4. Code to Adapt/Reuse
- **Refusal-LLMs**: Directly reuse extraction pipeline and GDV analysis
- **TransformerLens**: Core library for all activation extraction
- Adapt probing code from standard interpretability frameworks

### 5. Experimental Design Suggestion
The key novel experiment should test:
1. Extract activations for harmful/harmless prompts (replicate Hildebrandt et al.)
2. Identify linear features (e.g., refuse/comply direction) and nonlinear features (e.g., sub-clusters of harmful types)
3. Test whether the model can verbally report on:
   - Whether it will refuse (linear → should be accessible)
   - What sub-type of harmful content it detected (nonlinear → may not be accessible)
4. Compare self-report accuracy for linear vs nonlinear features as evidence for the conscious/unconscious analogy
