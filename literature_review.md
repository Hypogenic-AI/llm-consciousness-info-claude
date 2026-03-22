# Literature Review: Linear/Nonlinear Information in LLMs and Conscious/Unconscious Information in Humans

## Research Area Overview

This review covers the intersection of mechanistic interpretability, representation geometry in LLMs, and the analogy between linear/nonlinear information processing in neural networks and conscious/unconscious information in humans. The central question is whether LLMs can introspect on linearly-encoded information (analogous to conscious access) but not on nonlinearly-encoded information (analogous to unconscious processing), with refusal behavior as a key test case.

## Key Papers

### Paper 1: Refusal Behavior in Large Language Models: A Nonlinear Perspective (Hildebrandt et al., 2025) — **PRIMARY PAPER**
- **Authors**: Fabian Hildebrandt, Andreas Maier, Patrick Krauss, Achim Schilling
- **Year**: 2025
- **Source**: arXiv:2501.08145
- **Key Contribution**: Demonstrates that refusal behavior in LLMs is **nonlinear and multidimensional**, challenging the assumption that it resides in a single linear direction.
- **Methodology**:
  - Analyzed 6 LLMs across 3 families (Llama 3.2, Bloom, Qwen2) at varying scales
  - Used PCA (linear), t-SNE, and UMAP (nonlinear) dimensionality reduction on residual activations
  - Employed Generalized Discrimination Value (GDV) to quantify cluster separability
  - Extracted activations at last token position after attention and MLP layers using TransformerLens
  - Used difference-in-means to identify refusal direction
- **Datasets Used**:
  - mlabonne/harmless_alpaca (from Stanford ALPACA, 52K instructions)
  - mlabonne/harmful_behaviors (from LLM Attacks dataset)
- **Key Results**:
  - UMAP/t-SNE consistently achieve better separability (lower GDV) than PCA across all models
  - Best GDV: Qwen2-1.5B-Instruct achieved -0.89 with UMAP vs -0.44 with PCA
  - Sub-clusters emerge within harmful instruction embeddings (visible only with nonlinear methods)
  - Architecture-specific refusal patterns: Qwen encodes early, Bloom in middle layers, Llama refines gradually
- **Code Available**: https://github.com/FabianHildebrandt/Refusal-LLMs
- **Relevance**: Directly establishes that refusal is a nonlinear feature — central to the hypothesis that LLMs cannot introspect on nonlinear information

### Paper 2: Refusal in Language Models Is Mediated by a Single Direction (Arditi et al., 2024)
- **Authors**: Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, Neel Nanda
- **Year**: 2024
- **Source**: arXiv:2406.11717 (519 citations)
- **Key Contribution**: Shows refusal is mediated by a **one-dimensional subspace** across 13 models up to 72B parameters. Removing this direction disables refusal; adding it induces refusal on harmless prompts.
- **Methodology**: Difference-in-means to find refusal direction; weight orthogonalization to ablate it
- **Relevance**: Establishes the "linear refusal" baseline that Hildebrandt et al. challenge. The tension between these two findings is central to the research hypothesis.

### Paper 3: Not All Language Model Features Are One-Dimensionally Linear (Engels et al., 2025)
- **Authors**: Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, Max Tegmark
- **Year**: 2025 (ICLR 2025)
- **Source**: arXiv:2405.14860
- **Key Contribution**: Provides rigorous theoretical framework for **irreducible multi-dimensional features** in LLMs. Discovers circular features (days of week, months of year) in GPT-2 and Mistral 7B.
- **Methodology**:
  - Formal definitions of reducibility via separability index and ε-mixture index
  - Uses sparse autoencoders (SAEs) to find multi-dimensional features automatically
  - Intervention experiments showing circular features are causally used for modular arithmetic
- **Key Definitions**:
  - A feature is "irreducible" if no rotation/translation makes it separable or a mixture of lower-dimensional features
  - Multi-dimensional Superposition Hypothesis: hidden states are sums of sparse, low-dimensional irreducible features
- **Relevance**: Provides the theoretical grounding that some LLM representations are inherently multi-dimensional (nonlinear in 1D sense), which connects directly to the hypothesis about nonlinear = unconscious

### Paper 4: The Geometry of Truth (Marks & Tegmark, 2023)
- **Authors**: Samuel Marks, Max Tegmark
- **Year**: 2023
- **Source**: arXiv:2310.06824 (427 citations)
- **Key Contribution**: LLMs linearly represent truth/falsehood of factual statements at sufficient scale. Simple difference-in-mean probes generalize across datasets and are causally implicated.
- **Relevance**: Establishes that some types of "self-knowledge" (truthfulness) are linear — a model could potentially introspect on these

### Paper 5: The Linear Representation Hypothesis (Park et al., 2023)
- **Authors**: Kiho Park, Yo Joong Choe, Victor Veitch
- **Year**: 2023
- **Source**: arXiv:2311.03658 (408 citations)
- **Key Contribution**: Formalizes what "linear representation" means using counterfactuals. Connects linear representations to probing and steering. Identifies a non-Euclidean inner product respecting language structure.
- **Relevance**: Theoretical foundation for understanding what "linear" means in LLM representations

### Paper 6: Discovering Latent Knowledge Without Supervision (Burns et al., 2022)
- **Authors**: Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt
- **Year**: 2022
- **Source**: arXiv:2212.03827 (599 citations)
- **Key Contribution**: Discovers what LLMs "know" (vs what they "say") by finding directions in activation space satisfying logical consistency. Achieves this without supervision.
- **Relevance**: Demonstrates that LLMs have internal knowledge that may differ from their outputs — analogous to unconscious knowledge

### Paper 7: Inference-Time Intervention (Li et al., 2023)
- **Authors**: Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
- **Year**: 2023
- **Source**: arXiv:2306.03341 (951 citations)
- **Key Contribution**: Shifts model activations during inference to enhance truthfulness. Identifies that LLMs may have internal representations of truth likelihood even while producing falsehoods.
- **Relevance**: Shows LLMs have "hidden knowledge" accessible via linear intervention — supports the conscious/unconscious analogy

### Paper 8: Representation Engineering (Zou et al., 2023)
- **Authors**: Andy Zou et al.
- **Year**: 2023
- **Source**: arXiv:2310.01405
- **Key Contribution**: Top-down approach to AI transparency by reading and controlling representations of high-level concepts (honesty, emotion, bias) via linear directions.
- **Relevance**: Framework for manipulating LLM behavior through representation space — relevant for testing introspection

### Paper 9: Factual Self-Awareness in Language Models (Tamoyan et al., 2025)
- **Authors**: Hovhannes Tamoyan, Subhabrata Dutta, Iryna Gurevych
- **Year**: 2025
- **Source**: arXiv:2501.03489
- **Key Contribution**: LLMs encode **linear features** in the residual stream that predict whether they will recall correct facts. Self-awareness emerges rapidly during training and peaks in intermediate layers.
- **Relevance**: Direct evidence for "conscious-like" linear self-monitoring in LLMs

### Paper 10: When Do LLMs Admit Their Mistakes (Yang & Jia, 2025)
- **Authors**: Yuqing Yang, Robin Jia
- **Year**: 2025
- **Source**: arXiv:2501.04648
- **Key Contribution**: LLM retraction (admitting mistakes) is driven by internal "belief" states that can be linearly probed. Model belief causally drives retraction behavior.
- **Relevance**: Shows LLMs have linearly accessible self-assessment that influences behavior — another instance of "conscious-like" information

### Paper 11: Emergent Linear Representations in World Models (Nanda et al., 2023)
- **Authors**: Neel Nanda, Andrew Lee, Martin Wattenberg
- **Year**: 2023
- **Source**: arXiv:2309.00941 (290 citations)
- **Key Contribution**: Othello-playing models learn linear representations of board state. "My color" vs "opponent's color" probing reveals interpretable internal states.
- **Relevance**: Demonstrates linear world models emerge in sequence models, supporting the linear=accessible hypothesis

### Paper 12: Probing Classifiers: Promises, Shortcomings, and Advances (Belinkov, 2021)
- **Authors**: Yonatan Belinkov
- **Year**: 2021
- **Source**: arXiv:2102.12452 (666 citations)
- **Key Contribution**: Critical review of probing methodology — highlights that probe complexity matters for interpretability claims.
- **Relevance**: Methodological foundation — linear probes detect linear info, nonlinear probes detect nonlinear info; this distinction is central to the research hypothesis

## Common Methodologies

1. **Linear Probing**: Training linear classifiers on model activations to detect features (used in Marks & Tegmark, Burns et al., Tamoyan et al., Yang & Jia)
2. **Difference-in-Means**: Computing mean activation difference between contrasting sets to find feature directions (Arditi et al., Hildebrandt et al.)
3. **Dimensionality Reduction**: PCA (linear), t-SNE, UMAP (nonlinear) for visualizing activation structure (Hildebrandt et al.)
4. **Sparse Autoencoders (SAEs)**: Decomposing activations into sparse, interpretable features (Engels et al.)
5. **Activation Steering/Intervention**: Adding or removing directions from activation space to control model behavior (Li et al., Zou et al., Turner et al.)
6. **GDV (Generalized Discrimination Value)**: Quantifying cluster separability invariant to scaling/shifting (Hildebrandt et al.)

## Standard Baselines
- **Linear probes** (logistic regression on activations) as baseline for detecting information
- **PCA** as linear dimensionality reduction baseline
- **Difference-in-means** direction as baseline for feature identification
- **Random directions** as negative control for steering experiments

## Evaluation Metrics
- **GDV**: Cluster separability metric (more negative = better separation), ranges from 0 to -1
- **Probing accuracy**: Classification accuracy of linear/nonlinear probes
- **Mutual information**: Between probe predictions and ground truth
- **Separability index**: Minimum mutual information across rotations (Engels et al.)
- **ε-mixture index**: Fraction of distribution projectable near zero (Engels et al.)

## Datasets in the Literature
- **mlabonne/harmless_alpaca**: Harmless instructions (from ALPACA), used for refusal studies
- **mlabonne/harmful_behaviors**: Harmful instructions (from LLM Attacks), used for refusal studies
- **TruthfulQA**: For truthfulness probing (Li et al.)
- **True/false factual datasets**: For truth representation studies (Marks & Tegmark)
- **Various NLG benchmarks**: 700+ datasets across 15 tasks (Ji et al.)

## Gaps and Opportunities

1. **No direct test of introspection on linear vs nonlinear features**: Papers show refusal is nonlinear and truth is linear, but no one has explicitly tested whether models can "report" on their linear features but not their nonlinear ones
2. **Missing consciousness analogy formalization**: The analogy between linear/conscious and nonlinear/unconscious is implicit but hasn't been rigorously tested
3. **Limited nonlinear probing**: Most interpretability work uses linear probes; systematic comparison with nonlinear probes for the same features is rare
4. **Cross-feature comparison**: No study compares introspective access across multiple feature types varying in linearity
5. **Scale effects**: How does the linear/nonlinear distinction scale with model size?

## Recommendations for Our Experiment

### Recommended Datasets
1. **mlabonne/harmless_alpaca** + **mlabonne/harmful_behaviors** — directly from the primary paper, enables replication and extension
2. **TruthfulQA** — for testing introspection on truthfulness (a linear feature)

### Recommended Baselines
1. **Linear probes** (logistic regression) on model activations
2. **Nonlinear probes** (MLP) on model activations
3. **PCA** vs **UMAP/t-SNE** separability comparison (replicating Hildebrandt et al.)
4. **Difference-in-means** direction identification

### Recommended Metrics
1. **GDV** for activation separability (using Hildebrandt et al.'s implementation)
2. **Probing accuracy** (linear vs nonlinear) for feature detection
3. **Self-report accuracy** — asking the model to report on its own features
4. **Steering success rate** — whether manipulating linear vs nonlinear features changes model behavior predictably

### Methodological Considerations
- Use **TransformerLens** for activation extraction (well-tested, used by primary paper)
- Start with smaller models (Qwen2-0.5B, Llama-3.2-1B) for computational feasibility
- Compare multiple model families to test universality
- The key experiment: present models with prompts that activate linear features (truth, sentiment) vs nonlinear features (refusal sub-clusters), then test whether the model can verbally report on the presence/nature of these features
- Consider both "behavioral introspection" (asking the model) and "mechanical introspection" (probing whether the model's self-reports correlate with activation patterns)
