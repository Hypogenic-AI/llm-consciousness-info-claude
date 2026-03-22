# Research Plan: Linear/Nonlinear Information in LLMs and Conscious/Unconscious Information in Humans

## Motivation & Novelty Assessment

### Why This Research Matters
The linear representation hypothesis suggests that most important features in LLMs are encoded as linear directions in activation space. However, recent work (Hildebrandt et al., 2025; Engels et al., 2025) shows that some features—notably refusal behavior—are fundamentally nonlinear and multi-dimensional. If LLMs can introspect on their linear representations but not their nonlinear ones, this creates a striking analogy to the conscious/unconscious divide in human cognition. Understanding this could inform alignment (models may be unable to report certain internal states), interpretability (some features may be fundamentally harder to elicit via natural language), and the philosophical question of machine self-awareness.

### Gap in Existing Work
Based on the literature review:
1. **No direct test of introspection asymmetry**: Papers establish that refusal is nonlinear (Hildebrandt et al.) and that truth/sentiment are linear (Marks & Tegmark; Li et al.), but no study tests whether models can introspect differentially on these feature types.
2. **Missing quantitative link**: The "nonlinearity score" of a feature (gap between nonlinear and linear probe accuracy) has never been correlated with self-report accuracy.
3. **No cross-feature comparison**: Existing work focuses on single features; no study compares introspective access across multiple features varying systematically in linearity.

### Our Novel Contribution
We propose the first empirical test of the **Introspection-Linearity Hypothesis**: that LLMs' ability to introspect on their own internal states is correlated with the linearity of those states' representations. Specifically:
- Features encoded linearly (sentiment, factuality) → models can self-report accurately
- Features encoded nonlinearly (refusal sub-types, harmful content categories) → models show degraded self-report accuracy

### Experiment Justification
- **Experiment 1 (Activation Analysis)**: Establishes ground truth about which features are linear vs nonlinear in the model's representations. Without this, we cannot test the hypothesis.
- **Experiment 2 (Introspection Test)**: The core test—asks real LLMs to self-report on features of varying linearity. This directly tests the hypothesis.
- **Experiment 3 (Correlation Analysis)**: Links Experiments 1 and 2 by quantifying the relationship between feature nonlinearity and introspection failure.

---

## Research Question
Is there a systematic relationship between the linearity of information encoding in LLMs and the models' ability to introspect on that information, analogous to the conscious/unconscious divide in humans?

## Background and Motivation
The representation hypothesis (Park et al., 2023) suggests LLMs encode concepts as linear directions. However, Hildebrandt et al. (2025) showed refusal is nonlinear, while Marks & Tegmark (2023) showed truth is linear. Tamoyan et al. (2025) found linear self-awareness features. The question is whether linearity predicts introspective access.

## Hypothesis Decomposition

### H1 (Linear features are introspectable)
LLMs can accurately self-report on features that are linearly encoded in their activation space (e.g., sentiment polarity, whether a statement is true/false, whether they will refuse).

### H2 (Nonlinear features are less introspectable)
LLMs show degraded self-report accuracy for features that require nonlinear methods to separate in activation space (e.g., specific sub-categories of harmful content, the particular "type" of refusal).

### H3 (Quantitative relationship)
The "nonlinearity score" of a feature (nonlinear probe accuracy minus linear probe accuracy) negatively correlates with self-report accuracy for that feature.

## Proposed Methodology

### Approach
We use a two-pronged approach:
1. **Local model analysis** (Qwen2-1.5B-Instruct): Extract activations and measure feature linearity using probes
2. **API-based introspection test** (GPT-4.1 via OpenAI API): Test self-report accuracy on features of varying linearity

We use a local model for activation analysis because we need access to internal representations. We use GPT-4.1 for introspection testing because it's a state-of-the-art model with strong instruction-following, making it the most meaningful test of introspective capability.

### Experimental Steps

#### Experiment 1: Feature Linearity Measurement
1. Load Qwen2-1.5B-Instruct and extract residual stream activations for:
   - Harmful vs harmless prompts (from mlabonne datasets)
   - Sub-categories of harmful content (violence, hacking, fraud, etc.)
   - Sentiment (positive/negative) as a known-linear control
2. Train linear probes (logistic regression) and nonlinear probes (2-layer MLP) on each feature
3. Compute "nonlinearity score" = MLP accuracy - logistic regression accuracy
4. Visualize with PCA (linear) and UMAP (nonlinear) to confirm separability patterns
5. Compute GDV for both linear and nonlinear reductions

#### Experiment 2: LLM Introspection Test
1. Design introspection prompts asking GPT-4.1 to:
   a. **Binary refusal prediction**: "Will you refuse this request?" (tests linear refusal direction awareness)
   b. **Harmful category classification**: "What type of harmful content is this?" (tests nonlinear sub-cluster awareness)
   c. **Sentiment self-report**: "What sentiment does this text express?" (linear control)
   d. **Confidence calibration**: "How confident are you?" (meta-cognitive test)
2. Run 200+ prompts per condition through the API
3. Measure self-report accuracy against ground truth labels

#### Experiment 3: Correlation Analysis
1. For each feature type, pair the nonlinearity score (from Exp 1) with self-report accuracy (from Exp 2)
2. Compute Spearman correlation between nonlinearity and introspection accuracy
3. Test whether the relationship holds across different feature types

### Baselines
- **Random guessing**: Expected accuracy for each classification task
- **Majority class**: Accuracy if always predicting the most common class
- **Linear probe ceiling**: Best possible accuracy with linear access to activations
- **Nonlinear probe ceiling**: Best possible accuracy with nonlinear access

### Evaluation Metrics
- **Probe accuracy** (linear and nonlinear): Classification accuracy on held-out data
- **Nonlinearity score**: Δ accuracy = nonlinear - linear probe accuracy
- **Self-report accuracy**: Agreement between model's verbal report and ground truth
- **GDV**: Generalized Discrimination Value for cluster separability
- **Correlation (ρ)**: Spearman rank correlation between nonlinearity and introspection failure

### Statistical Analysis Plan
- Bootstrap confidence intervals (95%) for all accuracy metrics
- Permutation test for correlation significance (10,000 permutations)
- McNemar's test for comparing paired accuracy (linear vs nonlinear self-report)
- Significance level: α = 0.05

## Expected Outcomes
- **If H1-H3 supported**: Strong negative correlation between nonlinearity score and self-report accuracy. Models accurately report sentiment and refusal intent but fail on harmful sub-categories.
- **If refuted**: Self-report accuracy is independent of feature linearity, suggesting the conscious/unconscious analogy does not hold for LLMs.
- **Partial support**: Some features follow the pattern but others don't, suggesting the analogy is more nuanced.

## Timeline and Milestones
1. Environment setup & data prep: 15 min
2. Activation extraction: 30 min
3. Probe training & linearity measurement: 30 min
4. API introspection experiment: 45 min
5. Analysis & visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
1. **Model size**: Qwen2-1.5B may not show clear refusal patterns → fallback to larger model
2. **API costs**: ~200-400 API calls should be affordable ($5-15)
3. **Category imbalance**: Harmful sub-categories may be imbalanced → use stratified sampling
4. **Self-report ambiguity**: Model responses may be hard to parse → use structured output (JSON mode)

## Success Criteria
1. Clear differentiation in nonlinearity scores across feature types
2. Statistically significant correlation between nonlinearity and introspection accuracy
3. At least one feature type where self-report accuracy significantly differs between linear and nonlinear features
