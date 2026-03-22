# Cloned Repositories

## Repo 1: Refusal-LLMs (User-Specified)
- **URL**: https://github.com/FabianHildebrandt/Refusal-LLMs
- **Purpose**: Official implementation of the primary paper (Hildebrandt et al., 2025). Extracts and analyzes refusal activations across LLM families.
- **Location**: code/Refusal-LLMs/
- **Key files**:
  - `extract_refusal.ipynb` — Extracts refusal activations from models using TransformerLens
  - `analyze_refusal.ipynb` — Analyzes activations with PCA, t-SNE, UMAP; computes GDV
  - `config.yaml` — Configuration for model selection and HuggingFace token
  - `utils/` — Helper functions for extraction and analysis
  - `GDV_Results.xlsx` — Pre-computed GDV results across all models
  - Per-model directories (e.g., `Llama-3.2-3B-Instruct/`) — Cached activations
- **Dependencies**: See `requirements.txt` (TransformerLens, scikit-learn, umap-learn, etc.)
- **Notes**: Requires HuggingFace token for model access. GPU recommended for activation extraction.

## Repo 2: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Library for mechanistic interpretability research. Enables caching and manipulating internal activations in transformer models.
- **Location**: code/TransformerLens/
- **Key files**:
  - Core library for hooking into transformer internals
  - Used by the primary paper and many interpretability studies
- **Notes**: Well-maintained library by Neel Nanda and community. Install via `pip install transformer-lens`.
