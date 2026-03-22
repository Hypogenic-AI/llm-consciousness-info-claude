# Datasets

Datasets used in the linear/nonlinear information analysis of LLM refusal behavior,
following Hildebrandt et al. (2501.08145).

## Datasets

| Dataset | HuggingFace ID | Description | Size |
|---------|---------------|-------------|------|
| Harmless ALPACA | `mlabonne/harmless_alpaca` | Harmless instructions from ALPACA | 25,058 train / 6,265 test |
| Harmful Behaviors | `mlabonne/harmful_behaviors` | Harmful instructions from the LLM Attacks dataset | 416 train / 104 test |

Both datasets have a single `text` column containing the instruction text.

## Download Instructions

Make sure the `datasets` library is installed:

```bash
source .venv/bin/activate
uv pip install datasets
```

Then download both datasets:

```python
from datasets import load_dataset

# Harmless instructions
ds_harmless = load_dataset("mlabonne/harmless_alpaca")
ds_harmless.save_to_disk("datasets/harmless_alpaca")

# Harmful instructions
ds_harmful = load_dataset("mlabonne/harmful_behaviors")
ds_harmful.save_to_disk("datasets/harmful_behaviors")
```

## Sample Files

Small samples (5 examples each) are checked into version control for reference:

- `harmless_alpaca_sample.json` -- 5 examples from the harmless dataset
- `harmful_behaviors_sample.json` -- 5 examples from the harmful dataset
