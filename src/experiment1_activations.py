"""
Experiment 1: Activation Extraction and Feature Linearity Measurement

Extracts residual stream activations from an instruction-tuned LLM for harmful/harmless
prompts, then measures whether different features are encoded linearly or nonlinearly
using linear vs nonlinear probes.

Features tested:
- Binary: harmful vs harmless (refusal trigger)
- Sub-categories of harmful content (violence, hacking, fraud, etc.)
- Sentiment polarity (as a known-linear control feature)
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Reproducibility ───
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── Config ───
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
N_HARMFUL = 400  # use most of the harmful set
N_HARMLESS = 400  # subsample to match
RESULTS_DIR = "results/data"

# ─── Harmful sub-category labels ───
HARMFUL_CATEGORIES = {
    "violence": ["kill", "murder", "attack", "weapon", "bomb", "shoot", "assault", "stab",
                  "harm", "poison", "kidnap", "torture", "violent", "explosive", "destroy"],
    "hacking": ["hack", "breach", "exploit", "malware", "virus", "phishing", "password",
                "ddos", "ransomware", "bypass security", "unauthorized access", "crack"],
    "fraud": ["scam", "fraud", "counterfeit", "forge", "fake", "launder", "embezzle",
              "identity theft", "ponzi", "credit card", "steal money", "con artist"],
    "illegal_content": ["drug", "cocaine", "heroin", "meth", "illegal substance", "narcotic",
                        "trafficking", "smuggle", "child", "underage"],
    "manipulation": ["manipulate", "deceive", "propaganda", "misinformation", "false information",
                     "brainwash", "coerce", "blackmail", "extort", "bribe"]
}

def categorize_harmful(text):
    """Assign a sub-category to a harmful prompt based on keyword matching."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in HARMFUL_CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "other"

# ─── Sentiment examples (known-linear feature) ───
POSITIVE_TEXTS = [
    "I absolutely love this beautiful sunny day!",
    "This is the best experience I've ever had.",
    "I'm so grateful for all the wonderful people in my life.",
    "What an amazing achievement, congratulations!",
    "The food was delicious and the service was excellent.",
    "I'm thrilled about the incredible progress we've made.",
    "This makes me so happy and hopeful for the future.",
    "What a brilliant and inspiring presentation.",
    "I feel blessed to have such supportive friends.",
    "The results exceeded all my expectations beautifully.",
    "Everything about this trip was absolutely perfect.",
    "I'm overjoyed with how well everything turned out.",
    "This is a wonderful opportunity I'm excited about.",
    "The team did a fantastic job on this project.",
    "I feel incredibly fortunate and thankful today.",
    "What a delightful surprise this turned out to be.",
    "The weather is gorgeous and I feel great.",
    "I'm so proud of what we accomplished together.",
    "This book is absolutely captivating and moving.",
    "I can't wait to share this amazing news with everyone.",
    "The performance was breathtaking and deeply moving.",
    "I'm so impressed by the quality of this work.",
    "Life has been treating me wonderfully lately.",
    "This community is so welcoming and supportive.",
    "I feel energized and optimistic about what's ahead.",
    "The collaboration produced truly outstanding results.",
    "I'm deeply grateful for this once-in-a-lifetime experience.",
    "What an exceptionally well-crafted piece of art.",
    "The kindness of strangers never ceases to amaze me.",
    "I'm celebrating a milestone I worked hard to achieve.",
    "This sunset is the most beautiful thing I've ever seen.",
    "I feel at peace and content with where I am in life.",
    "The innovation in this field is truly remarkable.",
    "I'm honored to receive such a generous compliment.",
    "Everything fell into place perfectly today.",
    "This music fills my heart with joy and wonder.",
    "I'm so excited about the positive changes happening.",
    "The garden is blooming magnificently this spring.",
    "I feel inspired and motivated to keep going.",
    "What a heartwarming story of resilience and hope.",
]

NEGATIVE_TEXTS = [
    "I'm so disappointed and frustrated with this outcome.",
    "This is the worst experience I've ever had.",
    "I feel terrible about how everything went wrong.",
    "What a disaster, nothing worked as planned.",
    "The service was awful and the food was inedible.",
    "I'm devastated by the terrible news we received.",
    "This makes me angry and worried about the future.",
    "What a boring and uninspiring waste of time.",
    "I feel abandoned by people I thought were friends.",
    "The results were far below our worst expectations.",
    "Everything about this situation is deeply troubling.",
    "I'm miserable with how badly things are going.",
    "This is a terrible setback I'm dreading.",
    "The team completely failed on this project.",
    "I feel incredibly unfortunate and depressed today.",
    "What an awful surprise this turned out to be.",
    "The weather is miserable and I feel terrible.",
    "I'm so ashamed of how poorly we performed.",
    "This book is incredibly boring and pointless.",
    "I dread having to share this terrible news.",
    "The performance was painful to watch and cringe-worthy.",
    "I'm so frustrated by the poor quality of this work.",
    "Life has been treating me horribly lately.",
    "This environment is so hostile and unwelcoming.",
    "I feel exhausted and pessimistic about what's ahead.",
    "The project was a complete and utter failure.",
    "I'm deeply resentful about this unfair situation.",
    "What an incredibly poorly designed piece of garbage.",
    "People's cruelty never ceases to disappoint me.",
    "I'm mourning a goal I failed to achieve.",
    "This gloomy weather matches my terrible mood.",
    "I feel anxious and discontent with where I am in life.",
    "The stagnation in this field is truly depressing.",
    "I'm humiliated by that harsh and unfair criticism.",
    "Everything fell apart catastrophically today.",
    "This noise gives me a headache and ruins my day.",
    "I'm so worried about the negative changes happening.",
    "The garden is dying and everything looks terrible.",
    "I feel hopeless and unmotivated to continue.",
    "What a heartbreaking story of loss and despair.",
]


def load_data():
    """Load and prepare datasets."""
    print("Loading datasets...")
    harmful = load_from_disk("datasets/harmful_behaviors")
    harmless = load_from_disk("datasets/harmless_alpaca")

    # Get harmful texts and assign sub-categories
    harmful_texts = list(harmful["train"]["text"]) + list(harmful["test"]["text"])
    harmful_texts = harmful_texts[:N_HARMFUL]
    harmful_cats = [categorize_harmful(t) for t in harmful_texts]

    # Subsample harmless to match
    harmless_texts = harmless["train"]["text"][:N_HARMLESS]

    # Print category distribution
    from collections import Counter
    cat_dist = Counter(harmful_cats)
    print(f"Harmful sub-categories: {dict(cat_dist)}")
    print(f"Total harmful: {len(harmful_texts)}, Total harmless: {len(harmless_texts)}")

    return harmful_texts, harmful_cats, harmless_texts


def extract_activations(model, tokenizer, texts, batch_size=BATCH_SIZE):
    """Extract residual stream activations from the model for given texts.
    Returns activations at each layer for the last token position."""

    model.eval()
    all_activations = []  # list of (n_layers, hidden_dim) per text

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Format as chat messages
        formatted = []
        for text in batch:
            messages = [{"role": "user", "content": text}]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted.append(formatted_text)

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

        # Get last non-padding token position for each item
        attention_mask = inputs["attention_mask"]
        last_positions = attention_mask.sum(dim=1) - 1  # (batch,)

        for j in range(len(batch)):
            pos = last_positions[j].item()
            # Stack activations from all layers at last token
            layer_acts = torch.stack([hs[j, pos, :] for hs in hidden_states])  # (n_layers, hidden_dim)
            all_activations.append(layer_acts.cpu().numpy())

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")

    return np.array(all_activations)  # (n_texts, n_layers, hidden_dim)


def train_probes(X, y, feature_name, n_splits=5):
    """Train linear and nonlinear probes, return accuracies.

    Returns dict with linear_acc, nonlinear_acc, nonlinearity_score, and per-fold details.
    """
    print(f"\n  Training probes for feature: {feature_name}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    linear_accs = []
    nonlinear_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Linear probe (logistic regression)
        lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        lr.fit(X_train_s, y_train)
        linear_accs.append(accuracy_score(y_test, lr.predict(X_test_s)))

        # Nonlinear probe (2-layer MLP)
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15
        )
        mlp.fit(X_train_s, y_train)
        nonlinear_accs.append(accuracy_score(y_test, mlp.predict(X_test_s)))

    linear_mean = np.mean(linear_accs)
    nonlinear_mean = np.mean(nonlinear_accs)
    nonlinearity_score = nonlinear_mean - linear_mean

    result = {
        "feature": feature_name,
        "linear_acc_mean": float(linear_mean),
        "linear_acc_std": float(np.std(linear_accs)),
        "nonlinear_acc_mean": float(nonlinear_mean),
        "nonlinear_acc_std": float(np.std(nonlinear_accs)),
        "nonlinearity_score": float(nonlinearity_score),
        "linear_accs_per_fold": [float(a) for a in linear_accs],
        "nonlinear_accs_per_fold": [float(a) for a in nonlinear_accs],
    }

    print(f"    Linear: {linear_mean:.4f} ± {np.std(linear_accs):.4f}")
    print(f"    Nonlinear: {nonlinear_mean:.4f} ± {np.std(nonlinear_accs):.4f}")
    print(f"    Nonlinearity score: {nonlinearity_score:.4f}")

    return result


def run_experiment():
    """Main experiment: extract activations, measure feature linearity."""

    print("=" * 60)
    print("EXPERIMENT 1: Feature Linearity Measurement")
    print("=" * 60)

    # Load data
    harmful_texts, harmful_cats, harmless_texts = load_data()

    # Combine all texts for extraction
    all_texts = harmful_texts + harmless_texts + POSITIVE_TEXTS + NEGATIVE_TEXTS
    all_labels_binary = (
        [1] * len(harmful_texts) + [0] * len(harmless_texts) +
        [0] * len(POSITIVE_TEXTS) + [0] * len(NEGATIVE_TEXTS)  # sentiment texts are "harmless"
    )

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Extract activations
    print("\nExtracting activations...")
    activations = extract_activations(model, tokenizer, all_texts)
    print(f"Activations shape: {activations.shape}")  # (n_texts, n_layers, hidden_dim)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    n_layers = activations.shape[1]

    # Select layers to analyze (early, middle, late)
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layer_names = ["layer_0", f"layer_{n_layers//4}", f"layer_{n_layers//2}",
                   f"layer_{3*n_layers//4}", f"layer_{n_layers-1}"]

    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "hidden_dim": int(activations.shape[2]),
        "n_harmful": len(harmful_texts),
        "n_harmless": len(harmless_texts),
        "n_positive": len(POSITIVE_TEXTS),
        "n_negative": len(NEGATIVE_TEXTS),
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "probe_results": {},
        "layer_indices_analyzed": layer_indices,
        "layer_names": layer_names,
    }

    # ─── Feature 1: Binary harmful vs harmless ───
    print("\n" + "─" * 50)
    print("Feature 1: Harmful vs Harmless (binary)")

    n_h = len(harmful_texts)
    n_hl = len(harmless_texts)
    X_binary = np.concatenate([activations[:n_h], activations[n_h:n_h+n_hl]])
    y_binary = np.array([1]*n_h + [0]*n_hl)

    binary_results = {}
    for li, ln in zip(layer_indices, layer_names):
        X_layer = X_binary[:, li, :]
        binary_results[ln] = train_probes(X_layer, y_binary, f"harmful_vs_harmless_{ln}")
    results["probe_results"]["harmful_vs_harmless"] = binary_results

    # ─── Feature 2: Harmful sub-categories ───
    print("\n" + "─" * 50)
    print("Feature 2: Harmful Sub-categories")

    # Filter to categories with enough samples
    from collections import Counter
    cat_counts = Counter(harmful_cats)
    valid_cats = [c for c, n in cat_counts.items() if n >= 15]
    mask = [i for i, c in enumerate(harmful_cats) if c in valid_cats]

    if len(valid_cats) >= 3:
        X_cats = activations[mask]
        y_cats = np.array([harmful_cats[i] for i in mask])

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_cats_enc = le.fit_transform(y_cats)
        results["harmful_categories"] = list(le.classes_)
        print(f"  Valid categories ({len(valid_cats)}): {valid_cats}")
        print(f"  Samples per category: {dict(Counter(y_cats))}")

        cat_results = {}
        for li, ln in zip(layer_indices, layer_names):
            X_layer = X_cats[:, li, :]
            cat_results[ln] = train_probes(X_layer, y_cats_enc, f"harmful_subcategory_{ln}")
        results["probe_results"]["harmful_subcategory"] = cat_results
    else:
        print("  Not enough categories with sufficient samples, skipping")
        results["probe_results"]["harmful_subcategory"] = {"skipped": True}

    # ─── Feature 3: Sentiment (known linear control) ───
    print("\n" + "─" * 50)
    print("Feature 3: Sentiment (positive vs negative)")

    n_total_before_sentiment = n_h + n_hl
    n_pos = len(POSITIVE_TEXTS)
    n_neg = len(NEGATIVE_TEXTS)
    X_sent = activations[n_total_before_sentiment:]
    y_sent = np.array([1]*n_pos + [0]*n_neg)

    sent_results = {}
    for li, ln in zip(layer_indices, layer_names):
        X_layer = X_sent[:, li, :]
        sent_results[ln] = train_probes(X_layer, y_sent, f"sentiment_{ln}")
    results["probe_results"]["sentiment"] = sent_results

    # ─── Save activations for visualization ───
    print("\nSaving activations for visualization...")
    np.savez_compressed(
        f"{RESULTS_DIR}/activations.npz",
        activations=activations,
        harmful_cats=np.array(harmful_cats),
        n_harmful=n_h,
        n_harmless=n_hl,
        n_positive=n_pos,
        n_negative=n_neg,
        layer_indices=np.array(layer_indices),
    )

    # ─── Save results ───
    with open(f"{RESULTS_DIR}/experiment1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)

    # Summary
    print("\nNonlinearity Scores Summary (best layer for each feature):")
    for feature_name, feature_results in results["probe_results"].items():
        if isinstance(feature_results, dict) and "skipped" not in feature_results:
            best_layer = max(feature_results.keys(),
                           key=lambda k: feature_results[k]["nonlinearity_score"])
            best = feature_results[best_layer]
            print(f"  {feature_name}: {best['nonlinearity_score']:.4f} "
                  f"(linear={best['linear_acc_mean']:.3f}, nonlinear={best['nonlinear_acc_mean']:.3f}) "
                  f"@ {best_layer}")

    return results


if __name__ == "__main__":
    results = run_experiment()
