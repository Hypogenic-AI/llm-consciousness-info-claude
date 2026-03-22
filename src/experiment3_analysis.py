"""
Experiment 3: Analysis and Visualization

Combines results from Experiments 1 and 2 to:
1. Visualize activation space with PCA and UMAP
2. Correlate feature nonlinearity with introspection accuracy
3. Generate publication-quality figures
4. Run statistical tests
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results/data"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight"
})


def load_results():
    """Load results from experiments 1 and 2."""
    with open(f"{RESULTS_DIR}/experiment1_results.json") as f:
        exp1 = json.load(f)
    with open(f"{RESULTS_DIR}/experiment2_results.json") as f:
        exp2 = json.load(f)
    data = np.load(f"{RESULTS_DIR}/activations.npz", allow_pickle=True)
    return exp1, exp2, data


def plot_activation_space(data, exp1):
    """Visualize activation space with PCA and UMAP."""
    print("\nGenerating activation space visualizations...")

    activations = data["activations"]
    n_harmful = int(data["n_harmful"])
    n_harmless = int(data["n_harmless"])
    n_positive = int(data["n_positive"])
    n_negative = int(data["n_negative"])
    harmful_cats = data["harmful_cats"]
    layer_indices = data["layer_indices"]

    # Use the last layer for visualization
    last_layer_idx = layer_indices[-1]
    X = activations[:, last_layer_idx, :]

    # Labels
    labels_binary = (
        ["Harmful"] * n_harmful +
        ["Harmless"] * n_harmless +
        ["Positive"] * n_positive +
        ["Negative"] * n_negative
    )

    # ─── PCA (linear) visualization ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    colors = {"Harmful": "#e74c3c", "Harmless": "#2ecc71", "Positive": "#3498db", "Negative": "#9b59b6"}
    for label in ["Harmful", "Harmless", "Positive", "Negative"]:
        mask = [i for i, l in enumerate(labels_binary) if l == label]
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[label],
                       label=label, alpha=0.5, s=20)
    axes[0].set_title(f"PCA (Linear) - Last Layer\nVariance explained: {pca.explained_variance_ratio_[:2].sum():.2%}")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    # ─── UMAP (nonlinear) visualization ───
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        X_umap = reducer.fit_transform(X_scaled)

        for label in ["Harmful", "Harmless", "Positive", "Negative"]:
            mask = [i for i, l in enumerate(labels_binary) if l == label]
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors[label],
                           label=label, alpha=0.5, s=20)
        axes[1].set_title("UMAP (Nonlinear) - Last Layer")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        axes[1].legend()
    except ImportError:
        axes[1].text(0.5, 0.5, "UMAP not available", ha="center", va="center", transform=axes[1].transAxes)

    plt.suptitle("Activation Space: Linear vs Nonlinear Dimensionality Reduction", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/activation_space_comparison.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/activation_space_comparison.png")

    # ─── Harmful sub-category visualization with UMAP ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    X_harmful = X[:n_harmful]
    X_harmful_scaled = scaler.fit_transform(X_harmful)
    cats = list(harmful_cats)

    # PCA on harmful only
    pca_h = PCA(n_components=2)
    X_h_pca = pca_h.fit_transform(X_harmful_scaled)

    unique_cats = list(set(cats))
    cat_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cats)))
    cat_color_map = dict(zip(unique_cats, cat_colors))

    for cat in unique_cats:
        mask = [i for i, c in enumerate(cats) if c == cat]
        axes[0].scatter(X_h_pca[mask, 0], X_h_pca[mask, 1], c=[cat_color_map[cat]],
                       label=cat, alpha=0.6, s=30)
    axes[0].set_title("PCA (Linear) - Harmful Sub-categories")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=9)

    # UMAP on harmful only
    try:
        reducer_h = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.2)
        X_h_umap = reducer_h.fit_transform(X_harmful_scaled)
        for cat in unique_cats:
            mask = [i for i, c in enumerate(cats) if c == cat]
            axes[1].scatter(X_h_umap[mask, 0], X_h_umap[mask, 1], c=[cat_color_map[cat]],
                           label=cat, alpha=0.6, s=30)
        axes[1].set_title("UMAP (Nonlinear) - Harmful Sub-categories")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        axes[1].legend(fontsize=9)
    except Exception:
        axes[1].text(0.5, 0.5, "UMAP failed", ha="center", va="center", transform=axes[1].transAxes)

    plt.suptitle("Harmful Content Sub-categories: Linear vs Nonlinear Separation", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/harmful_subcategories.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/harmful_subcategories.png")


def plot_probe_comparison(exp1):
    """Plot linear vs nonlinear probe accuracy across features and layers."""
    print("\nGenerating probe comparison plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    feature_names = ["harmful_vs_harmless", "harmful_subcategory", "sentiment"]
    titles = ["Harmful vs Harmless\n(Binary Refusal)", "Harmful Sub-categories\n(Multi-class)", "Sentiment\n(Positive vs Negative)"]

    for idx, (feature, title) in enumerate(zip(feature_names, titles)):
        results = exp1["probe_results"].get(feature, {})
        if isinstance(results, dict) and "skipped" in results:
            axes[idx].text(0.5, 0.5, "Skipped (insufficient data)",
                          ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].set_title(title)
            continue

        layers = []
        linear_accs = []
        nonlinear_accs = []
        linear_stds = []
        nonlinear_stds = []

        for layer_name, layer_results in sorted(results.items()):
            layers.append(layer_name.replace("layer_", "L"))
            linear_accs.append(layer_results["linear_acc_mean"])
            nonlinear_accs.append(layer_results["nonlinear_acc_mean"])
            linear_stds.append(layer_results["linear_acc_std"])
            nonlinear_stds.append(layer_results["nonlinear_acc_std"])

        x = np.arange(len(layers))
        width = 0.35

        bars1 = axes[idx].bar(x - width/2, linear_accs, width, yerr=linear_stds,
                              label="Linear Probe", color="#3498db", alpha=0.8, capsize=3)
        bars2 = axes[idx].bar(x + width/2, nonlinear_accs, width, yerr=nonlinear_stds,
                              label="Nonlinear Probe", color="#e74c3c", alpha=0.8, capsize=3)

        axes[idx].set_xlabel("Layer")
        axes[idx].set_ylabel("Accuracy")
        axes[idx].set_title(title)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(layers, rotation=45)
        axes[idx].legend()
        axes[idx].set_ylim(0, 1.05)

        # Add nonlinearity score annotation
        best_layer = max(results.keys(), key=lambda k: results[k]["nonlinearity_score"])
        ns = results[best_layer]["nonlinearity_score"]
        axes[idx].annotate(f"Max Δ = {ns:.3f}", xy=(0.5, 0.02),
                          xycoords="axes fraction", ha="center", fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

    plt.suptitle("Linear vs Nonlinear Probe Accuracy by Feature Type", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/probe_comparison.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/probe_comparison.png")


def plot_introspection_results(exp2):
    """Plot introspection test results."""
    print("\nGenerating introspection results plots...")

    tests = exp2["tests"]
    test_names = {
        "sentiment": "Sentiment\n(Linear Control)",
        "refusal_prediction": "Refusal Prediction\n(Binary)",
        "category_classification": "Category\n(External View)",
        "introspective_refusal": "Category\n(Self-Introspection)"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    accs = []
    colors = []
    color_map = {
        "sentiment": "#2ecc71",  # green - linear
        "refusal_prediction": "#3498db",  # blue - linear
        "category_classification": "#e74c3c",  # red - nonlinear
        "introspective_refusal": "#9b59b6",  # purple - nonlinear + introspective
    }

    for key, name in test_names.items():
        if key in tests:
            names.append(name)
            accs.append(tests[key]["accuracy"])
            colors.append(color_map[key])

    bars = ax.bar(names, accs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add accuracy labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Accuracy")
    ax.set_title("LLM Self-Report Accuracy by Feature Type\n(Green/Blue = Linear Features, Red/Purple = Nonlinear Features)")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance level")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/introspection_accuracy.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/introspection_accuracy.png")


def plot_nonlinearity_vs_introspection(exp1, exp2):
    """Plot the key result: nonlinearity score vs introspection accuracy."""
    print("\nGenerating nonlinearity vs introspection plot...")

    # Collect data points
    points = []

    # Sentiment: low nonlinearity, high introspection accuracy
    sent_results = exp1["probe_results"].get("sentiment", {})
    if sent_results and "skipped" not in sent_results:
        best_layer = max(sent_results.keys(), key=lambda k: abs(sent_results[k]["nonlinearity_score"]))
        ns = sent_results[best_layer]["nonlinearity_score"]
        introspection_acc = exp2["tests"]["sentiment"]["accuracy"]
        points.append(("Sentiment", ns, introspection_acc))

    # Binary refusal: expected moderate nonlinearity
    binary_results = exp1["probe_results"].get("harmful_vs_harmless", {})
    if binary_results:
        best_layer = max(binary_results.keys(), key=lambda k: abs(binary_results[k]["nonlinearity_score"]))
        ns = binary_results[best_layer]["nonlinearity_score"]
        introspection_acc = exp2["tests"]["refusal_prediction"]["accuracy"]
        points.append(("Refusal (binary)", ns, introspection_acc))

    # Sub-category: expected high nonlinearity
    cat_results = exp1["probe_results"].get("harmful_subcategory", {})
    if cat_results and "skipped" not in cat_results:
        best_layer = max(cat_results.keys(), key=lambda k: abs(cat_results[k]["nonlinearity_score"]))
        ns = cat_results[best_layer]["nonlinearity_score"]
        # Use introspective refusal accuracy for the most direct comparison
        introspection_acc = exp2["tests"]["introspective_refusal"]["accuracy"]
        points.append(("Harmful Sub-type\n(Introspective)", ns, introspection_acc))

        # Also add external classification
        ext_acc = exp2["tests"]["category_classification"]["accuracy"]
        points.append(("Harmful Sub-type\n(External)", ns, ext_acc))

    if len(points) < 2:
        print("  Not enough data points for correlation plot")
        return {}

    fig, ax = plt.subplots(figsize=(8, 6))

    labels, ns_vals, intro_vals = zip(*points)
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"][:len(points)]

    for i, (label, ns, intro) in enumerate(points):
        ax.scatter(ns, intro, c=colors[i], s=150, zorder=5, edgecolors="black", linewidth=1)
        ax.annotate(label, (ns, intro), textcoords="offset points",
                   xytext=(10, 10), fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.2))

    # Fit line if we have enough points
    if len(points) >= 3:
        ns_arr = np.array(ns_vals)
        intro_arr = np.array(intro_vals)
        slope, intercept, r_value, p_value, std_err = stats.linregress(ns_arr, intro_arr)
        x_line = np.linspace(min(ns_arr) - 0.02, max(ns_arr) + 0.02, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.5, label=f"r = {r_value:.3f}, p = {p_value:.3f}")
        ax.legend(fontsize=11)

        # Spearman correlation
        rho, p_spearman = stats.spearmanr(ns_arr, intro_arr)
        print(f"  Pearson: r={r_value:.3f}, p={p_value:.3f}")
        print(f"  Spearman: ρ={rho:.3f}, p={p_spearman:.3f}")
    else:
        r_value = p_value = rho = p_spearman = None

    ax.set_xlabel("Nonlinearity Score\n(Nonlinear Probe Acc - Linear Probe Acc)", fontsize=12)
    ax.set_ylabel("LLM Self-Report Accuracy", fontsize=12)
    ax.set_title("Feature Nonlinearity vs LLM Introspection Accuracy\n"
                 "(Higher nonlinearity → lower self-report accuracy?)", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/nonlinearity_vs_introspection.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/nonlinearity_vs_introspection.png")

    return {
        "points": [{"label": l, "nonlinearity_score": float(n), "introspection_acc": float(a)}
                   for l, n, a in points],
        "pearson_r": float(r_value) if r_value is not None else None,
        "pearson_p": float(p_value) if p_value is not None else None,
        "spearman_rho": float(rho) if rho is not None else None,
        "spearman_p": float(p_spearman) if p_spearman is not None else None,
    }


def plot_confidence_analysis(exp2):
    """Analyze confidence calibration across tasks."""
    print("\nGenerating confidence analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect confidence by correctness for each test
    test_keys = ["refusal_prediction", "category_classification", "sentiment", "introspective_refusal"]
    test_labels = ["Refusal Pred.", "Category Class.", "Sentiment", "Introspective"]
    correct_confs = []
    incorrect_confs = []
    labels = []

    for key, label in zip(test_keys, test_labels):
        if key not in exp2["tests"]:
            continue
        details = exp2["tests"][key]["details"]
        cc = [d["confidence"] for d in details if d.get("correct") == True and d.get("confidence") is not None]
        ic = [d["confidence"] for d in details if d.get("correct") == False and d.get("confidence") is not None]
        if cc or ic:
            correct_confs.append(cc)
            incorrect_confs.append(ic)
            labels.append(label)

    if labels:
        # Box plot of confidence by correctness
        all_data = []
        all_labels_plot = []
        all_correct = []
        for i, label in enumerate(labels):
            for c in correct_confs[i]:
                all_data.append(c)
                all_labels_plot.append(label)
                all_correct.append("Correct")
            for c in incorrect_confs[i]:
                all_data.append(c)
                all_labels_plot.append(label)
                all_correct.append("Incorrect")

        import pandas as pd
        df = pd.DataFrame({"Confidence": all_data, "Task": all_labels_plot, "Correctness": all_correct})
        sns.boxplot(data=df, x="Task", y="Confidence", hue="Correctness", ax=axes[0], palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"})
        axes[0].set_title("Confidence by Correctness per Task")
        axes[0].set_ylim(0, 1.05)

        # Average confidence per task
        avg_confs = []
        for i, label in enumerate(labels):
            all_c = correct_confs[i] + incorrect_confs[i]
            avg_confs.append(np.mean(all_c) if all_c else 0)
        axes[1].bar(labels, avg_confs, color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"][:len(labels)], alpha=0.8)
        axes[1].set_title("Average Confidence by Task")
        axes[1].set_ylabel("Mean Confidence")
        axes[1].set_ylim(0, 1.05)
    else:
        axes[0].text(0.5, 0.5, "No confidence data", ha="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No confidence data", ha="center", transform=axes[1].transAxes)

    plt.suptitle("Confidence Calibration Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confidence_analysis.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/confidence_analysis.png")


def compute_statistical_tests(exp1, exp2):
    """Run statistical tests to validate findings."""
    print("\nRunning statistical tests...")

    stats_results = {}

    # Test 1: Is nonlinear probe significantly better than linear for each feature?
    for feature in ["harmful_vs_harmless", "harmful_subcategory", "sentiment"]:
        results = exp1["probe_results"].get(feature, {})
        if isinstance(results, dict) and "skipped" not in results:
            # Get the best layer (latest layer usually best)
            best_layer = list(results.keys())[-1]
            lr = results[best_layer]

            linear_accs = lr["linear_accs_per_fold"]
            nonlinear_accs = lr["nonlinear_accs_per_fold"]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(nonlinear_accs, linear_accs)
            # Wilcoxon signed-rank (non-parametric alternative)
            try:
                w_stat, w_p = stats.wilcoxon(
                    [n - l for n, l in zip(nonlinear_accs, linear_accs)]
                )
            except ValueError:
                w_stat, w_p = None, None

            # Effect size (Cohen's d for paired samples)
            diff = np.array(nonlinear_accs) - np.array(linear_accs)
            cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

            stats_results[feature] = {
                "best_layer": best_layer,
                "t_statistic": float(t_stat),
                "t_p_value": float(p_val),
                "wilcoxon_stat": float(w_stat) if w_stat else None,
                "wilcoxon_p": float(w_p) if w_p else None,
                "cohens_d": float(cohen_d),
                "mean_difference": float(np.mean(diff)),
            }
            print(f"  {feature} ({best_layer}):")
            print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"    Cohen's d: {cohen_d:.3f}")

    # Test 2: Compare introspection accuracies across tasks
    # Using proportions test (z-test for two proportions)
    tests = exp2["tests"]
    if "sentiment" in tests and "introspective_refusal" in tests:
        n1 = tests["sentiment"]["n_valid"]
        p1 = tests["sentiment"]["accuracy"]
        n2 = tests["introspective_refusal"]["n_valid"]
        p2 = tests["introspective_refusal"]["accuracy"]

        # Two-proportion z-test
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        stats_results["sentiment_vs_introspective"] = {
            "z_statistic": float(z),
            "p_value": float(p_val),
            "sentiment_acc": float(p1),
            "introspective_acc": float(p2),
            "difference": float(p1 - p2),
        }
        print(f"\n  Sentiment vs Introspective accuracy:")
        print(f"    z={z:.3f}, p={p_val:.4f}")
        print(f"    Sentiment: {p1:.3f}, Introspective: {p2:.3f}, Δ={p1-p2:.3f}")

    return stats_results


def run_analysis():
    """Run all analysis and generate plots."""
    print("=" * 60)
    print("EXPERIMENT 3: Analysis and Visualization")
    print("=" * 60)

    exp1, exp2, data = load_results()

    # Generate plots
    plot_activation_space(data, exp1)
    plot_probe_comparison(exp1)
    plot_introspection_results(exp2)
    correlation_results = plot_nonlinearity_vs_introspection(exp1, exp2)
    plot_confidence_analysis(exp2)

    # Statistical tests
    stats_results = compute_statistical_tests(exp1, exp2)

    # Save analysis results
    analysis = {
        "correlation": correlation_results,
        "statistical_tests": stats_results,
        "timestamp": str(np.datetime64("now")),
    }
    with open(f"{RESULTS_DIR}/experiment3_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return analysis


if __name__ == "__main__":
    analysis = run_analysis()
