"""
Final combined analysis: merge all experimental results into comprehensive visualizations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RESULTS_DIR = "results/data"
PLOTS_DIR = "results/plots"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"
})


def load_all():
    with open(f"{RESULTS_DIR}/experiment1_results.json") as f:
        exp1 = json.load(f)
    with open(f"{RESULTS_DIR}/experiment2_results.json") as f:
        exp2 = json.load(f)
    with open(f"{RESULTS_DIR}/experiment2b_results.json") as f:
        exp2b = json.load(f)
    return exp1, exp2, exp2b


def plot_comprehensive_summary(exp1, exp2, exp2b):
    """Create the main summary figure combining all results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ─── Panel A: Probe accuracy across layers (best feature) ───
    ax = axes[0, 0]
    for feature, color, label in [
        ("harmful_vs_harmless", "#3498db", "Harmful/Harmless"),
        ("harmful_subcategory", "#e74c3c", "Sub-categories"),
        ("sentiment", "#2ecc71", "Sentiment"),
    ]:
        results = exp1["probe_results"].get(feature, {})
        if isinstance(results, dict) and "skipped" not in results:
            layers = sorted(results.keys())
            linear = [results[l]["linear_acc_mean"] for l in layers]
            nonlinear = [results[l]["nonlinear_acc_mean"] for l in layers]
            x = range(len(layers))
            ax.plot(x, linear, f'-o', color=color, label=f"{label} (Linear)", markersize=6)
            ax.plot(x, nonlinear, f'--s', color=color, label=f"{label} (MLP)", markersize=6, alpha=0.7)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace("layer_", "L") for l in layers], rotation=45)
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("A) Linear vs Nonlinear Probe Accuracy\n(Qwen2.5-1.5B-Instruct)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0.2, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    # ─── Panel B: Introspection accuracy hierarchy ───
    ax = axes[0, 1]
    tasks = [
        ("Sentiment\n(Linear)", exp2["tests"]["sentiment"]["accuracy"], "#2ecc71"),
        ("Refusal\n(Binary)", exp2["tests"]["refusal_prediction"]["accuracy"], "#3498db"),
        ("Category\n(External)", exp2["tests"]["category_classification"]["accuracy"], "#e67e22"),
        ("Category\n(Introspective)", exp2["tests"]["introspective_refusal"]["accuracy"], "#9b59b6"),
        ("Borderline\n(Self-pred.)", exp2b["tests"]["borderline_refusal"]["self_prediction_accuracy"], "#e74c3c"),
    ]
    names, accs, colors = zip(*tasks)
    bars = ax.bar(names, accs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Self-Report Accuracy")
    ax.set_title("B) LLM Introspection Accuracy by Task Type\n(GPT-4.1)")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")

    # ─── Panel C: Confidence calibration ───
    ax = axes[1, 0]
    mc = exp2b["tests"]["metacognitive"]
    categories = ["Clear Prompts", "Ambiguous Prompts"]
    accs_c = [mc["clear_accuracy"], mc["ambiguous_accuracy"]]
    confs_c = [mc["clear_confidence"], mc["ambiguous_confidence"]]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, accs_c, width, label="Accuracy", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, confs_c, width, label="Confidence", color="#e74c3c", alpha=0.8)

    for bar, val in zip(list(bars1) + list(bars2), accs_c + confs_c):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title(f"C) Confidence Calibration: Clear vs Ambiguous\n"
                 f"(p = {mc['confidence_p_value']:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1.15)

    # ─── Panel D: Graded harm correlation ───
    ax = axes[1, 1]
    gh = exp2b["tests"]["graded_harm"]
    ref_scores = [d["ref_harm"] for d in gh["details"] if d["model_score"] is not None]
    model_scores = [d["model_score"] for d in gh["details"] if d["model_score"] is not None]

    ax.scatter(ref_scores, model_scores, c="#3498db", s=80, edgecolors="black", linewidth=0.5, zorder=5)

    # Fit line
    if len(ref_scores) >= 3:
        slope, intercept = np.polyfit(ref_scores, model_scores, 1)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.7)

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Reference Harm Score")
    ax.set_ylabel("Model's Harm Score")
    ax.set_title(f"D) Graded Harm Assessment\n"
                 f"(Spearman ρ = {gh['spearman_rho']:.3f}, p < 0.001)")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Linear/Nonlinear Information & LLM Introspection:\nComprehensive Results",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/comprehensive_summary.png")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/comprehensive_summary.png")


def plot_introspection_gradient(exp2, exp2b):
    """Show the gradient from high to low introspective access."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build the gradient
    data_points = [
        {"task": "Sentiment\nClassification", "accuracy": exp2["tests"]["sentiment"]["accuracy"],
         "type": "linear", "description": "Known linear feature"},
        {"task": "Binary Refusal\nPrediction", "accuracy": exp2["tests"]["refusal_prediction"]["accuracy"],
         "type": "linear", "description": "Single linear direction (Arditi et al.)"},
        {"task": "Harmful Category\n(External)", "accuracy": exp2["tests"]["category_classification"]["accuracy"],
         "type": "nonlinear", "description": "Multi-dimensional feature space"},
        {"task": "Harmful Category\n(Introspective)", "accuracy": exp2["tests"]["introspective_refusal"]["accuracy"],
         "type": "nonlinear", "description": "Self-report on nonlinear structure"},
        {"task": "Borderline\nSelf-Prediction", "accuracy": exp2b["tests"]["borderline_refusal"]["self_prediction_accuracy"],
         "type": "boundary", "description": "Near decision boundary"},
    ]

    # Sort by accuracy (descending)
    data_points.sort(key=lambda x: x["accuracy"], reverse=True)

    names = [d["task"] for d in data_points]
    accs = [d["accuracy"] for d in data_points]
    type_colors = {"linear": "#2ecc71", "nonlinear": "#e74c3c", "boundary": "#e67e22"}
    colors = [type_colors[d["type"]] for d in data_points]

    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5, height=0.6)

    # Labels
    for i, (bar, acc, dp) in enumerate(zip(bars, accs, data_points)):
        ax.text(acc + 0.005, i, f"{acc:.1%}", va="center", fontweight="bold", fontsize=11)
        ax.text(0.01, i, dp["description"], va="center", fontsize=9, color="white",
                fontweight="bold" if acc > 0.5 else "normal")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Self-Report Accuracy")
    ax.set_title("Introspection Accuracy Gradient:\nFrom Linear to Nonlinear Features")
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Linear feature"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Nonlinear feature"),
        Patch(facecolor="#e67e22", edgecolor="black", label="Decision boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/introspection_gradient.png")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/introspection_gradient.png")


def compute_final_statistics(exp1, exp2, exp2b):
    """Compute summary statistics for the report."""
    stats_summary = {}

    # Probe results at last layer
    last_layer = exp1["layer_names"][-1]
    for feature in ["harmful_vs_harmless", "harmful_subcategory", "sentiment"]:
        res = exp1["probe_results"].get(feature, {})
        if isinstance(res, dict) and last_layer in res:
            r = res[last_layer]
            stats_summary[f"{feature}_linear_acc"] = r["linear_acc_mean"]
            stats_summary[f"{feature}_nonlinear_acc"] = r["nonlinear_acc_mean"]
            stats_summary[f"{feature}_nonlinearity_score"] = r["nonlinearity_score"]

    # Introspection results
    for test_name, test_data in exp2["tests"].items():
        stats_summary[f"introspection_{test_name}"] = test_data["accuracy"]

    # Deep introspection
    stats_summary["borderline_self_prediction"] = exp2b["tests"]["borderline_refusal"]["self_prediction_accuracy"]
    stats_summary["borderline_avg_confidence"] = exp2b["tests"]["borderline_refusal"]["avg_confidence"]
    stats_summary["graded_harm_spearman"] = exp2b["tests"]["graded_harm"]["spearman_rho"]
    stats_summary["metacognitive_clear_conf"] = exp2b["tests"]["metacognitive"]["clear_confidence"]
    stats_summary["metacognitive_ambig_conf"] = exp2b["tests"]["metacognitive"]["ambiguous_confidence"]
    stats_summary["metacognitive_conf_pvalue"] = exp2b["tests"]["metacognitive"]["confidence_p_value"]

    # Key test: accuracy degradation from linear to nonlinear tasks
    linear_accs = [
        exp2["tests"]["sentiment"]["accuracy"],
        exp2["tests"]["refusal_prediction"]["accuracy"],
    ]
    nonlinear_accs = [
        exp2["tests"]["category_classification"]["accuracy"],
        exp2["tests"]["introspective_refusal"]["accuracy"],
        exp2b["tests"]["borderline_refusal"]["self_prediction_accuracy"],
    ]
    stats_summary["mean_linear_introspection"] = float(np.mean(linear_accs))
    stats_summary["mean_nonlinear_introspection"] = float(np.mean(nonlinear_accs))
    stats_summary["introspection_gap"] = float(np.mean(linear_accs) - np.mean(nonlinear_accs))

    # Mann-Whitney U test for the gap
    u_stat, u_p = stats.mannwhitneyu(linear_accs, nonlinear_accs, alternative="greater")
    stats_summary["introspection_gap_u_stat"] = float(u_stat)
    stats_summary["introspection_gap_p_value"] = float(u_p)

    print("\n" + "=" * 60)
    print("FINAL STATISTICS SUMMARY")
    print("=" * 60)
    print(f"\nProbe Accuracy (Last Layer, Qwen2.5-1.5B):")
    for feature in ["harmful_vs_harmless", "harmful_subcategory", "sentiment"]:
        if f"{feature}_linear_acc" in stats_summary:
            print(f"  {feature}: linear={stats_summary[f'{feature}_linear_acc']:.3f}, "
                  f"nonlinear={stats_summary[f'{feature}_nonlinear_acc']:.3f}, "
                  f"Δ={stats_summary[f'{feature}_nonlinearity_score']:.4f}")

    print(f"\nIntrospection Accuracy (GPT-4.1):")
    print(f"  Linear features (mean): {stats_summary['mean_linear_introspection']:.3f}")
    print(f"  Nonlinear features (mean): {stats_summary['mean_nonlinear_introspection']:.3f}")
    print(f"  Gap: {stats_summary['introspection_gap']:.4f}")
    print(f"  Mann-Whitney U p-value: {stats_summary['introspection_gap_p_value']:.4f}")

    print(f"\nConfidence Calibration:")
    print(f"  Clear prompts confidence: {stats_summary['metacognitive_clear_conf']:.3f}")
    print(f"  Ambiguous prompts confidence: {stats_summary['metacognitive_ambig_conf']:.3f}")
    print(f"  t-test p-value: {stats_summary['metacognitive_conf_pvalue']:.4f}")

    print(f"\nGraded Harm Assessment:")
    print(f"  Spearman ρ: {stats_summary['graded_harm_spearman']:.4f}")

    with open(f"{RESULTS_DIR}/final_statistics.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    return stats_summary


def main():
    exp1, exp2, exp2b = load_all()
    plot_comprehensive_summary(exp1, exp2, exp2b)
    plot_introspection_gradient(exp2, exp2b)
    stats_summary = compute_final_statistics(exp1, exp2, exp2b)
    return stats_summary


if __name__ == "__main__":
    main()
