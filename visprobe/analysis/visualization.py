"""
Visualization utilities for analysis results.

This module provides plotting functions for various analysis outputs.
"""

from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from .detailed_evaluation import DetailedResults
from .confidence import ConfidenceProfile


def plot_accuracy_curves(
    severities: np.ndarray,
    results_dict: Dict[str, np.ndarray],
    title: str = "Accuracy vs Perturbation Severity",
    xlabel: str = "Perturbation Severity",
    ylabel: str = "Accuracy (%)",
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot accuracy curves for multiple models.

    Args:
        severities: Array of severity levels
        results_dict: Dictionary mapping model names to accuracy arrays
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Optional color mapping for models
        markers: Optional marker mapping for models
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_accuracy_curves(
        ...     noise_levels,
        ...     {'Vanilla': vanilla_accs, 'Robust': robust_accs}
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (model_name, accuracies) in enumerate(results_dict.items()):
        color = colors.get(model_name) if colors else default_colors[i]
        marker = markers.get(model_name) if markers else default_markers[i % len(default_markers)]

        # Convert to percentage if needed
        if accuracies.max() <= 1.0:
            accuracies = accuracies * 100

        ax.plot(severities, accuracies,
               label=model_name,
               color=color,
               marker=marker,
               markersize=8,
               linewidth=2.5,
               alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-5, 105])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confidence_distribution(
    profile: ConfidenceProfile,
    title: str = "Confidence Distribution",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confidence distribution analysis.

    Args:
        profile: Confidence profile from confidence_profile()
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> profile = confidence_profile(results.samples)
        >>> fig = plot_confidence_distribution(profile)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram of confidences
    ax1 = axes[0]
    bins = np.array(profile.confidence_histogram['bins'])
    width = bins[1] - bins[0]
    x = bins[:-1] + width/2

    ax1.bar(x, profile.confidence_histogram['correct'],
           width=width*0.4, label='Correct', color='green', alpha=0.7)
    ax1.bar(x + width*0.4, profile.confidence_histogram['incorrect'],
           width=width*0.4, label='Incorrect', color='red', alpha=0.7)

    ax1.set_xlabel('Confidence', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mean confidence comparison
    ax2 = axes[1]
    categories = ['Overall', 'Correct', 'Incorrect']
    means = [profile.mean_confidence,
            profile.mean_confidence_correct,
            profile.mean_confidence_incorrect]
    colors_bar = ['blue', 'green', 'red']

    bars = ax2.bar(categories, means, color=colors_bar, alpha=0.7)
    ax2.set_ylabel('Mean Confidence', fontsize=11)
    ax2.set_title('Average Confidence by Correctness', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}', ha='center', va='bottom')

    # Key metrics
    ax3 = axes[2]
    ax3.axis('off')

    metrics_text = f"""Key Metrics:

• Calibration Error: {profile.calibration_error:.3f}
• High-Conf Errors: {profile.pct_high_confidence_errors:.1f}%
• Low-Conf Correct: {profile.pct_low_confidence_correct:.1f}%
• Overconfidence: {profile.overconfidence_score:.3f}
• Underconfidence: {profile.underconfidence_score:.3f}
"""

    ax3.text(0.1, 0.5, metrics_text, fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_class_vulnerabilities(
    vulnerabilities: List[Any],
    top_k: int = 10,
    title: str = "Most Vulnerable Classes",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class vulnerability analysis.

    Args:
        vulnerabilities: List of ClassVulnerability objects
        top_k: Number of top classes to show
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> vuln = class_vulnerability(clean, noisy, top_k=10)
        >>> fig = plot_class_vulnerabilities(vuln)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Take top-k most vulnerable
    plot_data = vulnerabilities[:min(top_k, len(vulnerabilities))]

    # Prepare data
    class_names = []
    clean_accs = []
    perturbed_accs = []
    drops = []

    for vuln in plot_data:
        name = vuln.class_name if vuln.class_name else f"Class {vuln.class_id}"
        class_names.append(f"{name}\n(n={vuln.n_samples})")
        clean_accs.append(vuln.clean_accuracy * 100)
        perturbed_accs.append(vuln.perturbed_accuracy * 100)
        drops.append(vuln.accuracy_drop * 100)

    x = np.arange(len(class_names))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, clean_accs, width,
                  label='Clean', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, perturbed_accs, width,
                  label='Perturbed', color='red', alpha=0.7)

    # Add drop annotations
    for i, (c, p, d) in enumerate(zip(clean_accs, perturbed_accs, drops)):
        ax.annotate(f'-{d:.1f}%',
                   xy=(i, p), xytext=(i, (c + p) / 2),
                   ha='center', fontsize=9, color='darkred',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_bootstrap_comparison(
    model_names: List[str],
    accuracies: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    title: str = "Model Comparison with 95% Confidence Intervals",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model comparison with bootstrap confidence intervals.

    Args:
        model_names: List of model names
        accuracies: List of mean accuracies
        lower_bounds: List of lower confidence bounds
        upper_bounds: List of upper confidence bounds
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_bootstrap_comparison(
        ...     ['Model A', 'Model B'],
        ...     [0.82, 0.79],
        ...     [0.80, 0.76],
        ...     [0.84, 0.82]
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(model_names))

    # Convert to percentages if needed
    if max(accuracies) <= 1.0:
        accuracies = [a * 100 for a in accuracies]
        lower_bounds = [l * 100 for l in lower_bounds]
        upper_bounds = [u * 100 for u in upper_bounds]

    # Plot bars with error bars
    bars = ax.bar(x, accuracies, color='steelblue', alpha=0.7)

    # Add error bars
    errors = [[accuracies[i] - lower_bounds[i] for i in range(len(accuracies))],
             [upper_bounds[i] - accuracies[i] for i in range(len(accuracies))]]

    ax.errorbar(x, accuracies, yerr=errors, fmt='none',
               color='black', capsize=5, capthick=2, elinewidth=2)

    # Add value labels
    for i, (bar, acc, lower, upper) in enumerate(zip(bars, accuracies, lower_bounds, upper_bounds)):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 1,
               f'{acc:.1f}%\n[{lower:.1f}, {upper:.1f}]',
               ha='center', va='bottom', fontsize=10)

    # Check for significant differences
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            # Check if confidence intervals overlap
            if lower_bounds[i] > upper_bounds[j] or lower_bounds[j] > upper_bounds[i]:
                # Significant difference
                y_pos = max(upper_bounds[i], upper_bounds[j]) + 3
                x_pos = (i + j) / 2
                ax.text(x_pos, y_pos, '***', ha='center', fontsize=12, color='red')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(upper_bounds) + 10])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_disagreement_heatmap(
    disagreement_matrix: np.ndarray,
    model_names: List[str],
    title: str = "Pairwise Model Disagreement",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of pairwise model disagreements.

    Args:
        disagreement_matrix: Square matrix of disagreement rates
        model_names: List of model names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> matrix = pairwise_disagreement_matrix([results1, results2, results3])
        >>> fig = plot_disagreement_heatmap(matrix, ['Model1', 'Model2', 'Model3'])
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(disagreement_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_yticklabels(model_names)

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Disagreement Rate', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            if i != j:
                text = ax.text(j, i, f'{disagreement_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if disagreement_matrix[i, j] < 0.5 else "white")

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig