from __future__ import annotations

"""Ecological response curve validation via partial dependence plots.

Generates response curves showing how predicted suitability varies
with each environmental variable while holding others at their median.
Flags ecologically implausible patterns (e.g., monotonic responses
across full range) that may indicate extrapolation risk.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_partial_dependence(
    model,
    X: np.ndarray,
    feature_idx: int,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute partial dependence for a single feature.

    Varies the target feature across its range while holding all other
    features at their median values.

    Returns (feature_values, predicted_suitability).
    """
    medians = np.nanmedian(X, axis=0)
    feature_min = np.nanmin(X[:, feature_idx])
    feature_max = np.nanmax(X[:, feature_idx])

    feature_values = np.linspace(feature_min, feature_max, n_points)
    predictions = np.zeros(n_points)

    for i, val in enumerate(feature_values):
        X_synthetic = np.tile(medians, (1, 1))
        X_synthetic[0, feature_idx] = val

        # Handle both sklearn Pipeline and elapid MaxentModel
        if hasattr(model, "predict_proba"):
            predictions[i] = model.predict_proba(X_synthetic)[:, 1][0]
        elif hasattr(model, "predict"):
            pred = model.predict(X_synthetic)
            predictions[i] = pred[0] if hasattr(pred, "__len__") else pred
        else:
            predictions[i] = np.nan

    return feature_values, predictions


def check_monotonicity(predictions: np.ndarray) -> str | None:
    """Check if response is monotonically increasing or decreasing.

    Returns warning string if monotonic, None otherwise.
    """
    diffs = np.diff(predictions)
    if np.all(diffs >= -1e-6):
        return "monotonically increasing"
    elif np.all(diffs <= 1e-6):
        return "monotonically decreasing"
    return None


def generate_response_curves(
    model,
    X: np.ndarray,
    feature_cols: list[str],
    output_dir: Path,
    top_n: int = 10,
    importance: pd.DataFrame | None = None,
) -> list[dict]:
    """Generate partial dependence plots for top features.

    Args:
        model: Fitted model (sklearn Pipeline or elapid MaxentModel)
        X: Training feature matrix
        feature_cols: Feature column names
        output_dir: Directory to save plots
        top_n: Number of top features to plot
        importance: Variable importance DataFrame (to select top features)

    Returns:
        List of dicts with feature name, monotonicity warnings, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select top features by importance, or use first N
    if importance is not None and len(importance) > 0:
        # Only keep features that are in feature_cols (handle expanded features)
        top_features = [
            f for f in importance["feature"].values
            if f in feature_cols
        ][:top_n]
    else:
        top_features = feature_cols[:top_n]

    results = []
    fig, axes = plt.subplots(
        nrows=(len(top_features) + 1) // 2,
        ncols=2,
        figsize=(14, 3 * ((len(top_features) + 1) // 2)),
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, feature_name in enumerate(top_features):
        if feature_name not in feature_cols:
            continue
        feature_idx = feature_cols.index(feature_name)

        logger.info(f"  Computing partial dependence for {feature_name}")
        values, predictions = compute_partial_dependence(model, X, feature_idx)

        # Check for ecological plausibility
        mono_warning = check_monotonicity(predictions)
        result = {
            "feature": feature_name,
            "min_suitability": float(np.min(predictions)),
            "max_suitability": float(np.max(predictions)),
        }
        if mono_warning:
            result["warning"] = f"{mono_warning} across full range (extrapolation risk)"
            logger.warning(f"    WARNING: {feature_name} is {mono_warning}")

        results.append(result)

        # Plot
        if i < len(axes):
            ax = axes[i]
            ax.plot(values, predictions, color="#4ecca3", linewidth=2)
            ax.fill_between(values, predictions, alpha=0.15, color="#4ecca3")

            # Rug plot showing data distribution
            feature_data = X[:, feature_idx]
            feature_data = feature_data[~np.isnan(feature_data)]
            ax.plot(
                feature_data,
                np.full_like(feature_data, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.02),
                "|", color="gray", alpha=0.3, markersize=4,
            )

            ax.set_xlabel(feature_name)
            ax.set_ylabel("Suitability")
            ax.set_title(feature_name)
            if mono_warning:
                ax.set_title(f"{feature_name} ⚠", color="orange")
            ax.set_ylim(-0.05, 1.05)

    # Hide unused axes
    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / "response_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Response curves saved to {plot_path}")

    # Save summary
    summary_path = output_dir / "response_curve_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
