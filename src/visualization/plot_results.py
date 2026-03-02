from __future__ import annotations

"""Generate publication-quality figures for the turtle habitat model.

Produces:
  1. Suitability map with occurrence overlay
  2. Variable importance bar chart
  3. Response curves for top predictors
  4. Model comparison chart
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# SDM-standard color ramp: blue (unsuitable) → yellow → red (highly suitable)
SUITABILITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "suitability",
    ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
)


def plot_suitability_map(
    suitability_tif: Path,
    occurrence_csv: Path | None = None,
    output_path: Path = Path("results/figures/suitability_map.png"),
    title: str = "Western Pond Turtle — Habitat Suitability",
) -> Path:
    """Render the suitability GeoTIFF as a map with occurrence overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(suitability_tif) as src:
        data = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        transform = src.transform

    # Mask nodata
    masked = np.ma.masked_where((data == nodata) | (data < 0), data)

    fig, ax = plt.subplots(figsize=(12, 8))

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(
        masked,
        extent=extent,
        origin="upper",
        cmap=SUITABILITY_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Overlay occurrences
    if occurrence_csv and occurrence_csv.exists():
        df = pd.read_csv(occurrence_csv)
        presence = df[df["presence"] == 1]
        background = df[df["presence"] == 0]

        # Plot background as small gray dots
        ax.scatter(
            background["longitude"],
            background["latitude"],
            c="gray",
            s=2,
            alpha=0.15,
            label=f"Background ({len(background)})",
            zorder=2,
        )
        # Plot presence as black-edged dots
        ax.scatter(
            presence["longitude"],
            presence["latitude"],
            c="lime",
            s=18,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            label=f"Presence ({len(presence)})",
            zorder=3,
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Habitat Suitability (0–1)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Suitability map saved: {output_path}")
    return output_path


def plot_variable_importance(
    importance_csv: Path,
    output_path: Path = Path("results/figures/variable_importance.png"),
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of variable importance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(importance_csv)

    # Aggregate polynomial features back to original variable names
    # Features like "bio1 bio2" (interaction) or "bio1^2" (quadratic)
    agg = {}
    for _, row in df.iterrows():
        feat = row["feature"]
        imp = row["importance"]
        # Find the base variable name(s)
        parts = feat.replace("^2", "").split()
        for part in parts:
            base = part.strip()
            if base:
                agg[base] = agg.get(base, 0) + imp / len(parts)

    agg_df = pd.DataFrame(
        [{"feature": k, "importance": v} for k, v in agg.items()]
    ).sort_values("importance", ascending=True)

    # Take top N
    agg_df = agg_df.tail(top_n)

    # Bioclim label lookup
    BIOCLIM_LABELS = {
        "bio1": "Annual Mean Temp",
        "bio2": "Mean Diurnal Range",
        "bio3": "Isothermality",
        "bio4": "Temp Seasonality",
        "bio5": "Max Temp Warmest Month",
        "bio6": "Min Temp Coldest Month",
        "bio7": "Temp Annual Range",
        "bio8": "Mean Temp Wettest Qtr",
        "bio9": "Mean Temp Driest Qtr",
        "bio10": "Mean Temp Warmest Qtr",
        "bio11": "Mean Temp Coldest Qtr",
        "bio12": "Annual Precipitation",
        "bio13": "Precip Wettest Month",
        "bio14": "Precip Driest Month",
        "bio15": "Precip Seasonality",
        "bio16": "Precip Wettest Qtr",
        "bio17": "Precip Driest Qtr",
        "bio18": "Precip Warmest Qtr",
        "bio19": "Precip Coldest Qtr",
        "slope": "Slope",
        "aspect": "Aspect",
        "twi": "Topographic Wetness",
    }

    labels = [
        f"{f} ({BIOCLIM_LABELS.get(f, f)})" if f in BIOCLIM_LABELS else f
        for f in agg_df["feature"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(agg_df)))
    ax.barh(labels, agg_df["importance"], color=colors)
    ax.set_xlabel("Aggregated Importance")
    ax.set_title("Variable Importance — Random Forest", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Variable importance chart saved: {output_path}")
    return output_path


def plot_response_curves(
    model_path: Path,
    training_csv: Path,
    output_path: Path = Path("results/figures/response_curves.png"),
    top_n: int = 6,
) -> Path:
    """Partial dependence plots for top N features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.models.maxent import load_model, predict_probability, get_feature_cols

    model, feature_cols = load_model(model_path)
    df = pd.read_csv(training_csv)
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    # Get variable importance to pick top features
    from src.models.maxent import variable_importance
    importance = variable_importance(model, feature_cols)

    # Aggregate to base variables
    agg = {}
    for _, row in importance.iterrows():
        parts = row["feature"].replace("^2", "").split()
        for part in parts:
            base = part.strip()
            if base in feature_cols:
                agg[base] = agg.get(base, 0) + row["importance"] / len(parts)

    top_features = sorted(agg, key=lambda k: agg[k], reverse=True)[:top_n]

    BIOCLIM_LABELS = {
        "bio1": "Annual Mean Temp (°C×10)",
        "bio2": "Mean Diurnal Range (°C×10)",
        "bio3": "Isothermality (%)",
        "bio4": "Temp Seasonality (SD×100)",
        "bio5": "Max Temp Warmest Mo (°C×10)",
        "bio6": "Min Temp Coldest Mo (°C×10)",
        "bio7": "Temp Annual Range (°C×10)",
        "bio8": "Mean Temp Wettest Qtr (°C×10)",
        "bio9": "Mean Temp Driest Qtr (°C×10)",
        "bio10": "Mean Temp Warmest Qtr (°C×10)",
        "bio11": "Mean Temp Coldest Qtr (°C×10)",
        "bio12": "Annual Precipitation (mm)",
        "bio13": "Precip Wettest Mo (mm)",
        "bio14": "Precip Driest Mo (mm)",
        "bio15": "Precip Seasonality (CV)",
        "bio16": "Precip Wettest Qtr (mm)",
        "bio17": "Precip Driest Qtr (mm)",
        "bio18": "Precip Warmest Qtr (mm)",
        "bio19": "Precip Coldest Qtr (mm)",
    }

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        feat_idx = feature_cols.index(feat)

        # Create grid across the feature range
        feat_vals = X[:, feat_idx]
        grid = np.linspace(np.percentile(feat_vals, 2), np.percentile(feat_vals, 98), 100)

        # Hold all other features at median
        X_partial = np.tile(np.nanmedian(X, axis=0), (100, 1))
        X_partial[:, feat_idx] = grid

        probs = predict_probability(model, X_partial)

        label = BIOCLIM_LABELS.get(feat, feat)
        ax.plot(grid, probs, color="#b2182b", linewidth=2)
        ax.fill_between(grid, 0, probs, alpha=0.15, color="#b2182b")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Suitability")
        ax.set_ylim(0, 1)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add rug plot of presence data
        presence_mask = df["presence"] == 1
        pres_vals = df.loc[presence_mask, feat].dropna()
        ax.plot(
            pres_vals, np.zeros_like(pres_vals) - 0.02,
            "|", color="black", alpha=0.3, markersize=4,
        )

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Response Curves — Top Predictors (partial dependence)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Response curves saved: {output_path}")
    return output_path


def plot_model_comparison(
    comparison_csv: Path,
    output_path: Path = Path("results/figures/model_comparison.png"),
) -> Path:
    """Bar chart comparing model AUC and TSS scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(comparison_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_labels = {"maxent": "MaxEnt", "rf": "Random Forest", "gbm": "Gradient Boosting"}
    labels = [model_labels.get(m, m) for m in df["model"]]
    colors = ["#2166ac", "#4daf4a", "#ff7f00"]
    x = np.arange(len(labels))

    # AUC
    ax = axes[0]
    bars = ax.bar(x, df["mean_auc"], yerr=df["std_auc"], capsize=5, color=colors, alpha=0.85)
    ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.7, label="Target (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_title("Area Under ROC Curve", fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # TSS
    ax = axes[1]
    bars = ax.bar(x, df["mean_tss"], yerr=df["std_tss"], capsize=5, color=colors, alpha=0.85)
    ax.axhline(y=0.70, color="red", linestyle="--", alpha=0.7, label="Target (0.70)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TSS")
    ax.set_title("True Skill Statistic", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Model Comparison — Spatial Block Cross-Validation (5-fold)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison chart saved: {output_path}")
    return output_path


def generate_all_figures(
    suitability_tif: Path,
    model_path: Path,
    training_csv: Path,
    importance_csv: Path,
    comparison_csv: Path,
    output_dir: Path = Path("results/figures"),
) -> list[Path]:
    """Generate all standard SDM figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(plot_suitability_map(
        suitability_tif,
        occurrence_csv=training_csv,
        output_path=output_dir / "suitability_map.png",
    ))

    paths.append(plot_variable_importance(
        importance_csv,
        output_path=output_dir / "variable_importance.png",
    ))

    paths.append(plot_response_curves(
        model_path,
        training_csv,
        output_path=output_dir / "response_curves.png",
    ))

    paths.append(plot_model_comparison(
        comparison_csv,
        output_path=output_dir / "model_comparison.png",
    ))

    logger.info(f"\nAll figures saved to {output_dir}/")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate habitat model visualization figures"
    )
    parser.add_argument("--suitability", type=Path, default=Path("results/suitability_map.tif"))
    parser.add_argument("--model", type=Path, default=Path("results/comparison/best_model_rf.pkl"))
    parser.add_argument("--training-data", type=Path, default=Path("data/training/features.csv"))
    parser.add_argument("--importance", type=Path, default=Path("results/comparison/variable_importance.csv"))
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison/model_comparison.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    generate_all_figures(
        suitability_tif=args.suitability,
        model_path=args.model,
        training_csv=args.training_data,
        importance_csv=args.importance,
        comparison_csv=args.comparison,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
