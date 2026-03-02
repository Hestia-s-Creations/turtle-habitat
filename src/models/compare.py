from __future__ import annotations

"""Compare model types using spatial block cross-validation.

Runs MaxEnt, Random Forest, and Gradient Boosting on the same data
with the same spatial CV folds for fair comparison.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.spatial_cv import run_spatial_cv
from src.models.maxent import get_feature_cols, train_model, save_model, variable_importance

logger = logging.getLogger(__name__)


def compare_models(
    training_data: Path,
    output_dir: Path,
    model_types: list[str] | None = None,
) -> pd.DataFrame:
    """Run spatial CV for multiple model types and compare."""
    if model_types is None:
        model_types = ["maxent", "rf", "gbm"]

    df = pd.read_csv(training_data)
    all_results = {}

    for model_type in model_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EVALUATING: {model_type.upper()}")
        logger.info(f"{'=' * 60}")

        results = run_spatial_cv(df, model_type=model_type)
        all_results[model_type] = results

    # Build comparison table
    rows = []
    for model_type, results in all_results.items():
        agg = results["aggregate"]
        rows.append({
            "model": model_type,
            "mean_auc": agg.get("mean_auc", 0),
            "std_auc": agg.get("std_auc", 0),
            "mean_tss": agg.get("mean_tss", 0),
            "std_tss": agg.get("std_tss", 0),
            "mean_sensitivity": agg.get("mean_sensitivity", 0),
            "mean_specificity": agg.get("mean_specificity", 0),
            "n_folds": agg.get("n_folds", 0),
        })

    comparison = pd.DataFrame(rows)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"MODEL COMPARISON")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n{comparison.to_string(index=False)}")

    # Identify best model
    best_idx = comparison["mean_auc"].idxmax()
    best_model = comparison.loc[best_idx, "model"]
    best_auc = comparison.loc[best_idx, "mean_auc"]
    logger.info(f"\nBest model by AUC: {best_model} ({best_auc:.3f})")

    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    with open(output_dir / "all_cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Train final model on all data using best type
    logger.info(f"\nTraining final {best_model} model on all data...")
    feature_cols = get_feature_cols(df)
    final_model = train_model(df, model_type=best_model)
    save_model(final_model, feature_cols, output_dir / f"best_model_{best_model}.pkl")

    importance = variable_importance(final_model, feature_cols)
    importance.to_csv(output_dir / "variable_importance.csv", index=False)
    logger.info(f"\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare SDM model types")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--model-types", nargs="+", default=["maxent", "rf", "gbm"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    compare_models(args.training_data, args.output_dir, args.model_types)


if __name__ == "__main__":
    main()
