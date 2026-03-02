from __future__ import annotations

"""Spatial block cross-validation for honest accuracy estimation.

Random train/test splitting inflates accuracy because nearby pixels
share spatial autocorrelation. Block CV divides the study area into
geographic tiles — training on 4 tiles and testing on the held-out 5th.
This gives honest estimates of performance on unseen landscapes.

Also computes AUC, TSS, and other SDM-standard metrics.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def assign_spatial_blocks(
    df: pd.DataFrame,
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
) -> pd.DataFrame:
    """Assign each point to a spatial block based on geographic position.

    Divides the study area into a grid of n_blocks_x * n_blocks_y blocks
    and assigns each point a fold number (0 to n_folds-1).
    """
    df = df.copy()

    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

    # Add small buffer to include edge points
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * 0.001
    lat_min -= lat_range * 0.001

    block_width = (lon_max - lon_min) / n_blocks_x
    block_height = (lat_max - lat_min) / n_blocks_y

    block_x = ((df["longitude"] - lon_min) / block_width).astype(int).clip(0, n_blocks_x - 1)
    block_y = ((df["latitude"] - lat_min) / block_height).astype(int).clip(0, n_blocks_y - 1)

    # Assign block index, then map to folds
    block_id = block_y * n_blocks_x + block_x
    n_blocks = n_blocks_x * n_blocks_y

    # Use checkerboard pattern for fold assignment (better spatial separation)
    fold_map = {}
    n_folds = 5
    for bx in range(n_blocks_x):
        for by in range(n_blocks_y):
            bid = by * n_blocks_x + bx
            fold_map[bid] = (bx + by) % n_folds

    df["fold"] = block_id.map(fold_map)

    # Log fold distribution
    for fold in sorted(df["fold"].unique()):
        n_pres = (df.loc[df["fold"] == fold, "presence"] == 1).sum()
        n_bg = (df.loc[df["fold"] == fold, "presence"] == 0).sum()
        logger.info(f"  Fold {fold}: {n_pres} presence, {n_bg} background")

    return df


def compute_tss(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Compute True Skill Statistic at optimal threshold.

    TSS = Sensitivity + Specificity - 1
    Range: -1 to +1, where >0.75 is good for SDMs.

    Returns (tss, optimal_threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    tss_values = tpr + specificity - 1

    best_idx = np.argmax(tss_values)
    return tss_values[best_idx], thresholds[best_idx]


def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute SDM evaluation metrics for a single fold."""
    auc = roc_auc_score(y_true, y_prob)
    tss, threshold = compute_tss(y_true, y_prob)

    # Binary predictions at optimal threshold
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "auc": auc,
        "tss": tss,
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_spatial_cv(
    df: pd.DataFrame,
    model_type: str = "maxent",
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
) -> dict:
    """Run spatial block cross-validation.

    Returns per-fold and aggregate metrics.
    """
    from src.models.maxent import get_feature_cols, train_model, predict_probability

    # Assign spatial blocks
    logger.info(f"Assigning spatial blocks ({n_blocks_x}x{n_blocks_y})")
    df = assign_spatial_blocks(df, n_blocks_x, n_blocks_y)

    feature_cols = get_feature_cols(df)
    folds = sorted(df["fold"].unique())
    fold_results = []

    for fold in folds:
        logger.info(f"\n--- Fold {fold} (test) ---")

        train_mask = df["fold"] != fold
        test_mask = df["fold"] == fold

        train_df = df[train_mask]
        test_df = df[test_mask]

        # Check minimum presence in both train and test
        train_pres = (train_df["presence"] == 1).sum()
        test_pres = (test_df["presence"] == 1).sum()
        logger.info(f"  Train: {len(train_df)} ({train_pres} presence)")
        logger.info(f"  Test: {len(test_df)} ({test_pres} presence)")

        if test_pres < 2 or train_pres < 5:
            logger.warning(f"  Skipping fold {fold}: insufficient presence points")
            continue

        # Train
        model = train_model(train_df, model_type=model_type)

        # Predict
        X_test = test_df[feature_cols].values
        y_test = test_df["presence"].values
        y_prob = predict_probability(model, X_test)

        # Filter NaN predictions
        valid = ~np.isnan(y_prob)
        if valid.sum() < 10:
            logger.warning(f"  Skipping fold {fold}: too few valid predictions")
            continue

        # Evaluate
        metrics = evaluate_fold(y_test[valid], y_prob[valid])
        metrics["fold"] = fold
        metrics["n_train"] = len(train_df)
        metrics["n_test"] = len(test_df)
        fold_results.append(metrics)

        logger.info(f"  AUC: {metrics['auc']:.3f}")
        logger.info(f"  TSS: {metrics['tss']:.3f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        logger.info(f"  Specificity: {metrics['specificity']:.3f}")

    if not fold_results:
        logger.error("No valid folds completed!")
        return {"folds": [], "aggregate": {}}

    # Aggregate
    fold_df = pd.DataFrame(fold_results)
    aggregate = {
        "mean_auc": fold_df["auc"].mean(),
        "std_auc": fold_df["auc"].std(),
        "mean_tss": fold_df["tss"].mean(),
        "std_tss": fold_df["tss"].std(),
        "mean_sensitivity": fold_df["sensitivity"].mean(),
        "mean_specificity": fold_df["specificity"].mean(),
        "n_folds": len(fold_results),
        "model_type": model_type,
    }

    logger.info(f"\n{'=' * 50}")
    logger.info(f"CROSS-VALIDATION RESULTS ({model_type})")
    logger.info(f"{'=' * 50}")
    logger.info(f"Mean AUC: {aggregate['mean_auc']:.3f} +/- {aggregate['std_auc']:.3f}")
    logger.info(f"Mean TSS: {aggregate['mean_tss']:.3f} +/- {aggregate['std_tss']:.3f}")
    logger.info(f"Mean Sensitivity: {aggregate['mean_sensitivity']:.3f}")
    logger.info(f"Mean Specificity: {aggregate['mean_specificity']:.3f}")
    logger.info(f"Folds completed: {aggregate['n_folds']}")

    # Target check
    if aggregate["mean_auc"] >= 0.80:
        logger.info(f"TARGET MET: AUC >= 0.80")
    else:
        logger.info(f"TARGET NOT MET: AUC {aggregate['mean_auc']:.3f} < 0.80")

    if aggregate["mean_tss"] >= 0.70:
        logger.info(f"TARGET MET: TSS >= 0.70")
    else:
        logger.info(f"TARGET NOT MET: TSS {aggregate['mean_tss']:.3f} < 0.70")

    return {"folds": fold_results, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial block cross-validation"
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--model-type", choices=["maxent", "rf", "gbm"], default="maxent")
    parser.add_argument("--blocks-x", type=int, default=3)
    parser.add_argument("--blocks-y", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/cv_results.json"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = pd.read_csv(args.training_data)
    results = run_spatial_cv(
        df,
        model_type=args.model_type,
        n_blocks_x=args.blocks_x,
        n_blocks_y=args.blocks_y,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
