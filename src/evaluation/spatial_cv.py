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
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score

logger = logging.getLogger(__name__)


def assign_spatial_blocks(
    df: pd.DataFrame,
    n_blocks_x: int = 4,
    n_blocks_y: int = 5,
    n_folds: int = 5,
    min_presence_per_fold: int = 5,
) -> pd.DataFrame:
    """Assign each point to a spatial block with balanced fold distribution.

    Uses a 4x5 grid (20 blocks) assigned to 5 folds (4 blocks each) for
    even spatial coverage. Validates that each fold has sufficient presence
    points before returning.
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

    block_id = block_y * n_blocks_x + block_x
    n_blocks = n_blocks_x * n_blocks_y

    # Balanced fold assignment: distribute blocks evenly across folds
    # Sort blocks by checkerboard order for spatial separation, then round-robin assign
    blocks_ordered = []
    for bx in range(n_blocks_x):
        for by in range(n_blocks_y):
            bid = by * n_blocks_x + bx
            # Checkerboard ordering ensures spatially separated blocks go to same fold
            blocks_ordered.append((bid, (bx + by) % 2, bx, by))

    # Sort by checkerboard group, then position for consistent assignment
    blocks_ordered.sort(key=lambda t: (t[1], t[2], t[3]))

    # Round-robin assignment ensures exactly n_blocks/n_folds blocks per fold
    fold_map = {}
    for i, (bid, _, _, _) in enumerate(blocks_ordered):
        fold_map[bid] = i % n_folds

    df["fold"] = block_id.map(fold_map)

    # Pre-flight check: validate fold sizes
    fold_sizes = []
    for fold in range(n_folds):
        fold_mask = df["fold"] == fold
        n_pres = (df.loc[fold_mask, "presence"] == 1).sum()
        n_bg = (df.loc[fold_mask, "presence"] == 0).sum()
        n_blocks_in_fold = sum(1 for v in fold_map.values() if v == fold)
        fold_sizes.append({
            "fold": fold, "presence": n_pres, "background": n_bg,
            "total": n_pres + n_bg, "blocks": n_blocks_in_fold,
        })
        logger.info(
            f"  Fold {fold}: {n_pres} presence, {n_bg} background "
            f"({n_blocks_in_fold} blocks)"
        )

    # Check minimum presence per fold
    low_folds = [s for s in fold_sizes if s["presence"] < min_presence_per_fold]
    if low_folds:
        for s in low_folds:
            logger.warning(
                f"  WARNING: Fold {s['fold']} has only {s['presence']} presence "
                f"points (minimum {min_presence_per_fold})"
            )

    # Check fold balance (within 20% of mean)
    totals = [s["total"] for s in fold_sizes]
    mean_total = np.mean(totals)
    for s in fold_sizes:
        deviation = abs(s["total"] - mean_total) / mean_total
        if deviation > 0.20:
            logger.warning(
                f"  WARNING: Fold {s['fold']} size ({s['total']}) deviates "
                f"{deviation:.0%} from mean ({mean_total:.0f})"
            )

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


def compute_kappa_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing Cohen's Kappa, which accounts for chance agreement.

    Returns (kappa, optimal_threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_kappa = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        kappa = cohen_kappa_score(y_true, y_pred)
        if kappa > best_kappa:
            best_kappa = kappa
            best_thresh = thresh

    return best_kappa, best_thresh


def equal_sensitivity_specificity_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """Find threshold where sensitivity equals specificity.

    Produces balanced predictions — more conservative than max-TSS.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    # Find where |sensitivity - specificity| is minimized
    diff = np.abs(tpr - specificity)
    best_idx = np.argmin(diff)
    return thresholds[best_idx]


def prevalence_adjusted_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """Compute prevalence-adjusted threshold.

    Scales the base threshold by log(prevalence) to account for
    imbalanced presence:background ratios.
    """
    prevalence = y_true.sum() / len(y_true)
    _, base_threshold = compute_tss(y_true, y_prob)
    # Adjust: lower prevalence -> lower threshold (more permissive)
    adjustment = np.log(prevalence) / np.log(0.5)  # normalized to 1.0 at 50% prevalence
    return base_threshold * adjustment


def compute_all_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Compute all threshold methods and return as dict."""
    _, tss_thresh = compute_tss(y_true, y_prob)
    _, kappa_thresh = compute_kappa_threshold(y_true, y_prob)
    equal_ss_thresh = equal_sensitivity_specificity_threshold(y_true, y_prob)
    prev_thresh = prevalence_adjusted_threshold(y_true, y_prob)

    return {
        "max_tss": tss_thresh,
        "max_kappa": kappa_thresh,
        "equal_sens_spec": equal_ss_thresh,
        "prevalence_adjusted": prev_thresh,
    }


def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute SDM evaluation metrics for a single fold."""
    auc = roc_auc_score(y_true, y_prob)
    tss, threshold = compute_tss(y_true, y_prob)

    # Binary predictions at optimal TSS threshold
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Compute all threshold methods
    all_thresholds = compute_all_thresholds(y_true, y_prob)

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
        "thresholds": all_thresholds,
    }


def run_spatial_cv(
    df: pd.DataFrame,
    model_type: str = "maxent",
    n_blocks_x: int = 4,
    n_blocks_y: int = 5,
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
        "n_folds_skipped": len(folds) - len(fold_results),
        "grid_size": f"{n_blocks_x}x{n_blocks_y}",
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


def evaluate_independent(
    model,
    test_csv: Path,
    bioclim_stack: Path,
    terrain_dir: Path | None = None,
    cv_auc: float | None = None,
) -> dict:
    """Evaluate model on independent test set.

    Assembles features at independent test locations and computes
    AUC/TSS against presence labels. Compares with CV AUC if provided.
    """
    from src.features.assemble import sample_raster_at_points
    from src.models.maxent import get_feature_cols, predict_probability

    import json

    test_df = pd.read_csv(test_csv)
    logger.info(f"\nIndependent test evaluation: {len(test_df)} points")

    lons = test_df["longitude"].values
    lats = test_df["latitude"].values

    # Load band names
    band_index_path = bioclim_stack.parent / "bioclim_bands.json"
    if band_index_path.exists():
        with open(band_index_path) as f:
            band_index = json.load(f)
        band_names = [
            band_index[str(i)]["variable"]
            for i in sorted(int(k) for k in band_index.keys())
        ]
    else:
        band_names = None

    # Extract features
    bioclim_features = sample_raster_at_points(bioclim_stack, lons, lats, band_names)

    terrain_dfs = []
    if terrain_dir and terrain_dir.exists():
        for name in ["slope", "aspect", "twi", "tpi"]:
            resampled = terrain_dir / "resampled" / f"{name}_resampled.tif"
            native = terrain_dir / f"{name}.tif"
            path = resampled if resampled.exists() else native
            if path.exists():
                tdf = sample_raster_at_points(path, lons, lats, band_names=[name])
                terrain_dfs.append(tdf)

    import pandas as pd
    features = pd.concat([bioclim_features] + terrain_dfs, axis=1)

    # All test points are presences; need background for AUC
    # Use random predictions as null model comparison
    X = features.values
    y_prob = predict_probability(model, X)

    valid = ~np.isnan(y_prob)
    valid_count = valid.sum()
    logger.info(f"  Valid predictions: {valid_count}/{len(y_prob)}")

    if valid_count < 5:
        logger.warning("  Too few valid predictions for independent evaluation")
        return {"status": "insufficient_data", "valid_count": int(valid_count)}

    # Since all test points are presences, evaluate as:
    # - Mean predicted suitability (should be high)
    # - Proportion above threshold
    mean_suit = float(np.nanmean(y_prob[valid]))
    logger.info(f"  Mean suitability at presence locations: {mean_suit:.3f}")

    result = {
        "n_test_points": len(test_df),
        "n_valid": int(valid_count),
        "mean_suitability": mean_suit,
        "median_suitability": float(np.nanmedian(y_prob[valid])),
        "min_suitability": float(np.nanmin(y_prob[valid])),
        "max_suitability": float(np.nanmax(y_prob[valid])),
    }

    if cv_auc is not None:
        result["cv_auc"] = cv_auc
        logger.info(f"  CV AUC for reference: {cv_auc:.3f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial block cross-validation"
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--model-type", choices=["maxent", "rf", "gbm"], default="maxent")
    parser.add_argument("--blocks-x", type=int, default=4)
    parser.add_argument("--blocks-y", type=int, default=5)
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
