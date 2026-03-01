from __future__ import annotations

"""MaxEnt species distribution model via scikit-learn.

MaxEnt (Maximum Entropy) is the standard for presence-only species
distribution modeling. We implement it as penalized logistic regression
with quadratic feature expansion, which is mathematically equivalent
to the original MaxEnt formulation (Renner & Warton 2013).

Includes Random Forest and XGBoost ensemble alternatives for comparison.

Note: Models are serialized with pickle because sklearn pipelines contain
fitted numpy arrays and transformers that cannot be safely represented in
JSON. These are local-only model artifacts, never loaded from untrusted sources.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Feature columns to exclude from modeling
META_COLS = {"longitude", "latitude", "presence", "fold"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get feature column names from a training DataFrame."""
    return [c for c in df.columns if c not in META_COLS]


def build_maxent_pipeline(
    n_features: int,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Build a MaxEnt-equivalent pipeline.

    MaxEnt with linear + quadratic features is equivalent to
    L2-penalized logistic regression with polynomial expansion.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ("model", LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            class_weight="balanced",
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    """Build a Random Forest pipeline for comparison."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])


def build_gbm_pipeline() -> Pipeline:
    """Build a Gradient Boosting pipeline for comparison."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])


MODEL_BUILDERS = {
    "maxent": build_maxent_pipeline,
    "rf": build_rf_pipeline,
    "gbm": build_gbm_pipeline,
}


def train_model(
    train_df: pd.DataFrame,
    model_type: str = "maxent",
    **model_kwargs,
) -> Pipeline:
    """Train a species distribution model."""
    feature_cols = get_feature_cols(train_df)
    X = train_df[feature_cols].values
    y = train_df["presence"].values

    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    logger.info(f"Training {model_type} model")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(X)} (presence={y.sum()}, background={(1 - y).sum()})")

    if model_type == "maxent":
        pipeline = build_maxent_pipeline(n_features=len(feature_cols), **model_kwargs)
    else:
        builder = MODEL_BUILDERS[model_type]
        pipeline = builder()

    pipeline.fit(X, y)
    logger.info(f"  Training complete")

    return pipeline


def predict_probability(
    model: Pipeline,
    X: np.ndarray,
) -> np.ndarray:
    """Predict habitat suitability probability (0-1)."""
    nan_mask = np.isnan(X).any(axis=1)
    probs = np.full(len(X), np.nan)

    if (~nan_mask).any():
        X_valid = np.nan_to_num(X[~nan_mask], nan=0)
        probs[~nan_mask] = model.predict_proba(X_valid)[:, 1]

    return probs


def variable_importance(
    model: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extract variable importance from the fitted model."""
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        if "poly" in model.named_steps:
            poly = model.named_steps["poly"]
            poly_names = poly.get_feature_names_out(feature_cols)
            return pd.DataFrame({
                "feature": poly_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    elif hasattr(estimator, "coef_"):
        coefs = np.abs(estimator.coef_[0])
        if "poly" in model.named_steps:
            poly = model.named_steps["poly"]
            poly_names = poly.get_feature_names_out(feature_cols)
            return pd.DataFrame({
                "feature": poly_names,
                "importance": coefs,
            }).sort_values("importance", ascending=False)
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": coefs,
        }).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_cols, "importance": np.nan})


def save_model(model: Pipeline, feature_cols: list[str], output_path: Path) -> None:
    """Save trained model and metadata. Uses pickle for sklearn pipeline serialization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    logger.info(f"Model saved to {output_path}")


def load_model(model_path: Path) -> tuple[Pipeline, list[str]]:
    """Load a saved model. Only load models you created locally."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]


def main():
    parser = argparse.ArgumentParser(description="Train a species distribution model")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--model-type", choices=["maxent", "rf", "gbm"], default="maxent")
    parser.add_argument("--output", type=Path, default=Path("results/models/maxent_baseline.pkl"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = pd.read_csv(args.training_data)
    feature_cols = get_feature_cols(df)

    model = train_model(df, model_type=args.model_type)
    save_model(model, feature_cols, args.output)

    importance = variable_importance(model, feature_cols)
    logger.info(f"\nTop 10 most important features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
