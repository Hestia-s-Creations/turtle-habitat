from __future__ import annotations

"""Assemble environmental features at occurrence and background points.

Extracts raster values from the bioclim stack and terrain derivatives
at each occurrence point and at background sampling points, producing
a training-ready DataFrame for the MaxEnt model.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)


def sample_raster_at_points(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    band_names: list[str] | None = None,
) -> pd.DataFrame:
    """Extract raster values at given lon/lat coordinates.

    Returns a DataFrame with one column per band.
    """
    with rasterio.open(raster_path) as src:
        n_bands = src.count

        if band_names is None:
            band_names = [
                src.descriptions[i] if src.descriptions[i]
                else f"band_{i + 1}"
                for i in range(n_bands)
            ]

        # Convert lon/lat to pixel coordinates
        rows, cols = rasterio.transform.rowcol(
            src.transform, lons, lats
        )
        rows = np.array(rows)
        cols = np.array(cols)

        # Clip to valid range
        valid = (
            (rows >= 0) & (rows < src.height) &
            (cols >= 0) & (cols < src.width)
        )

        values = np.full((len(lons), n_bands), np.nan, dtype=np.float32)
        if valid.any():
            for band_idx in range(n_bands):
                data = src.read(band_idx + 1)
                values[valid, band_idx] = data[rows[valid], cols[valid]]

    df = pd.DataFrame(values, columns=band_names[:n_bands])
    return df


def generate_background_points(
    raster_path: Path,
    n_points: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random background points within the raster extent.

    Excludes nodata pixels. Returns (lons, lats).
    """
    rng = np.random.default_rng(seed)

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        data = src.read(1)
        nodata = src.nodata

        # Create mask of valid pixels
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data)

    # Generate more points than needed, filter to valid
    oversample = 3
    lons = rng.uniform(bounds.left, bounds.right, n_points * oversample)
    lats = rng.uniform(bounds.bottom, bounds.top, n_points * oversample)

    # Filter to valid pixels
    with rasterio.open(raster_path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
        rows = np.array(rows)
        cols = np.array(cols)

        in_bounds = (
            (rows >= 0) & (rows < src.height) &
            (cols >= 0) & (cols < src.width)
        )
        valid = in_bounds.copy()
        valid[in_bounds] &= valid_mask[rows[in_bounds], cols[in_bounds]]

    lons = lons[valid][:n_points]
    lats = lats[valid][:n_points]

    logger.info(f"Generated {len(lons)} background points")
    return lons, lats


def generate_target_group_background(
    background_csv: Path,
    n_points: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Use target-group herpetofauna observations as background points.

    This corrects for observer bias by sampling background from locations
    where observers were present (detected other species) but did not
    detect the target species.
    """
    df = pd.read_csv(background_csv)
    logger.info(f"Loaded {len(df)} target-group observations")

    if len(df) > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=n_points, replace=False)
        df = df.iloc[idx]

    return df["longitude"].values, df["latitude"].values


def assemble_training_data(
    occurrence_csv: Path,
    bioclim_stack: Path,
    terrain_dir: Path | None = None,
    background_csv: Path | None = None,
    n_background: int = 10000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the full training dataset.

    Combines:
    - Presence points (from occurrence CSV)
    - Background/pseudo-absence points
    - Environmental features at each point
    """
    # Load occurrences
    occ = pd.read_csv(occurrence_csv)
    logger.info(f"Loaded {len(occ)} occurrence records")

    occ_lons = occ["longitude"].values
    occ_lats = occ["latitude"].values

    # Generate or load background
    if background_csv and background_csv.exists():
        bg_lons, bg_lats = generate_target_group_background(
            background_csv, n_points=n_background
        )
    else:
        bg_lons, bg_lats = generate_background_points(
            bioclim_stack, n_points=n_background
        )

    # Combine
    all_lons = np.concatenate([occ_lons, bg_lons])
    all_lats = np.concatenate([occ_lats, bg_lats])
    labels = np.concatenate([
        np.ones(len(occ_lons), dtype=int),
        np.zeros(len(bg_lons), dtype=int),
    ])

    # Load band names from bioclim stack
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

    # Sample bioclim features
    logger.info("Extracting bioclim features...")
    bioclim_df = sample_raster_at_points(
        bioclim_stack, all_lons, all_lats, band_names
    )

    # Sample terrain features
    terrain_dfs = []
    if terrain_dir and terrain_dir.exists():
        for name in ["slope", "aspect", "twi", "tpi"]:
            # Check for resampled version first
            resampled = terrain_dir / "resampled" / f"{name}_resampled.tif"
            native = terrain_dir / f"{name}.tif"
            path = resampled if resampled.exists() else native

            if path.exists():
                logger.info(f"Extracting {name} features...")
                tdf = sample_raster_at_points(
                    path, all_lons, all_lats, band_names=[name]
                )
                terrain_dfs.append(tdf)

    # Combine all features
    result = pd.DataFrame({
        "longitude": all_lons,
        "latitude": all_lats,
        "presence": labels,
    })
    result = pd.concat([result, bioclim_df] + terrain_dfs, axis=1)

    # Drop rows with all-NaN features (points outside raster extent)
    feature_cols = [c for c in result.columns if c not in ("longitude", "latitude", "presence")]
    before = len(result)
    result = result.dropna(subset=feature_cols, how="all")
    if len(result) < before:
        logger.info(f"Dropped {before - len(result)} points outside raster extent")

    logger.info(f"\n=== Training data ===")
    logger.info(f"Presence points: {(result['presence'] == 1).sum()}")
    logger.info(f"Background points: {(result['presence'] == 0).sum()}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Feature names: {feature_cols}")

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info(f"Saved training data to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Assemble environmental features at occurrence and background points"
    )
    parser.add_argument(
        "--occurrences",
        type=Path,
        required=True,
        help="CSV with occurrence records (must have longitude, latitude columns)",
    )
    parser.add_argument(
        "--bioclim-stack",
        type=Path,
        required=True,
        help="Path to bioclim feature stack GeoTIFF",
    )
    parser.add_argument(
        "--terrain-dir",
        type=Path,
        default=None,
        help="Directory containing terrain rasters (slope.tif, aspect.tif, twi.tif)",
    )
    parser.add_argument(
        "--background-csv",
        type=Path,
        default=None,
        help="CSV with target-group background observations (for bias correction)",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=10000,
        help="Number of background points to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/features.csv"),
        help="Output CSV path (default: data/training/features.csv)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    assemble_training_data(
        occurrence_csv=args.occurrences,
        bioclim_stack=args.bioclim_stack,
        terrain_dir=args.terrain_dir,
        background_csv=args.background_csv,
        n_background=args.n_background,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
