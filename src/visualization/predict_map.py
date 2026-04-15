from __future__ import annotations

"""Generate habitat suitability maps from trained models.

Reads the bioclim + terrain feature stack, runs the model on every pixel,
and writes a GeoTIFF suitability map (0-1 probability).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


def predict_raster(
    model_path: Path,
    bioclim_stack: Path,
    terrain_dir: Path | None = None,
    output_path: Path = Path("results/suitability_map.tif"),
    chunk_rows: int = 256,
) -> Path:
    """Generate a suitability map by predicting on every pixel.

    Processes in row chunks to manage memory.
    """
    from src.models.maxent import load_model, predict_probability

    model, feature_cols = load_model(model_path)

    # Determine which bands come from bioclim vs terrain
    bioclim_band_names = []
    terrain_paths = {}

    with rasterio.open(bioclim_stack) as src:
        n_bioclim = src.count
        profile = src.profile.copy()
        height = src.height
        width = src.width

        for i in range(n_bioclim):
            desc = src.descriptions[i] if src.descriptions[i] else f"band_{i + 1}"
            # Extract short name (e.g., "bio10" from "bio10: Mean Temperature...")
            short_name = desc.split(":")[0].strip() if ":" in desc else desc
            bioclim_band_names.append(short_name)

    # Check for terrain rasters
    terrain_names = ["slope", "aspect", "twi", "tpi"]
    if terrain_dir and terrain_dir.exists():
        for name in terrain_names:
            resampled = terrain_dir / "resampled" / f"{name}_resampled.tif"
            native = terrain_dir / f"{name}.tif"
            path = resampled if resampled.exists() else native
            if path.exists():
                terrain_paths[name] = path

    all_band_names = bioclim_band_names + list(terrain_paths.keys())

    # Map model's feature_cols to raster band indices
    band_indices = []
    for fc in feature_cols:
        if fc in all_band_names:
            band_indices.append(all_band_names.index(fc))
        else:
            logger.warning(f"  Feature '{fc}' not found in raster bands")

    n_selected = len(band_indices)

    logger.info(f"Predicting suitability map")
    logger.info(f"  Raster size: {width} x {height}")
    logger.info(f"  Available bands: {len(all_band_names)}")
    logger.info(f"  Model features: {feature_cols}")
    logger.info(f"  Selected {n_selected} bands from rasters")

    # Set up output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({
        "count": 1,
        "dtype": "float32",
        "nodata": -9999,
    })

    # Open all inputs
    bioclim_src = rasterio.open(bioclim_stack)
    terrain_srcs = {name: rasterio.open(path) for name, path in terrain_paths.items()}

    with rasterio.open(output_path, "w", **out_profile) as dst:
        for row_start in range(0, height, chunk_rows):
            row_end = min(row_start + chunk_rows, height)
            n_rows = row_end - row_start
            window = rasterio.windows.Window(0, row_start, width, n_rows)

            # Read bioclim bands
            bioclim_data = bioclim_src.read(window=window).astype(np.float32)
            # Shape: (n_bioclim, n_rows, width)

            # Read terrain bands
            terrain_data = []
            for name in terrain_paths:
                tdata = terrain_srcs[name].read(1, window=window).astype(np.float32)
                terrain_data.append(tdata)

            # Stack all bands: (n_total, n_rows, width)
            if terrain_data:
                all_data = np.concatenate(
                    [bioclim_data, np.stack(terrain_data)], axis=0
                )
            else:
                all_data = bioclim_data

            # Select only the bands the model expects
            selected_data = all_data[band_indices]

            # Reshape to (n_pixels, n_features)
            n_pixels = n_rows * width
            X = selected_data.reshape(n_selected, n_pixels).T

            # Predict
            probs = predict_probability(model, X)

            # Reshape back and write
            prob_grid = probs.reshape(n_rows, width)
            prob_grid[np.isnan(prob_grid)] = -9999
            dst.write(prob_grid.astype(np.float32), 1, window=window)

            progress = row_end / height * 100
            if progress % 25 < (chunk_rows / height * 100):
                logger.info(f"  Progress: {progress:.0f}%")

    # Clean up
    bioclim_src.close()
    for src in terrain_srcs.values():
        src.close()

    logger.info(f"Suitability map saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate habitat suitability map from trained model"
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model .pkl")
    parser.add_argument("--bioclim-stack", type=Path, required=True, help="Bioclim feature stack GeoTIFF")
    parser.add_argument("--terrain-dir", type=Path, default=None, help="Terrain derivatives directory")
    parser.add_argument("--output", type=Path, default=Path("results/suitability_map.tif"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    predict_raster(
        model_path=args.model,
        bioclim_stack=args.bioclim_stack,
        terrain_dir=args.terrain_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
