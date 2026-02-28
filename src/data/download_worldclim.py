"""Download WorldClim bioclimatic variables for the study area.

WorldClim v2.1 provides 19 bioclimatic variables derived from monthly
temperature and precipitation data at multiple resolutions. We use
the 30-second (~1km) resolution for species distribution modeling.

Data source: https://www.worldclim.org/data/worldclim21.html
"""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.windows import from_bounds
import requests
from shapely.geometry import box

logger = logging.getLogger(__name__)

# WorldClim v2.1 download URLs (30s resolution = ~1km)
WORLDCLIM_BASE = "https://biogeo.ucanr.edu/data/worldclim/v2.1/base"
RESOLUTION = "30s"

# 19 bioclimatic variables
BIOCLIM_VARS = {
    "bio1": "Annual Mean Temperature",
    "bio2": "Mean Diurnal Range",
    "bio3": "Isothermality",
    "bio4": "Temperature Seasonality",
    "bio5": "Max Temperature of Warmest Month",
    "bio6": "Min Temperature of Coldest Month",
    "bio7": "Temperature Annual Range",
    "bio8": "Mean Temperature of Wettest Quarter",
    "bio9": "Mean Temperature of Driest Quarter",
    "bio10": "Mean Temperature of Warmest Quarter",
    "bio11": "Mean Temperature of Coldest Quarter",
    "bio12": "Annual Precipitation",
    "bio13": "Precipitation of Wettest Month",
    "bio14": "Precipitation of Driest Month",
    "bio15": "Precipitation Seasonality",
    "bio16": "Precipitation of Wettest Quarter",
    "bio17": "Precipitation of Driest Quarter",
    "bio18": "Precipitation of Warmest Quarter",
    "bio19": "Precipitation of Coldest Quarter",
}

# Study area bounding boxes
STUDY_AREAS = {
    "oregon": (-124.6, 41.9, -116.5, 46.3),
    "pnw": (-125.0, 41.0, -116.0, 49.0),
}


def download_bioclim(output_dir: Path, timeout: int = 300) -> Path:
    """Download WorldClim bioclimatic variables zip file.

    Returns path to the extracted directory.
    """
    url = f"{WORLDCLIM_BASE}/wc2.1_{RESOLUTION}_bio.zip"
    zip_path = output_dir / f"wc2.1_{RESOLUTION}_bio.zip"
    extract_dir = output_dir / "worldclim_raw"

    if extract_dir.exists() and any(extract_dir.glob("*.tif")):
        logger.info(f"WorldClim data already extracted at {extract_dir}")
        return extract_dir

    if not zip_path.exists():
        logger.info(f"Downloading WorldClim bioclim from {url}")
        logger.info("  This may take a few minutes (~100MB)...")

        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) == 0:
                    logger.info(
                        f"  Downloaded {downloaded / 1024 / 1024:.0f}/"
                        f"{total / 1024 / 1024:.0f} MB"
                    )

        logger.info(f"  Download complete: {zip_path}")
    else:
        logger.info(f"Using cached zip: {zip_path}")

    # Extract
    logger.info(f"Extracting to {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def clip_to_study_area(
    input_path: Path,
    output_path: Path,
    bounds: tuple[float, float, float, float],
) -> None:
    """Clip a global raster to the study area bounding box."""
    west, south, east, north = bounds

    with rasterio.open(input_path) as src:
        window = from_bounds(west, south, east, north, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        data = src.read(1, window=window)

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)


def build_feature_stack(
    raw_dir: Path,
    output_dir: Path,
    bounds: tuple[float, float, float, float],
    variables: list[str] | None = None,
) -> Path:
    """Clip all bioclim variables to study area and stack into multi-band raster.

    Returns path to the output stacked GeoTIFF.
    """
    if variables is None:
        variables = list(BIOCLIM_VARS.keys())

    clipped_dir = output_dir / "clipped"
    clipped_dir.mkdir(parents=True, exist_ok=True)

    band_paths = []
    for var in variables:
        # WorldClim naming: wc2.1_30s_bio_1.tif through wc2.1_30s_bio_19.tif
        var_num = var.replace("bio", "")
        candidates = list(raw_dir.glob(f"*bio_{var_num}.tif")) + \
                     list(raw_dir.glob(f"*bio{var_num}.tif"))

        if not candidates:
            logger.warning(f"  Could not find raster for {var}, skipping")
            continue

        input_path = candidates[0]
        clipped_path = clipped_dir / f"{var}.tif"

        if not clipped_path.exists():
            logger.info(f"  Clipping {var} ({BIOCLIM_VARS.get(var, '')})")
            clip_to_study_area(input_path, clipped_path, bounds)

        band_paths.append((var, clipped_path))

    if not band_paths:
        raise RuntimeError("No bioclim variables found to stack")

    # Stack into a single multi-band GeoTIFF
    stack_path = output_dir / "bioclim_stack.tif"
    logger.info(f"Stacking {len(band_paths)} bands into {stack_path}")

    # Read first band for reference
    with rasterio.open(band_paths[0][1]) as ref:
        profile = ref.profile.copy()
        height = ref.height
        width = ref.width

    profile.update({
        "count": len(band_paths),
        "dtype": "float32",
    })

    with rasterio.open(stack_path, "w", **profile) as dst:
        for i, (var, path) in enumerate(band_paths, 1):
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                # WorldClim uses -3.4e38 as nodata for some bands
                data[data < -1e10] = np.nan
                dst.write(data, i)
                dst.set_band_description(i, f"{var}: {BIOCLIM_VARS.get(var, '')}")

    logger.info(f"Feature stack saved: {stack_path} ({len(band_paths)} bands)")

    # Write band index file for reference
    index_path = output_dir / "bioclim_bands.json"
    import json
    band_index = {
        i + 1: {"variable": var, "description": BIOCLIM_VARS.get(var, "")}
        for i, (var, _) in enumerate(band_paths)
    }
    with open(index_path, "w") as f:
        json.dump(band_index, f, indent=2)

    return stack_path


def main():
    parser = argparse.ArgumentParser(
        description="Download WorldClim bioclimatic variables"
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Specific bioclim variables to download (e.g., bio1 bio12). Default: all 19",
    )
    parser.add_argument(
        "--region",
        choices=["oregon", "pnw"],
        default="oregon",
        help="Study area for clipping (default: oregon)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/worldclim"),
        help="Output directory (default: data/worldclim)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = STUDY_AREAS[args.region]

    # Download
    raw_dir = download_bioclim(args.output_dir)

    # Clip and stack
    stack_path = build_feature_stack(
        raw_dir, args.output_dir, bounds, variables=args.variables
    )

    logger.info(f"\nDone! Feature stack at: {stack_path}")
    logger.info(f"Bounds: {bounds}")
    logger.info(f"Variables: {args.variables or 'all 19'}")


if __name__ == "__main__":
    main()
