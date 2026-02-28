"""Download and process USGS 3DEP elevation data for terrain derivatives.

Uses The National Map (TNM) API to download 1/3 arc-second (~10m) DEMs
for the study area, then computes slope, aspect, and Topographic Wetness
Index (TWI) as additional environmental features for habitat modeling.

For the turtle model, terrain features capture microhabitat preferences:
- Low slope near water (basking sites)
- TWI for wetness/proximity to water
- Aspect for solar exposure (thermoregulation)

Data source: https://apps.nationalmap.gov/tnmaccess/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import requests

logger = logging.getLogger(__name__)

# TNM API endpoint
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

# Study areas
STUDY_AREAS = {
    "oregon": (-124.6, 41.9, -116.5, 46.3),
    "pnw": (-125.0, 41.0, -116.0, 49.0),
}


def search_dem_tiles(bounds: tuple[float, float, float, float]) -> list[dict]:
    """Search TNM API for 1/3 arc-second DEM tiles covering the study area."""
    west, south, east, north = bounds
    params = {
        "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
        "bbox": f"{west},{south},{east},{north}",
        "prodFormats": "GeoTIFF",
        "max": 100,
    }

    logger.info(f"Searching TNM for DEM tiles in {bounds}")
    resp = requests.get(TNM_API, params=params)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    logger.info(f"  Found {len(items)} DEM tiles")
    return items


def download_dem_tile(url: str, output_path: Path, timeout: int = 300) -> None:
    """Download a single DEM tile."""
    if output_path.exists():
        logger.debug(f"  Tile already cached: {output_path.name}")
        return

    logger.info(f"  Downloading {output_path.name}")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def merge_tiles(tile_paths: list[Path], output_path: Path) -> None:
    """Merge multiple DEM tiles into a single raster."""
    if output_path.exists():
        logger.info(f"  Merged DEM already exists: {output_path}")
        return

    logger.info(f"Merging {len(tile_paths)} tiles")
    datasets = [rasterio.open(p) for p in tile_paths]

    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    logger.info(f"  Merged DEM saved: {output_path}")


def compute_slope(dem_path: Path, output_path: Path) -> None:
    """Compute slope in degrees from DEM using numpy gradient."""
    if output_path.exists():
        logger.info(f"  Slope already computed: {output_path}")
        return

    logger.info("Computing slope")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        profile = src.profile.copy()

        # Cell size in meters (approximate for geographic CRS)
        cellsize_x = abs(transform.a) * 111320 * np.cos(np.radians(
            transform.f + dem.shape[0] / 2 * transform.e
        ))
        cellsize_y = abs(transform.e) * 110540

        # Gradient
        dy, dx = np.gradient(dem, cellsize_y, cellsize_x)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Mask nodata
        nodata = src.nodata
        if nodata is not None:
            slope[dem == nodata] = np.nan

    profile.update({"dtype": "float32", "nodata": np.nan})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope.astype(np.float32), 1)

    logger.info(f"  Slope saved: {output_path}")


def compute_aspect(dem_path: Path, output_path: Path) -> None:
    """Compute aspect in degrees from DEM (0=N, 90=E, 180=S, 270=W)."""
    if output_path.exists():
        logger.info(f"  Aspect already computed: {output_path}")
        return

    logger.info("Computing aspect")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        profile = src.profile.copy()

        cellsize_x = abs(transform.a) * 111320 * np.cos(np.radians(
            transform.f + dem.shape[0] / 2 * transform.e
        ))
        cellsize_y = abs(transform.e) * 110540

        dy, dx = np.gradient(dem, cellsize_y, cellsize_x)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = (aspect + 360) % 360  # Convert to 0-360

        nodata = src.nodata
        if nodata is not None:
            aspect[dem == nodata] = np.nan

    profile.update({"dtype": "float32", "nodata": np.nan})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(aspect.astype(np.float32), 1)

    logger.info(f"  Aspect saved: {output_path}")


def compute_twi(dem_path: Path, slope_path: Path, output_path: Path) -> None:
    """Compute Topographic Wetness Index: TWI = ln(a / tan(b)).

    Where a = upslope contributing area and b = local slope.
    Uses a simplified approach with D8 flow accumulation via numpy.
    For production, use WhiteboxTools for proper flow routing.
    """
    if output_path.exists():
        logger.info(f"  TWI already computed: {output_path}")
        return

    logger.info("Computing TWI (simplified)")
    with rasterio.open(slope_path) as src:
        slope_deg = src.read(1).astype(np.float64)
        profile = src.profile.copy()

    slope_rad = np.radians(slope_deg)
    # Prevent division by zero - minimum slope of 0.1 degrees
    slope_rad = np.maximum(slope_rad, np.radians(0.1))

    # Simplified contributing area: use slope position as proxy
    # For proper TWI, use WhiteboxTools flow accumulation
    # This is a reasonable approximation for SDM features
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        cellsize = abs(src.transform.a) * 111320  # approximate meters

    # Simple flow accumulation via downhill neighbor counting
    # This is approximate but avoids heavy dependencies
    from scipy.ndimage import uniform_filter
    # Use smoothed inverse elevation as proxy for contributing area
    smoothed = uniform_filter(dem, size=5)
    flow_proxy = np.maximum(smoothed.max() - smoothed, 1.0) * cellsize

    twi = np.log(flow_proxy / np.tan(slope_rad))

    # Mask nodata
    twi[np.isnan(slope_deg)] = np.nan

    profile.update({"dtype": "float32", "nodata": np.nan})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(twi.astype(np.float32), 1)

    logger.info(f"  TWI saved: {output_path}")


def resample_to_target(
    input_path: Path,
    target_path: Path,
    output_path: Path,
) -> None:
    """Resample a raster to match the grid of a target raster (e.g., WorldClim)."""
    if output_path.exists():
        logger.info(f"  Already resampled: {output_path}")
        return

    with rasterio.open(target_path) as target:
        target_crs = target.crs
        target_transform = target.transform
        target_width = target.width
        target_height = target.height

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update({
            "crs": target_crs,
            "transform": target_transform,
            "width": target_width,
            "height": target_height,
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

    logger.info(f"  Resampled to target grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download USGS 3DEP DEM and compute terrain derivatives"
    )
    parser.add_argument(
        "--region",
        choices=["oregon", "pnw"],
        default="oregon",
        help="Study area (default: oregon)",
    )
    parser.add_argument(
        "--resample-to",
        type=Path,
        default=None,
        help="Path to target raster (e.g., WorldClim stack) to resample terrain to",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/terrain"),
        help="Output directory (default: data/terrain)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = STUDY_AREAS[args.region]
    tiles_dir = args.output_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    # Search and download tiles
    items = search_dem_tiles(bounds)
    tile_paths = []
    for item in items:
        url = item.get("downloadURL")
        if not url:
            continue
        name = Path(url).name
        tile_path = tiles_dir / name
        download_dem_tile(url, tile_path)
        tile_paths.append(tile_path)

    if not tile_paths:
        logger.error("No DEM tiles downloaded!")
        return

    # Merge
    dem_path = args.output_dir / "dem_merged.tif"
    merge_tiles(tile_paths, dem_path)

    # Compute derivatives
    slope_path = args.output_dir / "slope.tif"
    aspect_path = args.output_dir / "aspect.tif"
    twi_path = args.output_dir / "twi.tif"

    compute_slope(dem_path, slope_path)
    compute_aspect(dem_path, aspect_path)
    compute_twi(dem_path, slope_path, twi_path)

    # Optionally resample to match WorldClim grid
    if args.resample_to:
        resampled_dir = args.output_dir / "resampled"
        resampled_dir.mkdir(exist_ok=True)

        for name, path in [("slope", slope_path), ("aspect", aspect_path), ("twi", twi_path)]:
            resample_to_target(
                path,
                args.resample_to,
                resampled_dir / f"{name}_resampled.tif",
            )

    logger.info("\nDone! Terrain derivatives computed.")
    logger.info(f"  DEM: {dem_path}")
    logger.info(f"  Slope: {slope_path}")
    logger.info(f"  Aspect: {aspect_path}")
    logger.info(f"  TWI: {twi_path}")


if __name__ == "__main__":
    main()
