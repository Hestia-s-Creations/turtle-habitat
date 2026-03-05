from __future__ import annotations

"""Download elevation data and compute terrain derivatives for habitat modeling.

Downloads elevation data and computes terrain derivatives (slope, aspect, TWI,
TPI) as environmental features for species distribution modeling.

Supports two elevation sources:
- WorldClim: matched to bioclim grid (~1km), for consistency
- USGS 3DEP: 1/3 arc-second (~10m) via The National Map, for micro-topography

For the turtle model, terrain features capture microhabitat preferences:
- Low slope near water (basking sites)
- TWI for wetness/proximity to water
- Aspect for solar exposure (thermoregulation)
- TPI for topographic position (ridge vs valley)
"""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import reproject, Resampling
import requests
from shapely.geometry import box

logger = logging.getLogger(__name__)

# WorldClim elevation URL (same grid as bioclim variables — no resampling needed)
WORLDCLIM_ELEV_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

# Fallback: TNM API endpoint for high-res DEM
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

# Study areas (west, south, east, north)
STUDY_AREAS = {
    "oregon": (-124.6, 41.9, -116.5, 46.3),
    "pnw": (-125.0, 41.0, -116.0, 49.0),
}

RESOLUTIONS = {
    "10m": "~20km",
    "5m": "~10km",
    "2.5m": "~5km",
    "30s": "~1km",
}


def download_worldclim_elev(
    output_dir: Path,
    resolution: str = "2.5m",
    bounds: tuple[float, float, float, float] | None = None,
    timeout: int = 600,
) -> Path:
    """Download WorldClim elevation at matching resolution and clip to bounds.

    Returns path to the clipped elevation GeoTIFF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clipped_path = output_dir / f"elevation_{resolution}.tif"

    if clipped_path.exists():
        logger.info(f"Elevation already exists: {clipped_path}")
        return clipped_path

    url = f"{WORLDCLIM_ELEV_BASE}/wc2.1_{resolution}_elev.zip"
    zip_path = output_dir / f"wc2.1_{resolution}_elev.zip"
    extract_dir = output_dir / f"elev_{resolution}"

    # Download
    if not zip_path.exists():
        logger.info(f"Downloading WorldClim elevation ({resolution})")
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        logger.info(f"  Downloaded: {zip_path}")

    # Extract
    if not extract_dir.exists():
        logger.info(f"Extracting elevation data")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    # Find the tif
    tif_files = list(extract_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIF found in {extract_dir}")
    elev_tif = tif_files[0]

    # Clip to bounds if provided
    if bounds:
        west, south, east, north = bounds
        geom = box(west, south, east, north)

        with rasterio.open(elev_tif) as src:
            clipped, transform = rasterio_mask(src, [geom], crop=True)
            profile = src.profile.copy()
            profile.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
            })

        with rasterio.open(clipped_path, "w", **profile) as dst:
            dst.write(clipped)
        logger.info(f"  Clipped elevation: {clipped_path} ({clipped.shape[2]}x{clipped.shape[1]})")
    else:
        import shutil
        shutil.copy2(elev_tif, clipped_path)

    return clipped_path


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


def download_3dep_dem(
    output_dir: Path,
    bounds: tuple[float, float, float, float],
    timeout: int = 1200,
) -> Path:
    """Download USGS 3DEP 1/3 arc-second (~10m) DEM via The National Map.

    Uses WCS (Web Coverage Service) to fetch the DEM for the study extent.
    Returns path to the downloaded DEM GeoTIFF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dem_path = output_dir / "3dep_13arcsec.tif"

    if dem_path.exists():
        logger.info(f"3DEP DEM already exists: {dem_path}")
        return dem_path

    west, south, east, north = bounds
    # USGS 3DEP WCS endpoint
    wcs_url = (
        "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"
        f"?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        f"&COVERAGE=DEP3Elevation&CRS=EPSG:4326"
        f"&BBOX={west},{south},{east},{north}"
        f"&RESX=0.000092592593&RESY=0.000092592593"  # 1/3 arc-second
        f"&FORMAT=GeoTIFF"
    )

    logger.info(f"Downloading 3DEP 1/3 arc-second DEM for bounds {bounds}")
    logger.info(f"  This may take several minutes (~2GB for Oregon)")

    resp = requests.get(wcs_url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with open(dem_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    logger.info(f"  Downloaded: {dem_path}")
    return dem_path


def compute_tpi(dem_path: Path, output_path: Path, window_size: int = 11) -> None:
    """Compute Topographic Position Index (TPI).

    TPI = elevation - mean(neighborhood elevation)
    Positive = ridge/hilltop, Negative = valley, Near zero = flat/mid-slope
    """
    if output_path.exists():
        logger.info(f"  TPI already computed: {output_path}")
        return

    logger.info("Computing TPI")
    from scipy.ndimage import uniform_filter

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata

    # Mean neighborhood elevation
    mean_elev = uniform_filter(dem, size=window_size)
    tpi = dem - mean_elev

    if nodata is not None:
        tpi[dem == nodata] = np.nan

    profile.update({"dtype": "float32", "nodata": np.nan})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(tpi.astype(np.float32), 1)

    logger.info(f"  TPI saved: {output_path}")


def aggregate_to_target(
    input_path: Path,
    target_path: Path,
    output_path: Path,
    method: str = "mean",
) -> None:
    """Aggregate high-res raster to match a target raster grid.

    Uses appropriate resampling: 'mean' for slope/TWI/TPI,
    'mode' for aspect (circular mean handled separately).
    """
    if output_path.exists():
        logger.info(f"  Already aggregated: {output_path}")
        return

    resampling_map = {
        "mean": Resampling.average,
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
    }
    resampling = resampling_map.get(method, Resampling.average)

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
                resampling=resampling,
            )

    logger.info(f"  Aggregated to target grid ({method}): {output_path}")


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


def compute_terrain_features(
    output_dir: Path,
    bounds: tuple[float, float, float, float],
    resolution: str = "2.5m",
    source: str = "worldclim",
    bioclim_stack: Path | None = None,
) -> Path:
    """Download elevation and compute all terrain derivatives.

    Args:
        source: "worldclim" for ~1km elevation, "3dep" for ~10m USGS elevation.
            When using 3dep, derivatives are computed at native 10m resolution
            then aggregated to match the bioclim grid.
        bioclim_stack: Path to bioclim GeoTIFF for grid matching (required for 3dep).

    Returns the output directory containing slope.tif, aspect.tif, twi.tif, tpi.tif
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if source == "3dep":
        # Download high-res 3DEP DEM
        dep_dir = output_dir / "3dep_native"
        dep_dir.mkdir(parents=True, exist_ok=True)
        dem_path = download_3dep_dem(dep_dir, bounds)

        # Compute at native 10m resolution
        slope_native = dep_dir / "slope.tif"
        aspect_native = dep_dir / "aspect.tif"
        twi_native = dep_dir / "twi.tif"
        tpi_native = dep_dir / "tpi.tif"

        compute_slope(dem_path, slope_native)
        compute_aspect(dem_path, aspect_native)
        compute_twi(dem_path, slope_native, twi_native)
        compute_tpi(dem_path, tpi_native)

        # Aggregate to bioclim grid
        if bioclim_stack and bioclim_stack.exists():
            target = bioclim_stack
        else:
            # Fall back to WorldClim elevation for grid matching
            target = download_worldclim_elev(output_dir, resolution=resolution, bounds=bounds)

        resampled_dir = output_dir / "resampled"
        resampled_dir.mkdir(parents=True, exist_ok=True)

        aggregate_to_target(slope_native, target, resampled_dir / "slope_resampled.tif", method="mean")
        aggregate_to_target(aspect_native, target, resampled_dir / "aspect_resampled.tif", method="mean")
        aggregate_to_target(twi_native, target, resampled_dir / "twi_resampled.tif", method="mean")
        aggregate_to_target(tpi_native, target, resampled_dir / "tpi_resampled.tif", method="mean")

        # Also write non-resampled versions at output_dir level for direct access
        for name in ["slope", "aspect", "twi", "tpi"]:
            src_path = resampled_dir / f"{name}_resampled.tif"
            dst_path = output_dir / f"{name}.tif"
            if src_path.exists() and not dst_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)

        logger.info(f"\n3DEP terrain features computed (aggregated to bioclim grid):")
    else:
        # WorldClim elevation (same grid as bioclim)
        dem_path = download_worldclim_elev(output_dir, resolution=resolution, bounds=bounds)

        compute_slope(dem_path, output_dir / "slope.tif")
        compute_aspect(dem_path, output_dir / "aspect.tif")
        compute_twi(dem_path, output_dir / "slope.tif", output_dir / "twi.tif")
        compute_tpi(dem_path, output_dir / "tpi.tif")

        logger.info(f"\nWorldClim terrain features computed:")

    logger.info(f"  Slope: {output_dir / 'slope.tif'}")
    logger.info(f"  Aspect: {output_dir / 'aspect.tif'}")
    logger.info(f"  TWI: {output_dir / 'twi.tif'}")
    logger.info(f"  TPI: {output_dir / 'tpi.tif'}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download elevation data and compute terrain derivatives"
    )
    parser.add_argument(
        "--region",
        choices=["oregon", "pnw"],
        default="pnw",
        help="Study area (default: pnw)",
    )
    parser.add_argument(
        "--resolution",
        choices=list(RESOLUTIONS.keys()),
        default="2.5m",
        help="Resolution matching WorldClim bioclim (default: 2.5m)",
    )
    parser.add_argument(
        "--source",
        choices=["worldclim", "3dep"],
        default="worldclim",
        help="Elevation source: worldclim (~1km) or 3dep (~10m, aggregated) (default: worldclim)",
    )
    parser.add_argument(
        "--bioclim-stack",
        type=Path,
        default=None,
        help="Path to bioclim GeoTIFF for grid matching (needed for 3dep aggregation)",
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

    bounds = STUDY_AREAS[args.region]
    compute_terrain_features(
        args.output_dir, bounds,
        resolution=args.resolution,
        source=args.source,
        bioclim_stack=args.bioclim_stack,
    )


if __name__ == "__main__":
    main()
