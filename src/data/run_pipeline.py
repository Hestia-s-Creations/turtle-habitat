from __future__ import annotations

"""Run the full data acquisition pipeline.

Downloads all data sources, computes features, and assembles
the training-ready dataset in one command.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline(
    region: str = "oregon",
    data_dir: Path = Path("data"),
    n_background: int = 10000,
    include_target_group: bool = True,
    include_terrain: bool = True,
    resolution: str = "10m",
    verbose: bool = False,
):
    """Execute the full data pipeline."""
    from src.data.download_occurrences import (
        REGIONS,
        SPECIES_NAMES,
        download_target_group,
        fetch_all_occurrences,
        get_species_key,
        parse_records,
        spatial_deduplicate,
    )
    from src.data.download_worldclim import (
        STUDY_AREAS,
        build_feature_stack,
        download_bioclim,
    )
    from src.features.assemble import assemble_training_data

    # === Step 1: Download occurrence data ===
    logger.info("=" * 60)
    logger.info("STEP 1: Download occurrence data from GBIF")
    logger.info("=" * 60)

    occ_dir = data_dir / "occurrences"
    occ_dir.mkdir(parents=True, exist_ok=True)
    gbif_region = REGIONS[region]

    all_records = []
    for name in SPECIES_NAMES:
        key = get_species_key(name)
        if key:
            logger.info(f"Fetching {name} (key={key})")
            records = fetch_all_occurrences(key, gbif_region)
            all_records.extend(records)
        else:
            logger.warning(f"No GBIF match for {name}")

    if not all_records:
        logger.error("No occurrence records found! Aborting.")
        sys.exit(1)

    df = parse_records(all_records)
    df = df[
        df["coordinate_uncertainty_m"].isna()
        | (df["coordinate_uncertainty_m"] <= 1000)
    ]
    df = df.drop_duplicates(subset=["gbif_id"])
    df = spatial_deduplicate(df, resolution_km=1.0)

    occ_path = occ_dir / "turtle_occurrences_raw.csv"
    df.to_csv(occ_path, index=False)
    logger.info(f"Saved {len(df)} occurrences to {occ_path}")

    # Target group background
    bg_path = None
    if include_target_group:
        logger.info("\nDownloading target-group background observations...")
        download_target_group(gbif_region, occ_dir)
        bg_path = occ_dir / "target_group_background.csv"

    # === Step 2: Download WorldClim ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Download WorldClim bioclimatic variables")
    logger.info("=" * 60)

    wc_dir = data_dir / "worldclim"
    wc_dir.mkdir(parents=True, exist_ok=True)
    bounds = STUDY_AREAS[region]

    raw_dir = download_bioclim(wc_dir, resolution=resolution)
    stack_path = build_feature_stack(raw_dir, wc_dir, bounds)

    # === Step 3: Download terrain (optional) ===
    terrain_dir = None
    if include_terrain:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Download terrain derivatives")
        logger.info("=" * 60)

        from src.data.download_terrain import compute_terrain_features

        terrain_dir = data_dir / "terrain"
        compute_terrain_features(terrain_dir, bounds, resolution=resolution)

    # === Step 4: Assemble training data ===
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Assemble training dataset")
    logger.info("=" * 60)

    training_path = data_dir / "training" / "features.csv"
    result = assemble_training_data(
        occurrence_csv=occ_path,
        bioclim_stack=stack_path,
        terrain_dir=terrain_dir,
        background_csv=bg_path,
        n_background=n_background,
        output_path=training_path,
    )

    # === Summary ===
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Occurrences: {occ_path}")
    logger.info(f"Bioclim stack: {stack_path}")
    if terrain_dir:
        logger.info(f"Terrain: {terrain_dir}")
    logger.info(f"Training data: {training_path}")
    logger.info(f"  Presence: {(result['presence'] == 1).sum()}")
    logger.info(f"  Background: {(result['presence'] == 0).sum()}")
    feature_cols = [c for c in result.columns if c not in ("longitude", "latitude", "presence")]
    logger.info(f"  Features: {len(feature_cols)}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the full data acquisition and feature assembly pipeline"
    )
    parser.add_argument(
        "--region",
        choices=["oregon", "pnw"],
        default="oregon",
        help="Study area (default: oregon)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=10000,
        help="Number of background points (default: 10000)",
    )
    parser.add_argument(
        "--resolution",
        choices=["10m", "5m", "2.5m", "30s"],
        default="10m",
        help="WorldClim resolution (default: 10m). 10m=fast test, 2.5m=real modeling, 30s=production",
    )
    parser.add_argument(
        "--skip-target-group",
        action="store_true",
        help="Skip downloading herpetofauna target-group background",
    )
    parser.add_argument(
        "--skip-terrain",
        action="store_true",
        help="Skip downloading and computing terrain derivatives",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_pipeline(
        region=args.region,
        data_dir=args.data_dir,
        n_background=args.n_background,
        include_target_group=not args.skip_target_group,
        include_terrain=not args.skip_terrain,
        resolution=args.resolution,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
