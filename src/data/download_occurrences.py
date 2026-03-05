from __future__ import annotations

"""Download Western Pond Turtle occurrence records from GBIF.

Uses the GBIF REST API to fetch Actinemys marmorata and A. pallida
observations (plus historical synonyms), filters to research-grade
records in the Pacific Northwest, and applies spatial deduplication
to reduce observer bias.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# GBIF API
GBIF_API = "https://api.gbif.org/v1"

# Western Pond Turtle species keys
# Actinemys marmorata (current accepted name - northern species)
# Actinemys pallida (southern species, recognized in 2019 taxonomic split)
# Emys marmorata (former name pre-2001)
# Clemmys marmorata (historical name)
SPECIES_NAMES = [
    "Actinemys marmorata",
    "Actinemys pallida",
    "Emys marmorata",
    "Clemmys marmorata",
]

# Pacific Northwest bounding box (OR, WA, Northern CA)
PNW_BBOX = {
    "decimalLatitude": "41.0,49.0",
    "decimalLongitude": "-125.0,-116.0",
}

# Oregon-only bounding box
OREGON_BBOX = {
    "decimalLatitude": "41.9,46.3",
    "decimalLongitude": "-124.6,-116.5",
}

REGIONS = {
    "pnw": PNW_BBOX,
    "oregon": OREGON_BBOX,
}


def get_species_key(name: str) -> int | None:
    """Look up GBIF taxon key for a species name."""
    resp = requests.get(
        f"{GBIF_API}/species/match",
        params={"name": name, "strict": True},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("matchType") == "NONE":
        return None
    return data.get("usageKey")


def fetch_occurrences(
    species_key: int,
    region: dict[str, str],
    limit: int = 300,
    offset: int = 0,
) -> dict:
    """Fetch a page of occurrence records from GBIF."""
    params = {
        "taxonKey": species_key,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "limit": min(limit, 300),  # GBIF max per page
        "offset": offset,
        **region,
    }
    resp = requests.get(f"{GBIF_API}/occurrence/search", params=params)
    resp.raise_for_status()
    return resp.json()


def fetch_all_occurrences(
    species_key: int,
    region: dict[str, str],
) -> list[dict]:
    """Fetch all occurrence pages for a species in a region."""
    all_records = []
    offset = 0
    limit = 300

    while True:
        data = fetch_occurrences(species_key, region, limit=limit, offset=offset)
        results = data.get("results", [])
        all_records.extend(results)

        total = data.get("count", 0)
        logger.info(
            f"  Fetched {len(all_records)}/{total} records (offset={offset})"
        )

        if data.get("endOfRecords", True) or len(results) == 0:
            break

        offset += limit
        time.sleep(0.5)  # Be polite to GBIF API

    return all_records


def parse_records(raw_records: list[dict]) -> pd.DataFrame:
    """Extract relevant fields from raw GBIF records."""
    rows = []
    for r in raw_records:
        lat = r.get("decimalLatitude")
        lon = r.get("decimalLongitude")
        if lat is None or lon is None:
            continue

        rows.append({
            "gbif_id": r.get("key"),
            "species": r.get("species", r.get("acceptedScientificName", "")),
            "queried_name": r.get("_queried_name", ""),
            "latitude": lat,
            "longitude": lon,
            "coordinate_uncertainty_m": r.get("coordinateUncertaintyInMeters"),
            "year": r.get("year"),
            "month": r.get("month"),
            "day": r.get("day"),
            "basis_of_record": r.get("basisOfRecord"),
            "institution": r.get("institutionCode"),
            "dataset": r.get("datasetName"),
            "country": r.get("country"),
            "state_province": r.get("stateProvince"),
            "county": r.get("county"),
            "issue_flags": ";".join(r.get("issues", [])),
        })

    df = pd.DataFrame(rows)
    logger.info(f"  Parsed {len(df)} records with valid coordinates")
    return df


def spatial_deduplicate(
    df: pd.DataFrame,
    resolution_km: float = 1.0,
) -> pd.DataFrame:
    """Remove duplicate observations in the same grid cell.

    At 1km resolution, this prevents multiple observations at the same
    site from inflating model performance. Keeps the most recent record.
    """
    if df.empty:
        return df

    # Approximate grid cell at given resolution
    # 1 degree latitude ~ 111 km
    cell_size = resolution_km / 111.0
    df = df.copy()
    df["grid_lat"] = (df["latitude"] / cell_size).round(0)
    df["grid_lon"] = (df["longitude"] / cell_size).round(0)

    # Keep most recent record per grid cell
    df_sorted = df.sort_values("year", ascending=False, na_position="last")
    df_dedup = df_sorted.drop_duplicates(subset=["grid_lat", "grid_lon"], keep="first")
    df_dedup = df_dedup.drop(columns=["grid_lat", "grid_lon"])

    n_removed = len(df) - len(df_dedup)
    logger.info(
        f"  Spatial dedup at {resolution_km}km: {len(df)} -> {len(df_dedup)} "
        f"({n_removed} duplicates removed)"
    )
    return df_dedup


def download_target_group(
    region: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Download all herpetofauna observations for target-group background sampling.

    Target-group background uses observations of related species to model
    observer effort, correcting for the bias that observations cluster near
    roads, trails, and population centers.
    """
    # Order Testudines (turtles) + Order Caudata (salamanders) + Order Anura (frogs)
    # GBIF higher taxon keys
    herpetofauna_keys = {
        "Testudines": 789,      # Turtles
        "Caudata": 952,         # Salamanders
        "Anura": 750,           # Frogs
        "Squamata": 715,        # Lizards/snakes
    }

    all_records = []
    for taxon_name, taxon_key in herpetofauna_keys.items():
        logger.info(f"Fetching {taxon_name} (key={taxon_key}) for background sampling")
        records = fetch_all_occurrences(taxon_key, region)
        all_records.extend(records)
        logger.info(f"  {taxon_name}: {len(records)} records")
        time.sleep(1.0)

    df = parse_records(all_records)

    # Save raw
    output_path = output_dir / "target_group_background.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} target-group background points to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download Western Pond Turtle occurrences from GBIF"
    )
    parser.add_argument(
        "--species",
        default="Actinemys marmorata",
        help="Species name to search (default: Actinemys marmorata)",
    )
    parser.add_argument(
        "--region",
        choices=["pnw", "oregon"],
        default="oregon",
        help="Geographic region to filter (default: oregon)",
    )
    parser.add_argument(
        "--dedup-resolution",
        type=float,
        default=1.0,
        help="Spatial deduplication grid size in km (default: 1.0)",
    )
    parser.add_argument(
        "--include-target-group",
        action="store_true",
        help="Also download herpetofauna observations for background sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/occurrences"),
        help="Output directory (default: data/occurrences)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    region = REGIONS[args.region]

    # Resolve species key and tag records with queried name for provenance
    all_records = []
    for name in SPECIES_NAMES:
        key = get_species_key(name)
        if key:
            logger.info(f"Found species key for {name}: {key}")
            records = fetch_all_occurrences(key, region)
            for r in records:
                r["_queried_name"] = name
            all_records.extend(records)
        else:
            logger.warning(f"No GBIF match for {name}")

    if not all_records:
        logger.error("No occurrence records found!")
        return

    # Parse and deduplicate
    df = parse_records(all_records)

    # Remove records with high coordinate uncertainty (>1km)
    if "coordinate_uncertainty_m" in df.columns:
        before = len(df)
        df = df[
            df["coordinate_uncertainty_m"].isna()
            | (df["coordinate_uncertainty_m"] <= 1000)
        ]
        logger.info(
            f"  Filtered high-uncertainty records: {before} -> {len(df)}"
        )

    # Deduplicate by GBIF ID first (cross-name duplicates)
    df = df.drop_duplicates(subset=["gbif_id"])

    # Spatial deduplication
    df = spatial_deduplicate(df, resolution_km=args.dedup_resolution)

    # Save
    raw_path = args.output_dir / "turtle_occurrences_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved {len(df)} occurrence records to {raw_path}")

    # Summary stats
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    logger.info(f"Lat range: {df['latitude'].min():.2f} - {df['latitude'].max():.2f}")
    logger.info(f"Lon range: {df['longitude'].min():.2f} - {df['longitude'].max():.2f}")
    if "state_province" in df.columns:
        logger.info(f"States: {df['state_province'].value_counts().to_dict()}")

    # Target-group background
    if args.include_target_group:
        download_target_group(region, args.output_dir)


if __name__ == "__main__":
    main()
