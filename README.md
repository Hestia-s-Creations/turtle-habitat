# Western Pond Turtle Habitat Prediction

Species distribution modeling for Western Pond Turtle (*Actinemys marmorata*) habitat suitability using MaxEnt and ensemble methods with Sentinel-2 imagery and WorldClim bioclimatic variables.

Part of the [Pacific Northwest Conservation AI](https://github.com/Hestia-s-Creations/EEIS) project suite by Hestia's Creations.

## What It Does

Predicts habitat suitability for the Western Pond Turtle across the Pacific Northwest using presence-only occurrence data and environmental covariates. The model identifies high-probability habitat areas to guide field surveys, conservation planning, and land management decisions.

### Key Capabilities

- **MaxEnt species distribution modeling** via scikit-learn (presence-only data)
- **GBIF occurrence data** ingestion with spatial filtering and bias correction
- **WorldClim bioclimatic variables** (19 layers) at 1km resolution
- **Sentinel-2 derived habitat features** - NDVI, NDWI, land cover proxies
- **Spatial block cross-validation** to prevent data leakage from spatial autocorrelation
- **Target-group background sampling** to correct observer bias
- **Suitability maps** with confidence intervals as GeoTIFF outputs

## Target Metrics

| Metric | Target | Published Range |
|--------|--------|----------------|
| AUC | > 0.80 | 0.75-0.95 |
| TSS | > 0.70 | 0.60-0.94 |
| True Positive Rate | > 60% | - |
| Specificity | > 80% | - |

## Data Sources (all free)

| Source | Data | Resolution |
|--------|------|------------|
| GBIF | Turtle occurrence records | Point locations |
| WorldClim | 19 bioclimatic variables | 1km |
| Sentinel-2 | Vegetation/water indices | 10-20m |
| USGS 3DEP | Terrain derivatives | 1m LiDAR |
| Oregon DFW | State survey records | Point locations |

## Project Structure

```
turtle-habitat/
+-- src/
|   +-- data/           # Data acquisition (GBIF, WorldClim, Sentinel-2)
|   +-- features/       # Feature engineering and environmental layers
|   +-- models/         # MaxEnt, Random Forest, ensemble methods
|   +-- evaluation/     # Cross-validation, metrics, spatial assessment
|   +-- visualization/  # Suitability maps, response curves, variable importance
+-- notebooks/          # Exploratory analysis and prototyping
+-- tests/              # Unit and integration tests
+-- configs/            # Model hyperparameters and study area definitions
+-- data/               # Local data cache (gitignored)
+-- results/            # Model outputs and figures (gitignored)
```

## Quick Start

```bash
pip install -r requirements.txt

# Download occurrence data from GBIF
python -m src.data.download_occurrences --species "Actinemys marmorata" --region oregon

# Download environmental layers
python -m src.data.download_worldclim --variables all --resolution 1km
python -m src.data.download_sentinel --study-area configs/oregon_study_area.geojson

# Train model
python -m src.models.train --config configs/maxent_baseline.yaml

# Generate suitability map
python -m src.visualization.predict_map --model results/models/maxent_baseline.pkl
```

## Approach

1. **Occurrence data** from GBIF + Oregon DFW, filtered to research-grade observations
2. **Background sampling** using target-group approach (all herpetofauna observations) to correct for observer bias
3. **Environmental layers**: 19 WorldClim bioclim variables + terrain (slope, aspect, TWI) + Sentinel-2 seasonal composites (NDVI, NDWI)
4. **MaxEnt baseline** via sklearn logistic regression with quadratic features, then Random Forest and XGBoost ensemble
5. **Spatial block cross-validation** (4 train / 1 test geographic tiles) for honest accuracy estimates
6. **Multi-species joint learning** with related herpetofauna (tree frogs, salamanders) for shared niche representations

## Relationship to EEIS

This project shares satellite data infrastructure with [EEIS](https://github.com/Hestia-s-Creations/EEIS) (Environmental Early Impact Scanner). Both use the same Sentinel-2 processing pipeline, PostGIS backend, and React/Leaflet frontend for visualization. The habitat predictions integrate into EEIS's disturbance monitoring to flag when detected changes overlap with high-suitability turtle habitat.

## References

- Original spec: "Pacific Northwest Conservation AI: Technical Specifications for Four Climate-Transferable Projects"
- Phillips et al. (2006) MaxEnt for species distribution modeling
- GBIF occurrence download API: https://www.gbif.org/developer/occurrence

## License

Copyright 2025-2026 Hestia's Creations

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
