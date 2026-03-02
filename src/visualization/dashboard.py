from __future__ import annotations

"""Interactive web dashboard for turtle habitat model results.

Serves a Leaflet-based map with:
- Suitability raster overlay with adjustable opacity
- Threshold classification with area statistics
- Click-to-query suitability values
- Occurrence point markers with environmental feature popups
- Chart.js interactive charts (variable importance, model comparison, per-fold AUC)
- Species ecology information and conservation status

Note: All data rendered in this dashboard is locally-generated model output.
No user-supplied or untrusted content is injected into the HTML.
"""

import argparse
import base64
import http.server
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)

BIOCLIM_LABELS = {
    "bio1": "Annual Mean Temp", "bio2": "Mean Diurnal Range",
    "bio3": "Isothermality", "bio4": "Temp Seasonality",
    "bio5": "Max Temp Warmest Mo", "bio6": "Min Temp Coldest Mo",
    "bio7": "Temp Annual Range", "bio8": "Mean Temp Wettest Qtr",
    "bio9": "Mean Temp Driest Qtr", "bio10": "Mean Temp Warmest Qtr",
    "bio11": "Mean Temp Coldest Qtr", "bio12": "Annual Precipitation",
    "bio13": "Precip Wettest Mo", "bio14": "Precip Driest Mo",
    "bio15": "Precip Seasonality", "bio16": "Precip Wettest Qtr",
    "bio17": "Precip Driest Qtr", "bio18": "Precip Warmest Qtr",
    "bio19": "Precip Coldest Qtr", "slope": "Slope",
    "aspect": "Aspect", "twi": "Topographic Wetness",
}


def raster_to_png_base64(tif_path: Path) -> tuple[str, list]:
    """Convert a GeoTIFF to a transparent PNG data URI and return bounds."""
    from PIL import Image

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
        nodata = src.nodata

    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    valid = (data != nodata) & (data >= 0) & (~np.isnan(data))

    colors = np.array([
        [33, 102, 172, 180],
        [103, 169, 207, 190],
        [209, 229, 240, 200],
        [253, 219, 199, 210],
        [239, 138, 98, 220],
        [178, 24, 43, 240],
    ], dtype=np.uint8)

    vals = np.clip(data[valid], 0, 1)
    indices = vals * (len(colors) - 1)
    lower = np.floor(indices).astype(int)
    upper = np.minimum(lower + 1, len(colors) - 1)
    frac = (indices - lower).reshape(-1, 1)
    interp = colors[lower] * (1 - frac) + colors[upper] * frac
    rgba[valid] = interp.astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    leaflet_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return f"data:image/png;base64,{b64}", leaflet_bounds


def raster_to_grid(tif_path: Path) -> dict:
    """Extract raw suitability values for JS threshold classification and click queries."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        bounds = src.bounds
        nodata = src.nodata
        transform = src.transform

    mask = (data == nodata) | np.isnan(data) | (data < 0)
    data[mask] = -1

    center_lat = (bounds.top + bounds.bottom) / 2
    dx_km = abs(transform.a) * 111.32 * np.cos(np.radians(center_lat))
    dy_km = abs(transform.e) * 110.54
    cell_area_km2 = dx_km * dy_km

    return {
        "grid": np.round(data, 4).tolist(),
        "south": bounds.bottom, "north": bounds.top,
        "west": bounds.left, "east": bounds.right,
        "rows": data.shape[0], "cols": data.shape[1],
        "cellAreaKm2": round(cell_area_km2, 2),
    }


def occurrences_to_geojson(csv_path: Path) -> dict:
    """Convert training CSV to GeoJSON with environmental feature values."""
    df = pd.read_csv(csv_path)
    presence = df[df["presence"] == 1]

    popup_cols = ["bio1", "bio6", "bio11", "bio12", "bio15"]
    features = []
    for _, row in presence.iterrows():
        props = {"type": "presence"}
        for col in popup_cols:
            if col in row.index and pd.notna(row[col]):
                props[col] = round(float(row[col]), 1)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def figure_to_base64(png_path: Path) -> str:
    """Convert a PNG file to a data URI."""
    data = png_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def build_dashboard_html(
    suitability_tif: Path,
    training_csv: Path,
    comparison_csv: Path,
    cv_results_json: Path,
    importance_csv: Path,
    figures_dir: Path,
) -> str:
    """Build the complete dashboard HTML.

    All content is derived from local model outputs -- no untrusted input
    is injected into the generated HTML.
    """
    logger.info("Converting suitability raster to PNG overlay...")
    raster_uri, bounds = raster_to_png_base64(suitability_tif)

    logger.info("Extracting suitability grid for interactive queries...")
    grid = raster_to_grid(suitability_tif)

    logger.info("Converting occurrences to GeoJSON...")
    geojson = occurrences_to_geojson(training_csv)

    logger.info("Loading model comparison data...")
    comparison = pd.read_csv(comparison_csv)
    cv_results = json.loads(cv_results_json.read_text())
    importance = pd.read_csv(importance_csv)

    # Embed response curves figure
    figure_uris = {}
    for name in ["response_curves", "suitability_map"]:
        fig_path = figures_dir / f"{name}.png"
        if fig_path.exists():
            figure_uris[name] = figure_to_base64(fig_path)

    # Prepare JS data objects
    n_presence = len(geojson["features"])
    best_row = comparison.loc[comparison["mean_auc"].idxmax()]
    best_auc = f"{best_row['mean_auc']:.3f}"
    best_tss = f"{comparison['mean_tss'].max():.3f}"

    # Model comparison data for Chart.js
    comparison_data = {
        "models": [],
        "auc": [], "aucStd": [],
        "tss": [], "tssStd": [],
    }
    model_labels = {"maxent": "MaxEnt", "rf": "Random Forest", "gbm": "Gradient Boosting"}
    for _, row in comparison.iterrows():
        comparison_data["models"].append(model_labels.get(row["model"], row["model"]))
        comparison_data["auc"].append(round(float(row["mean_auc"]), 4))
        comparison_data["aucStd"].append(round(float(row["std_auc"]), 4))
        comparison_data["tss"].append(round(float(row["mean_tss"]), 4))
        comparison_data["tssStd"].append(round(float(row["std_tss"]), 4))

    # Per-fold data for Chart.js scatter
    fold_data = {}
    for model_name in ["maxent", "rf", "gbm"]:
        if model_name in cv_results:
            fold_data[model_name] = cv_results[model_name]["folds"]

    # Importance data for Chart.js
    top_n = 15
    imp_data = {"labels": [], "fullLabels": [], "values": []}
    for _, row in importance.head(top_n).iterrows():
        feat = row["feature"]
        label = BIOCLIM_LABELS.get(feat, feat)
        imp_data["labels"].append(feat)
        imp_data["fullLabels"].append(f"{feat} ({label})")
        imp_data["values"].append(round(float(row["importance"]), 5))
    # Reverse for horizontal bar (top at top)
    imp_data["labels"].reverse()
    imp_data["fullLabels"].reverse()
    imp_data["values"].reverse()

    n_features = len(importance)

    # Best model details
    rf_agg = cv_results.get("rf", {}).get("aggregate", {})
    sensitivity = f"{rf_agg.get('mean_sensitivity', 0):.3f}"
    specificity = f"{rf_agg.get('mean_specificity', 0):.3f}"

    # Build HTML panels from trusted data
    targets_html = (
        '<div class="target-row">'
        '<span class="target-label">AUC &ge; 0.80</span>'
        '<span class="badge met">MET</span>'
        '</div>'
        '<div class="target-row">'
        '<span class="target-label">TSS &ge; 0.70</span>'
        f'<span class="badge not-met">NOT MET ({best_tss})</span>'
        '</div>'
    )

    best_model_html = (
        '<div class="detail-grid">'
        '<span>Trees</span><span>500</span>'
        '<span>Min Samples Leaf</span><span>5</span>'
        '<span>Class Weight</span><span>Balanced</span>'
        f'<span>Sensitivity</span><span>{sensitivity}</span>'
        f'<span>Specificity</span><span>{specificity}</span>'
        '</div>'
    )

    figures_html = ""
    if "response_curves" in figure_uris:
        figures_html += (
            '<div class="panel">'
            '<h3>Response Curves</h3>'
            '<p class="panel-desc">Partial dependence plots showing how each top predictor '
            'affects habitat suitability while holding others at their median.</p>'
            '<div class="figure-container" data-figure="response_curves">'
            f'<img src="{figure_uris["response_curves"]}" alt="Response Curves">'
            '</div></div>'
        )

    # Template substitution
    replacements = {
        "__N_PRESENCE__": str(n_presence),
        "__BEST_AUC__": best_auc,
        "__BEST_TSS__": best_tss,
        "__N_FEATURES__": str(n_features),
        "__TARGETS_HTML__": targets_html,
        "__BEST_MODEL_HTML__": best_model_html,
        "__FIGURES_HTML__": figures_html,
        "__SENSITIVITY__": sensitivity,
        "__SPECIFICITY__": specificity,
        "__RASTER_URI__": raster_uri,
        "__BOUNDS_JSON__": json.dumps(bounds),
        "__GRID_JSON__": json.dumps(grid),
        "__GEOJSON_JSON__": json.dumps(geojson),
        "__IMPORTANCE_JSON__": json.dumps(imp_data),
        "__COMPARISON_JSON__": json.dumps(comparison_data),
        "__FOLD_DATA_JSON__": json.dumps(fold_data),
        "__FIGURE_URIS_JSON__": json.dumps(figure_uris),
    }

    html = DASHBOARD_TEMPLATE
    for key, value in replacements.items():
        html = html.replace(key, value)

    return html


DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Western Pond Turtle -- Habitat Suitability</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1923;color:#e0e0e0}

/* Header */
.header{
    background:linear-gradient(135deg,#0f1923 0%,#1a2a3a 100%);
    padding:14px 24px;display:flex;align-items:center;justify-content:space-between;
    border-bottom:2px solid #e94560;position:relative;z-index:1000;
}
.header h1{font-size:1.3em;color:#fff;letter-spacing:0.5px}
.header .subtitle{color:#7a8a9a;font-size:0.8em;margin-top:2px}
.header-stats{display:flex;gap:16px}
.stat-box{
    text-align:center;padding:6px 18px;border-radius:8px;
    background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);
}
.stat-box .val{font-size:1.25em;font-weight:700;color:#4ecca3}
.stat-box .lbl{font-size:0.7em;color:#6a7a8a;margin-top:1px}
.stat-box.warn .val{color:#e94560}

/* Layout */
.main{display:flex;height:calc(100vh - 68px)}
#map{flex:1;min-width:0;background:#0a1015}
.sidebar{
    width:420px;background:#111d2b;overflow-y:auto;
    border-left:1px solid #1e2e3e;display:flex;flex-direction:column;
}

/* Species card */
.species-card{
    padding:14px 18px;background:linear-gradient(135deg,#162030,#1a2840);
    border-bottom:1px solid #1e2e3e;
}
.species-header{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.species-name{font-style:italic;font-weight:600;color:#c0d0e0;font-size:0.95em}
.tag{
    font-size:0.65em;padding:2px 8px;border-radius:10px;font-weight:600;
    text-transform:uppercase;letter-spacing:0.5px;
}
.tag-or{background:rgba(255,165,0,0.15);color:#ffa500;border:1px solid rgba(255,165,0,0.3)}
.tag-wa{background:rgba(233,69,96,0.15);color:#e94560;border:1px solid rgba(233,69,96,0.3)}
.tag-iucn{background:rgba(78,204,163,0.15);color:#4ecca3;border:1px solid rgba(78,204,163,0.3)}
.species-desc{font-size:0.75em;color:#6a7a8a;line-height:1.5}

/* Tabs */
.tab-bar{
    display:flex;gap:2px;padding:10px 16px 0;background:#111d2b;
    position:sticky;top:0;z-index:10;
}
.tab{
    flex:1;padding:8px;text-align:center;border-radius:6px 6px 0 0;
    cursor:pointer;font-size:0.8em;font-weight:500;
    background:rgba(255,255,255,0.03);color:#6a7a8a;
    transition:all 0.2s;border:1px solid transparent;border-bottom:none;
}
.tab:hover{background:rgba(255,255,255,0.06);color:#a0b0c0}
.tab.active{background:#1a2a3a;color:#fff;border-color:#1e2e3e}
.tab-content{display:none;padding:12px 16px 16px;flex:1}
.tab-content.active{display:block}

/* Panels */
.panel{
    background:rgba(255,255,255,0.03);border-radius:8px;
    padding:14px;margin-bottom:12px;border:1px solid rgba(255,255,255,0.04);
}
.panel h3{
    font-size:0.8em;color:#4ecca3;margin-bottom:10px;
    text-transform:uppercase;letter-spacing:1px;font-weight:600;
}
.panel-desc{font-size:0.75em;color:#5a6a7a;line-height:1.5;margin-bottom:10px}

/* Target rows */
.target-row{
    display:flex;justify-content:space-between;align-items:center;
    padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);
}
.target-row:last-child{border-bottom:none}
.target-label{font-size:0.85em;color:#a0b0c0}
.badge{
    font-size:0.75em;font-weight:700;padding:3px 10px;border-radius:4px;
    letter-spacing:0.5px;
}
.badge.met{background:rgba(78,204,163,0.15);color:#4ecca3}
.badge.not-met{background:rgba(233,69,96,0.15);color:#e94560}

/* Detail grid */
.detail-grid{
    display:grid;grid-template-columns:1fr auto;gap:6px 16px;font-size:0.85em;
}
.detail-grid span:nth-child(odd){color:#7a8a9a}
.detail-grid span:nth-child(even){color:#c0d0e0;text-align:right;font-weight:500}

/* Figure container */
.figure-container{
    cursor:pointer;border-radius:6px;overflow:hidden;
    transition:transform 0.2s,box-shadow 0.2s;border:1px solid rgba(255,255,255,0.06);
}
.figure-container:hover{transform:scale(1.01);box-shadow:0 4px 20px rgba(0,0,0,0.3)}
.figure-container img{width:100%;display:block}

/* Map controls */
.map-controls{
    background:rgba(15,25,35,0.92)!important;padding:12px 16px!important;
    border-radius:8px!important;border:1px solid #1e2e3e!important;
    min-width:220px;backdrop-filter:blur(8px);
}
.control-row{
    display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:0.8em;
}
.control-row:last-child{margin-bottom:0}
.control-label{min-width:65px;color:#7a8a9a;font-weight:500}
.slider{
    flex:1;height:4px;-webkit-appearance:none;appearance:none;
    background:#1e2e3e;border-radius:2px;outline:none;cursor:pointer;
}
.slider::-webkit-slider-thumb{
    -webkit-appearance:none;width:14px;height:14px;border-radius:50%;
    background:#4ecca3;cursor:pointer;border:2px solid #0f1923;
}
.slider::-moz-range-thumb{
    width:14px;height:14px;border-radius:50%;
    background:#4ecca3;cursor:pointer;border:2px solid #0f1923;
}
.slider-val{min-width:36px;text-align:right;color:#a0b0c0;font-size:0.9em;font-variant-numeric:tabular-nums}
.area-stats{
    margin-top:6px;padding-top:8px;border-top:1px solid #1e2e3e;
    font-size:0.8em;color:#a0b0c0;line-height:1.6;
}
.area-stats .highlight{color:#4ecca3;font-weight:600}

/* Info box (click-to-query) */
.info-box{
    position:absolute;bottom:80px;left:10px;z-index:1000;
    background:rgba(15,25,35,0.92);padding:10px 14px;border-radius:8px;
    border:1px solid #1e2e3e;backdrop-filter:blur(8px);min-width:150px;
}
.info-title{font-size:0.7em;color:#5a6a7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.info-value{font-size:1.6em;font-weight:700;font-variant-numeric:tabular-nums}
.info-coords{font-size:0.75em;color:#5a6a7a;margin-top:2px}
.info-hint{font-size:0.65em;color:#3a4a5a;margin-top:4px;font-style:italic}

/* Legend */
.legend{
    position:absolute;bottom:24px;left:10px;z-index:1000;
    background:rgba(15,25,35,0.92);padding:10px 14px;border-radius:8px;
    border:1px solid #1e2e3e;font-size:0.8em;backdrop-filter:blur(8px);
}
.legend-title{font-weight:600;margin-bottom:6px;color:#4ecca3;font-size:0.85em}
.legend-bar{
    width:200px;height:12px;
    background:linear-gradient(90deg,#2166ac,#67a9cf,#d1e5f0,#fddbc7,#ef8a62,#b2182b);
    border-radius:3px;margin-bottom:4px;position:relative;
}
.legend-labels{display:flex;justify-content:space-between;font-size:0.8em;color:#5a6a7a}
.threshold-marker{
    position:absolute;top:-3px;width:2px;height:18px;background:#fff;
    transform:translateX(-1px);transition:left 0.15s;display:none;
}

/* Modal */
.modal{
    display:none;position:fixed;top:0;left:0;width:100%;height:100%;
    background:rgba(0,0,0,0.9);z-index:2000;
    justify-content:center;align-items:center;cursor:pointer;
}
.modal.active{display:flex}
.modal img{max-width:92%;max-height:92%;border-radius:8px;box-shadow:0 8px 40px rgba(0,0,0,0.5)}

/* Ecology interpretation */
.ecology-text{font-size:0.8em;line-height:1.6;color:#7a8a9a}
.ecology-text strong{color:#4ecca3;font-weight:600}

/* Chart containers */
.chart-container{position:relative;margin-bottom:4px}

/* Leaflet popup override */
.leaflet-popup-content-wrapper{background:#1a2a3a;color:#c0d0e0;border-radius:8px;border:1px solid #2a3a4a}
.leaflet-popup-tip{background:#1a2a3a}
.leaflet-popup-content{font-size:0.85em;line-height:1.6}
.popup-title{font-weight:600;color:#4ecca3;margin-bottom:4px}
.popup-row{display:flex;justify-content:space-between;gap:16px}
.popup-label{color:#5a6a7a}
.popup-val{color:#c0d0e0;font-weight:500}

/* Scrollbar */
.sidebar::-webkit-scrollbar{width:6px}
.sidebar::-webkit-scrollbar-track{background:#111d2b}
.sidebar::-webkit-scrollbar-thumb{background:#2a3a4a;border-radius:3px}
.sidebar::-webkit-scrollbar-thumb:hover{background:#3a4a5a}
</style>
</head>
<body>

<div class="header">
    <div>
        <h1>Western Pond Turtle</h1>
        <div class="subtitle">Actinemys marmorata -- Habitat Suitability Model</div>
    </div>
    <div class="header-stats">
        <div class="stat-box"><div class="val">__N_PRESENCE__</div><div class="lbl">Presence Records</div></div>
        <div class="stat-box"><div class="val">__BEST_AUC__</div><div class="lbl">Best AUC (RF)</div></div>
        <div class="stat-box warn"><div class="val">__BEST_TSS__</div><div class="lbl">Best TSS (RF)</div></div>
        <div class="stat-box"><div class="val">__N_FEATURES__</div><div class="lbl">Features</div></div>
    </div>
</div>

<div class="main">
    <div id="map"></div>
    <div class="sidebar">
        <div class="species-card">
            <div class="species-header">
                <span class="species-name">Actinemys marmorata</span>
                <span class="tag tag-or">OR Sensitive</span>
                <span class="tag tag-wa">WA Endangered</span>
                <span class="tag tag-iucn">IUCN Vulnerable</span>
            </div>
            <div class="species-desc">Semi-aquatic freshwater turtle of the Pacific coast. Requires slow streams, ponds, and wetlands with basking sites and adjacent upland nesting habitat. Range extends from SW British Columbia to NW Baja California.</div>
        </div>

        <div class="tab-bar">
            <div class="tab active" data-tab="models">Models</div>
            <div class="tab" data-tab="features">Features</div>
            <div class="tab" data-tab="figures">Figures</div>
        </div>

        <div class="tab-content active" id="tab-models">
            <div class="panel">
                <h3>Model Comparison (AUC)</h3>
                <div class="chart-container" style="height:180px">
                    <canvas id="chart-comparison"></canvas>
                </div>
            </div>
            <div class="panel">
                <h3>Per-Fold AUC</h3>
                <p class="panel-desc">Each dot is one spatial block fold. Spread shows geographic sensitivity.</p>
                <div class="chart-container" style="height:160px">
                    <canvas id="chart-folds"></canvas>
                </div>
            </div>
            <div class="panel">
                <h3>Targets</h3>
                __TARGETS_HTML__
            </div>
            <div class="panel">
                <h3>Best Model: Random Forest</h3>
                __BEST_MODEL_HTML__
            </div>
        </div>

        <div class="tab-content" id="tab-features">
            <div class="panel">
                <h3>Variable Importance (Top 15)</h3>
                <div class="chart-container" style="height:380px">
                    <canvas id="chart-importance"></canvas>
                </div>
            </div>
            <div class="panel">
                <h3>Ecological Interpretation</h3>
                <div class="ecology-text">
                    <p>
                        <strong>Cold tolerance dominates:</strong> bio11 (mean temp coldest quarter) and
                        bio6 (min temp coldest month) are the top predictors. Western Pond Turtles
                        are ectotherms whose northern range limit is set by winter severity.
                    </p>
                    <p style="margin-top:8px">
                        <strong>Precipitation seasonality</strong> (bio15) captures the dry-summer / wet-winter
                        Mediterranean climate they prefer. The model identifies a climatic envelope
                        consistent with published range maps.
                    </p>
                    <p style="margin-top:8px">
                        Terrain features (slope, aspect, TWI) rank lowest at ~5km resolution.
                        Finer-scale terrain data (~100m) would better capture microhabitat preferences
                        like basking sites near slow water.
                    </p>
                </div>
            </div>
        </div>

        <div class="tab-content" id="tab-figures">
            __FIGURES_HTML__
        </div>
    </div>
</div>

<div class="legend">
    <div class="legend-title">Habitat Suitability</div>
    <div class="legend-bar"><div class="threshold-marker" id="threshold-marker"></div></div>
    <div class="legend-labels"><span>0 (Low)</span><span>0.5</span><span>1 (High)</span></div>
</div>

<div class="info-box" id="info-box">
    <div class="info-title">Click map to query</div>
    <div class="info-value" id="info-value">--</div>
    <div class="info-coords" id="info-coords"></div>
</div>

<div class="modal" id="modal"><img id="modal-img" src="" alt="Figure"></div>

<script>
// ===== DATA (all from local model outputs, not user input) =====
var RASTER_URI = "__RASTER_URI__";
var BOUNDS = __BOUNDS_JSON__;
var GRID = __GRID_JSON__;
var OCCURRENCES = __GEOJSON_JSON__;
var IMPORTANCE = __IMPORTANCE_JSON__;
var COMPARISON = __COMPARISON_JSON__;
var FOLD_DATA = __FOLD_DATA_JSON__;
var FIGURE_URIS = __FIGURE_URIS_JSON__;

// ===== MAP SETUP =====
var map = L.map('map', {
    center: [44.0, -121.0],
    zoom: 7,
    zoomControl: true,
});

var darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '\u00a9 OpenStreetMap \u00a9 CARTO',
    maxZoom: 19,
}).addTo(map);

var satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: '\u00a9 Esri', maxZoom: 19 }
);

var suitOverlay = L.imageOverlay(RASTER_URI, BOUNDS, {
    opacity: 0.7,
    interactive: false,
}).addTo(map);

// Threshold overlay canvas
var threshCanvas = document.createElement('canvas');
threshCanvas.width = GRID.cols;
threshCanvas.height = GRID.rows;
var threshOverlay = null;

// Presence markers with rich popups
var presenceLayer = L.geoJSON(OCCURRENCES, {
    pointToLayer: function(feature, latlng) {
        return L.circleMarker(latlng, {
            radius: 5,
            fillColor: '#4ecca3',
            color: '#0f1923',
            weight: 1.5,
            opacity: 0.9,
            fillOpacity: 0.8,
        });
    },
    onEachFeature: function(feature, layer) {
        var c = feature.geometry.coordinates;
        var p = feature.properties;

        // Build popup with safe DOM construction
        var div = document.createElement('div');
        var title = document.createElement('div');
        title.className = 'popup-title';
        title.textContent = 'Presence Record';
        div.appendChild(title);

        var addRow = function(label, value) {
            var row = document.createElement('div');
            row.className = 'popup-row';
            var l = document.createElement('span');
            l.className = 'popup-label';
            l.textContent = label;
            var v = document.createElement('span');
            v.className = 'popup-val';
            v.textContent = value;
            row.appendChild(l);
            row.appendChild(v);
            div.appendChild(row);
        };

        addRow('Longitude', c[0].toFixed(4));
        addRow('Latitude', c[1].toFixed(4));

        // Query suitability at this point
        var row = Math.floor((GRID.north - c[1]) / ((GRID.north - GRID.south) / GRID.rows));
        var col = Math.floor((c[0] - GRID.west) / ((GRID.east - GRID.west) / GRID.cols));
        if (row >= 0 && row < GRID.rows && col >= 0 && col < GRID.cols) {
            var sv = GRID.grid[row][col];
            if (sv >= 0) addRow('Suitability', sv.toFixed(3));
        }

        if (p.bio1 !== undefined) addRow('Mean Temp', p.bio1.toFixed(1) + ' \u00b0C');
        if (p.bio6 !== undefined) addRow('Min Cold Mo', p.bio6.toFixed(1) + ' \u00b0C');
        if (p.bio12 !== undefined) addRow('Annual Precip', p.bio12.toFixed(0) + ' mm');

        layer.bindPopup(div);
    }
}).addTo(map);

// Layer control
L.control.layers(
    { "Dark": darkTiles, "Satellite": satellite },
    { "Suitability": suitOverlay, "Presence (__N_PRESENCE__)": presenceLayer },
    { collapsed: false }
).addTo(map);

// ===== MAP CONTROLS =====
var controls = L.control({ position: 'topright' });
controls.onAdd = function() {
    var div = L.DomUtil.create('div', 'map-controls leaflet-bar');

    // Opacity control
    var row1 = document.createElement('div');
    row1.className = 'control-row';
    var lbl1 = document.createElement('span');
    lbl1.className = 'control-label';
    lbl1.textContent = 'Opacity';
    var sl1 = document.createElement('input');
    sl1.type = 'range'; sl1.min = '0'; sl1.max = '100'; sl1.value = '70';
    sl1.className = 'slider';
    var v1 = document.createElement('span');
    v1.className = 'slider-val';
    v1.textContent = '70%';
    sl1.addEventListener('input', function() {
        suitOverlay.setOpacity(this.value / 100);
        v1.textContent = this.value + '%';
    });
    row1.appendChild(lbl1);
    row1.appendChild(sl1);
    row1.appendChild(v1);
    div.appendChild(row1);

    // Threshold control
    var row2 = document.createElement('div');
    row2.className = 'control-row';
    var lbl2 = document.createElement('span');
    lbl2.className = 'control-label';
    lbl2.textContent = 'Threshold';
    var sl2 = document.createElement('input');
    sl2.type = 'range'; sl2.min = '0'; sl2.max = '100'; sl2.value = '0';
    sl2.className = 'slider';
    var v2 = document.createElement('span');
    v2.className = 'slider-val';
    v2.textContent = 'Off';
    sl2.addEventListener('input', function() {
        var t = parseInt(this.value) / 100;
        if (t <= 0.01) {
            v2.textContent = 'Off';
            if (threshOverlay) { map.removeLayer(threshOverlay); threshOverlay = null; }
            var areaEl = document.getElementById('area-stats');
            if (areaEl) areaEl.style.display = 'none';
            document.getElementById('threshold-marker').style.display = 'none';
            return;
        }
        v2.textContent = t.toFixed(2);
        updateThresholdOverlay(t);
    });
    row2.appendChild(lbl2);
    row2.appendChild(sl2);
    row2.appendChild(v2);
    div.appendChild(row2);

    // Area statistics
    var areaDiv = document.createElement('div');
    areaDiv.id = 'area-stats';
    areaDiv.className = 'area-stats';
    areaDiv.style.display = 'none';
    div.appendChild(areaDiv);

    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    return div;
};
controls.addTo(map);

// ===== THRESHOLD OVERLAY =====
function updateThresholdOverlay(threshold) {
    var ctx = threshCanvas.getContext('2d');
    var imgData = ctx.createImageData(GRID.cols, GRID.rows);
    var d = imgData.data;
    var suitable = 0, total = 0;

    for (var r = 0; r < GRID.rows; r++) {
        for (var c = 0; c < GRID.cols; c++) {
            var val = GRID.grid[r][c];
            var idx = (r * GRID.cols + c) * 4;
            if (val >= 0) {
                total++;
                if (val >= threshold) {
                    suitable++;
                    // Green overlay for suitable areas
                    d[idx] = 78; d[idx+1] = 204; d[idx+2] = 163; d[idx+3] = 100;
                }
            }
        }
    }

    ctx.putImageData(imgData, 0, 0);
    var uri = threshCanvas.toDataURL('image/png');

    if (threshOverlay) {
        threshOverlay.setUrl(uri);
    } else {
        threshOverlay = L.imageOverlay(uri, BOUNDS, { opacity: 0.7 }).addTo(map);
    }

    // Update area statistics
    var areaDiv = document.getElementById('area-stats');
    areaDiv.style.display = 'block';
    var areaSuitable = Math.round(suitable * GRID.cellAreaKm2).toLocaleString();
    var pct = total > 0 ? ((suitable / total) * 100).toFixed(1) : '0';

    while (areaDiv.firstChild) areaDiv.removeChild(areaDiv.firstChild);
    var s1 = document.createElement('div');
    var hl = document.createElement('span');
    hl.className = 'highlight';
    hl.textContent = areaSuitable + ' km\u00b2';
    s1.appendChild(document.createTextNode('Suitable: '));
    s1.appendChild(hl);
    s1.appendChild(document.createTextNode(' (' + pct + '%)'));
    areaDiv.appendChild(s1);
    var s2 = document.createElement('div');
    s2.textContent = suitable + ' of ' + total + ' cells above threshold';
    s2.style.color = '#5a6a7a';
    areaDiv.appendChild(s2);

    // Update legend threshold marker
    var marker = document.getElementById('threshold-marker');
    marker.style.display = 'block';
    marker.style.left = (threshold * 100) + '%';
}

// ===== CLICK-TO-QUERY =====
map.on('click', function(e) {
    var lat = e.latlng.lat, lon = e.latlng.lng;
    var row = Math.floor((GRID.north - lat) / ((GRID.north - GRID.south) / GRID.rows));
    var col = Math.floor((lon - GRID.west) / ((GRID.east - GRID.west) / GRID.cols));

    var infoBox = document.getElementById('info-box');
    var valEl = document.getElementById('info-value');
    var coordsEl = document.getElementById('info-coords');

    document.querySelector('.info-title').textContent = 'Suitability';

    if (row >= 0 && row < GRID.rows && col >= 0 && col < GRID.cols) {
        var val = GRID.grid[row][col];
        if (val >= 0) {
            valEl.textContent = val.toFixed(3);
            // Color based on value
            if (val > 0.6) valEl.style.color = '#e94560';
            else if (val > 0.3) valEl.style.color = '#ffa500';
            else valEl.style.color = '#4ecca3';
        } else {
            valEl.textContent = 'No data';
            valEl.style.color = '#3a4a5a';
        }
    } else {
        valEl.textContent = 'Outside';
        valEl.style.color = '#3a4a5a';
    }

    var ns = lat >= 0 ? 'N' : 'S';
    var ew = lon >= 0 ? 'E' : 'W';
    coordsEl.textContent = Math.abs(lat).toFixed(4) + '\u00b0' + ns + ', ' + Math.abs(lon).toFixed(4) + '\u00b0' + ew;
});

// ===== TAB SWITCHING =====
document.querySelectorAll('.tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
        var name = this.getAttribute('data-tab');
        document.querySelectorAll('.tab-content').forEach(function(el) { el.classList.remove('active'); });
        document.querySelectorAll('.tab').forEach(function(el) { el.classList.remove('active'); });
        document.getElementById('tab-' + name).classList.add('active');
        this.classList.add('active');
    });
});

// ===== FIGURE MODAL =====
document.querySelectorAll('.figure-container').forEach(function(el) {
    el.addEventListener('click', function() {
        var name = this.getAttribute('data-figure');
        if (FIGURE_URIS[name]) {
            document.getElementById('modal-img').src = FIGURE_URIS[name];
            document.getElementById('modal').classList.add('active');
        }
    });
});
document.getElementById('modal').addEventListener('click', function() {
    this.classList.remove('active');
});

// ===== CHART.JS =====
Chart.defaults.color = '#7a8a9a';
Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';

// Model comparison grouped bar
(function() {
    var ctx = document.getElementById('chart-comparison').getContext('2d');
    var colors = ['#2166ac', '#4daf4a', '#ff7f00'];
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: COMPARISON.models,
            datasets: [
                {
                    label: 'AUC',
                    data: COMPARISON.auc,
                    backgroundColor: colors.map(function(c) { return c + 'cc'; }),
                    borderColor: colors,
                    borderWidth: 2,
                },
                {
                    label: 'TSS',
                    data: COMPARISON.tss,
                    backgroundColor: colors.map(function(c) { return c + '55'; }),
                    borderColor: colors,
                    borderWidth: 2,
                    borderDash: [4, 4],
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, ticks: { stepSize: 0.2 } },
                x: { grid: { display: false } },
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { usePointStyle: true, pointStyle: 'rect', padding: 12 },
                },
                tooltip: {
                    callbacks: {
                        afterBody: function(items) {
                            var i = items[0].dataIndex;
                            var ds = items[0].datasetIndex;
                            if (ds === 0) return '\u00b1' + COMPARISON.aucStd[i].toFixed(3);
                            return '\u00b1' + COMPARISON.tssStd[i].toFixed(3);
                        }
                    }
                }
            },
        },
    });
})();

// Per-fold AUC scatter
(function() {
    var ctx = document.getElementById('chart-folds').getContext('2d');
    var modelNames = { maxent: 'MaxEnt', rf: 'Random Forest', gbm: 'GBM' };
    var modelColors = { maxent: '#2166ac', rf: '#4daf4a', gbm: '#ff7f00' };
    var datasets = [];
    var offsets = { maxent: -0.15, rf: 0, gbm: 0.15 };

    Object.keys(FOLD_DATA).forEach(function(model) {
        var folds = FOLD_DATA[model];
        var points = folds.map(function(f) {
            return { x: parseInt(f.fold) + (offsets[model] || 0), y: f.auc };
        });
        datasets.push({
            label: modelNames[model] || model,
            data: points,
            backgroundColor: modelColors[model] || '#ccc',
            borderColor: modelColors[model] || '#ccc',
            pointRadius: 7,
            pointHoverRadius: 9,
            showLine: false,
        });
    });

    new Chart(ctx, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0.5, max: 1.0,
                    title: { display: true, text: 'AUC' },
                },
                x: {
                    min: -0.5,
                    max: Math.max.apply(null, Object.keys(FOLD_DATA).map(function(m) {
                        return FOLD_DATA[m].length - 1;
                    })) + 0.5,
                    title: { display: true, text: 'Fold' },
                    ticks: {
                        stepSize: 1,
                        callback: function(v) { return Number.isInteger(v) ? 'Fold ' + v : ''; }
                    },
                },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, pointStyle: 'circle', padding: 12 },
                },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            var d = FOLD_DATA[Object.keys(FOLD_DATA)[ctx.datasetIndex]];
                            var f = d[ctx.dataIndex];
                            return ctx.dataset.label + ': AUC=' + f.auc.toFixed(3) +
                                ', TSS=' + f.tss.toFixed(3) +
                                ' (n=' + f.n_test + ')';
                        }
                    }
                }
            },
        },
    });
})();

// Variable importance horizontal bar
(function() {
    var ctx = document.getElementById('chart-importance').getContext('2d');
    var n = IMPORTANCE.values.length;
    var bgColors = IMPORTANCE.values.map(function(v, i) {
        var t = (n - 1 - i) / Math.max(n - 1, 1);
        var r = Math.round(233 * (1 - t*0.4));
        var g = Math.round(69 + 135 * t);
        var b = Math.round(96 + 67 * t);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.85)';
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: IMPORTANCE.fullLabels,
            datasets: [{
                data: IMPORTANCE.values,
                backgroundColor: bgColors,
                borderWidth: 0,
                borderRadius: 3,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Importance' } },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } },
                },
            },
            plugins: { legend: { display: false } },
        },
    });
})();
</script>
</body>
</html>"""


def serve_dashboard(
    suitability_tif: Path = Path("results/suitability_map.tif"),
    training_csv: Path = Path("data/training/features_with_terrain.csv"),
    comparison_csv: Path = Path("results/comparison/model_comparison.csv"),
    cv_results_json: Path = Path("results/comparison/all_cv_results.json"),
    importance_csv: Path = Path("results/comparison/variable_importance.csv"),
    figures_dir: Path = Path("results/figures"),
    port: int = 8050,
) -> None:
    """Build and serve the dashboard on localhost."""
    html = build_dashboard_html(
        suitability_tif=suitability_tif,
        training_csv=training_csv,
        comparison_csv=comparison_csv,
        cv_results_json=cv_results_json,
        importance_csv=importance_csv,
        figures_dir=figures_dir,
    )

    output = Path("results/dashboard.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    logger.info(f"Dashboard written to {output}")
    logger.info(f"Serving at http://localhost:{port}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output.parent), **kwargs)

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Dashboard ready at http://localhost:{port}/dashboard.html")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Serve interactive habitat model dashboard")
    parser.add_argument("--suitability", type=Path, default=Path("results/suitability_map.tif"))
    parser.add_argument("--training-data", type=Path, default=Path("data/training/features_with_terrain.csv"))
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison/model_comparison.csv"))
    parser.add_argument("--cv-results", type=Path, default=Path("results/comparison/all_cv_results.json"))
    parser.add_argument("--importance", type=Path, default=Path("results/comparison/variable_importance.csv"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    serve_dashboard(
        suitability_tif=args.suitability,
        training_csv=args.training_data,
        comparison_csv=args.comparison,
        cv_results_json=args.cv_results,
        importance_csv=args.importance,
        figures_dir=args.figures,
        port=args.port,
    )


if __name__ == "__main__":
    main()
