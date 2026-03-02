from __future__ import annotations

"""Interactive web dashboard for turtle habitat model results.

Serves a Leaflet-based map with:
- Suitability raster overlay on satellite basemap
- Occurrence point markers (presence/background)
- Variable importance and model comparison charts
- Layer toggle controls

Note: All data rendered in this dashboard is locally-generated model output.
No user-supplied or untrusted content is injected into the HTML.
"""

import argparse
import base64
import http.server
import io
import json
import logging
import threading
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)


def raster_to_png_base64(tif_path: Path) -> tuple[str, list[float]]:
    """Convert a GeoTIFF to a transparent PNG data URI and return bounds."""
    from PIL import Image

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
        nodata = src.nodata

    # Create RGBA image
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Mask nodata
    valid = (data != nodata) & (data >= 0) & (~np.isnan(data))

    # Color ramp: blue -> cyan -> yellow -> orange -> red
    colors = np.array([
        [33, 102, 172, 180],   # 0.0 - dark blue
        [103, 169, 207, 190],  # 0.2 - light blue
        [209, 229, 240, 200],  # 0.4 - very light blue
        [253, 219, 199, 210],  # 0.6 - light orange
        [239, 138, 98, 220],   # 0.8 - orange
        [178, 24, 43, 240],    # 1.0 - dark red
    ], dtype=np.uint8)

    # Map values to colors
    vals = np.clip(data[valid], 0, 1)
    indices = vals * (len(colors) - 1)
    lower = np.floor(indices).astype(int)
    upper = np.minimum(lower + 1, len(colors) - 1)
    frac = (indices - lower).reshape(-1, 1)

    interp = colors[lower] * (1 - frac) + colors[upper] * frac
    rgba[valid] = interp.astype(np.uint8)

    # Convert to PNG
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Leaflet bounds: [[south, west], [north, east]]
    leaflet_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    return f"data:image/png;base64,{b64}", leaflet_bounds


def occurrences_to_geojson(csv_path: Path) -> dict:
    """Convert training CSV to GeoJSON for Leaflet markers."""
    df = pd.read_csv(csv_path)

    features = []
    presence = df[df["presence"] == 1]
    for _, row in presence.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": {
                "type": "presence",
            },
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

    All content is derived from local model outputs — no untrusted input
    is injected into the generated HTML.
    """
    logger.info("Converting suitability raster to PNG overlay...")
    raster_uri, bounds = raster_to_png_base64(suitability_tif)

    logger.info("Converting occurrences to GeoJSON...")
    geojson = occurrences_to_geojson(training_csv)

    logger.info("Loading model comparison data...")
    comparison = pd.read_csv(comparison_csv)
    cv_results = json.loads(cv_results_json.read_text())
    importance = pd.read_csv(importance_csv)

    # Embed figures
    figure_uris = {}
    for name in ["variable_importance", "model_comparison", "response_curves"]:
        fig_path = figures_dir / f"{name}.png"
        if fig_path.exists():
            figure_uris[name] = figure_to_base64(fig_path)

    # Build model stats for the sidebar
    model_stats = []
    for _, row in comparison.iterrows():
        model_stats.append({
            "name": {"maxent": "MaxEnt", "rf": "Random Forest", "gbm": "Gradient Boosting"}.get(row["model"], row["model"]),
            "auc": f"{row['mean_auc']:.3f} +/- {row['std_auc']:.3f}",
            "tss": f"{row['mean_tss']:.3f} +/- {row['std_tss']:.3f}",
            "sensitivity": f"{row['mean_sensitivity']:.3f}",
            "specificity": f"{row['mean_specificity']:.3f}",
            "auc_met": row["mean_auc"] >= 0.80,
            "tss_met": row["mean_tss"] >= 0.70,
        })

    # Top features
    bioclim_labels = {
        "bio1": "Annual Mean Temp", "bio2": "Mean Diurnal Range", "bio3": "Isothermality",
        "bio4": "Temp Seasonality", "bio5": "Max Temp Warmest Mo", "bio6": "Min Temp Coldest Mo",
        "bio7": "Temp Annual Range", "bio8": "Mean Temp Wettest Qtr", "bio9": "Mean Temp Driest Qtr",
        "bio10": "Mean Temp Warmest Qtr", "bio11": "Mean Temp Coldest Qtr", "bio12": "Annual Precip",
        "bio13": "Precip Wettest Mo", "bio14": "Precip Driest Mo", "bio15": "Precip Seasonality",
        "bio16": "Precip Wettest Qtr", "bio17": "Precip Driest Qtr", "bio18": "Precip Warmest Qtr",
        "bio19": "Precip Coldest Qtr", "slope": "Slope", "aspect": "Aspect", "twi": "TWI",
    }
    top_features = []
    max_imp = float(importance.iloc[0]["importance"])
    for _, row in importance.head(10).iterrows():
        feat = row["feature"]
        label = bioclim_labels.get(feat, feat)
        pct = float(row["importance"]) / max_imp * 100
        top_features.append({"name": feat, "label": label, "importance": f"{row['importance']:.4f}", "pct": f"{pct:.0f}"})

    n_presence = len(geojson["features"])
    best_auc = f"{comparison.iloc[comparison['mean_auc'].idxmax()]['mean_auc']:.3f}"
    best_tss = f"{comparison.iloc[comparison['mean_tss'].idxmax()]['mean_tss']:.3f}"

    # Build HTML sections from trusted local data
    model_rows_html = ""
    for m in model_stats:
        auc_class = "met" if m["auc_met"] else "not-met"
        tss_class = "met" if m["tss_met"] else "not-met"
        model_rows_html += (
            f'<div class="model-row">'
            f'<span class="model-name">{m["name"]}</span>'
            f'<span class="{auc_class}">AUC {m["auc"]}</span>'
            f'<span class="{tss_class}">TSS {m["tss"]}</span>'
            f'</div>\n'
        )

    feature_bars_html = ""
    for f in top_features:
        feature_bars_html += (
            f'<div class="feature-bar">'
            f'<span class="name">{f["name"]}</span>'
            f'<div class="bar-container"><div class="bar" style="width: {f["pct"]}%"></div></div>'
            f'<span class="val">{f["importance"]}</span>'
            f'</div>\n'
        )

    figures_html = ""
    for name, uri in figure_uris.items():
        title = name.replace("_", " ").title()
        figures_html += (
            f'<div class="panel">'
            f'<h3>{title}</h3>'
            f'<div class="figure-container" data-figure="{name}">'
            f'<img src="{uri}" alt="{title}">'
            f'</div></div>\n'
        )

    # Serialize data for JavaScript (all from local model outputs)
    bounds_json = json.dumps(bounds)
    geojson_json = json.dumps(geojson)
    figure_uris_json = json.dumps(figure_uris)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Western Pond Turtle - Habitat Suitability Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }}
.header {{
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 2px solid #e94560;
    z-index: 1000; position: relative;
}}
.header h1 {{ font-size: 1.4em; color: #fff; }}
.header .subtitle {{ color: #aaa; font-size: 0.85em; margin-top: 2px; }}
.header .stats {{ display: flex; gap: 24px; }}
.stat-box {{
    text-align: center; padding: 4px 16px; border-radius: 8px;
    background: rgba(255,255,255,0.08);
}}
.stat-box .value {{ font-size: 1.3em; font-weight: bold; color: #4ecca3; }}
.stat-box .label {{ font-size: 0.75em; color: #aaa; }}
.stat-box.warn .value {{ color: #e94560; }}
.main-container {{ display: flex; height: calc(100vh - 72px); }}
#map {{ flex: 1; min-width: 0; }}
.sidebar {{
    width: 380px; background: #16213e; overflow-y: auto;
    border-left: 1px solid #333; padding: 16px;
}}
.panel {{
    background: rgba(255,255,255,0.05); border-radius: 8px;
    padding: 14px; margin-bottom: 14px;
}}
.panel h3 {{
    font-size: 0.9em; color: #4ecca3; margin-bottom: 10px;
    text-transform: uppercase; letter-spacing: 1px;
}}
.model-row {{
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.85em;
}}
.model-row:last-child {{ border-bottom: none; }}
.model-name {{ font-weight: 600; min-width: 100px; }}
.met {{ color: #4ecca3; }}
.not-met {{ color: #e94560; }}
.feature-bar {{
    display: flex; align-items: center; margin-bottom: 6px; font-size: 0.8em;
}}
.feature-bar .name {{ min-width: 50px; color: #ccc; }}
.feature-bar .bar-container {{
    flex: 1; height: 16px; background: rgba(255,255,255,0.08);
    border-radius: 3px; margin: 0 8px; overflow: hidden;
}}
.feature-bar .bar {{
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #e94560, #ff6b6b);
}}
.feature-bar .val {{ min-width: 50px; text-align: right; color: #888; }}
.figure-container {{
    cursor: pointer; border-radius: 8px; overflow: hidden;
    margin-bottom: 10px; transition: transform 0.2s;
}}
.figure-container:hover {{ transform: scale(1.02); }}
.figure-container img {{ width: 100%; display: block; }}
.legend {{
    position: absolute; bottom: 30px; left: 10px;
    background: rgba(22, 33, 62, 0.92); padding: 10px 14px;
    border-radius: 8px; z-index: 1000; font-size: 0.8em; border: 1px solid #333;
}}
.legend-title {{ font-weight: 600; margin-bottom: 6px; color: #4ecca3; }}
.legend-bar {{
    width: 200px; height: 14px;
    background: linear-gradient(90deg, #2166ac, #67a9cf, #d1e5f0, #fddbc7, #ef8a62, #b2182b);
    border-radius: 3px; margin-bottom: 4px;
}}
.legend-labels {{ display: flex; justify-content: space-between; font-size: 0.85em; color: #aaa; }}
.modal {{
    display: none; position: fixed; top: 0; left: 0;
    width: 100%; height: 100%; background: rgba(0,0,0,0.85);
    z-index: 2000; justify-content: center; align-items: center; cursor: pointer;
}}
.modal.active {{ display: flex; }}
.modal img {{ max-width: 90%; max-height: 90%; border-radius: 8px; }}
.tab-bar {{ display: flex; gap: 4px; margin-bottom: 12px; }}
.tab {{
    flex: 1; padding: 8px; text-align: center;
    background: rgba(255,255,255,0.05); border-radius: 6px;
    cursor: pointer; font-size: 0.8em; transition: background 0.2s;
}}
.tab:hover {{ background: rgba(255,255,255,0.1); }}
.tab.active {{ background: #e94560; color: #fff; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>Western Pond Turtle</h1>
        <div class="subtitle">Actinemys marmorata &mdash; Habitat Suitability Model</div>
    </div>
    <div class="stats">
        <div class="stat-box"><div class="value">{n_presence}</div><div class="label">Presence Records</div></div>
        <div class="stat-box"><div class="value">{best_auc}</div><div class="label">Best AUC (RF)</div></div>
        <div class="stat-box warn"><div class="value">{best_tss}</div><div class="label">Best TSS (RF)</div></div>
        <div class="stat-box"><div class="value">22</div><div class="label">Features</div></div>
    </div>
</div>
<div class="main-container">
    <div id="map"></div>
    <div class="sidebar">
        <div class="tab-bar">
            <div class="tab active" id="btn-models">Models</div>
            <div class="tab" id="btn-features">Features</div>
            <div class="tab" id="btn-figures">Figures</div>
        </div>
        <div id="tab-models" class="tab-content active">
            <div class="panel"><h3>Spatial Block CV (5-fold)</h3>{model_rows_html}</div>
            <div class="panel"><h3>Targets</h3>
                <div class="model-row"><span>AUC &ge; 0.80</span><span class="met">MET</span></div>
                <div class="model-row"><span>TSS &ge; 0.70</span><span class="not-met">NOT MET ({best_tss})</span></div>
            </div>
            <div class="panel"><h3>Best Model: Random Forest</h3>
                <div class="model-row"><span>Trees</span><span>500</span></div>
                <div class="model-row"><span>Min Samples Leaf</span><span>5</span></div>
                <div class="model-row"><span>Class Weight</span><span>Balanced</span></div>
                <div class="model-row"><span>Sensitivity</span><span>{model_stats[1]["sensitivity"]}</span></div>
                <div class="model-row"><span>Specificity</span><span>{model_stats[1]["specificity"]}</span></div>
            </div>
        </div>
        <div id="tab-features" class="tab-content">
            <div class="panel"><h3>Top 10 Predictors</h3>{feature_bars_html}</div>
            <div class="panel"><h3>Ecological Interpretation</h3>
                <p style="font-size:0.8em; line-height:1.5; color:#aaa;">
                    Cold tolerance dominates: <strong style="color:#4ecca3">bio11</strong> (mean temp coldest quarter)
                    and <strong style="color:#4ecca3">bio6</strong> (min temp coldest month) are the top predictors.
                    Western Pond Turtles are ectotherms limited by winter severity.
                    <strong style="color:#4ecca3">bio15</strong> (precipitation seasonality) captures the
                    dry-summer / wet-winter Mediterranean climate they prefer.
                    Terrain features (slope, aspect, TWI) rank lowest at 5km resolution.
                </p>
            </div>
        </div>
        <div id="tab-figures" class="tab-content">{figures_html}</div>
    </div>
</div>
<div class="legend">
    <div class="legend-title">Habitat Suitability</div>
    <div class="legend-bar"></div>
    <div class="legend-labels"><span>0.0 (Unsuitable)</span><span>0.5</span><span>1.0 (Highly Suitable)</span></div>
</div>
<div class="modal" id="modal"><img id="modal-img" src="" alt="Full size figure"></div>
<script>
// All data below is generated from local model outputs, not user input.
var suitabilityBounds = {bounds_json};
var occurrences = {geojson_json};
var figureUris = {figure_uris_json};

var map = L.map('map', {{ center: [44.0, -121.0], zoom: 7, zoomControl: true }});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19,
}}).addTo(map);

var satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution: '&copy; Esri', maxZoom: 19 }}
);

var suitabilityOverlay = L.imageOverlay('{raster_uri}', suitabilityBounds, {{
    opacity: 0.7, interactive: false,
}}).addTo(map);

var presenceLayer = L.geoJSON(occurrences, {{
    pointToLayer: function(feature, latlng) {{
        return L.circleMarker(latlng, {{
            radius: 5, fillColor: '#4ecca3', color: '#000',
            weight: 1, opacity: 0.9, fillOpacity: 0.8,
        }});
    }},
    onEachFeature: function(feature, layer) {{
        var c = feature.geometry.coordinates;
        layer.bindPopup('Presence Record<br>Lon: ' + c[0].toFixed(4) + '<br>Lat: ' + c[1].toFixed(4));
    }}
}}).addTo(map);

var baseMaps = {{
    "Dark": L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{maxZoom: 19}}),
    "Satellite": satellite,
}};
var overlays = {{
    "Suitability": suitabilityOverlay,
    "Presence ({n_presence})": presenceLayer,
}};
L.control.layers(baseMaps, overlays, {{collapsed: false}}).addTo(map);

var oc = L.control({{position: 'topright'}});
oc.onAdd = function() {{
    var d = L.DomUtil.create('div', 'leaflet-bar');
    d.style.cssText = 'background:rgba(22,33,62,0.9);padding:8px 12px;border-radius:6px;color:#ccc;font-size:0.8em';
    var lbl = document.createElement('label');
    lbl.textContent = 'Opacity: ';
    var slider = document.createElement('input');
    slider.type = 'range'; slider.min = '0'; slider.max = '100'; slider.value = '70';
    slider.style.cssText = 'width:100px;vertical-align:middle';
    slider.addEventListener('input', function() {{ suitabilityOverlay.setOpacity(this.value / 100); }});
    lbl.appendChild(slider);
    d.appendChild(lbl);
    L.DomEvent.disableClickPropagation(d);
    return d;
}};
oc.addTo(map);

// Tab switching via event delegation
document.getElementById('btn-models').addEventListener('click', function() {{ switchTab('models', this); }});
document.getElementById('btn-features').addEventListener('click', function() {{ switchTab('features', this); }});
document.getElementById('btn-figures').addEventListener('click', function() {{ switchTab('figures', this); }});

function switchTab(name, btn) {{
    document.querySelectorAll('.tab-content').forEach(function(el) {{ el.classList.remove('active'); }});
    document.querySelectorAll('.tab').forEach(function(el) {{ el.classList.remove('active'); }});
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
}}

// Figure click -> modal (using event delegation)
document.querySelectorAll('.figure-container').forEach(function(el) {{
    el.addEventListener('click', function() {{
        var name = this.getAttribute('data-figure');
        if (figureUris[name]) {{
            document.getElementById('modal-img').src = figureUris[name];
            document.getElementById('modal').classList.add('active');
        }}
    }});
}});
document.getElementById('modal').addEventListener('click', function() {{
    this.classList.remove('active');
}});
</script>
</body>
</html>"""
    return html


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
