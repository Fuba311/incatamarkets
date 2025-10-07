# app.py
# --- Lightweight production-ready Dash app (local + Render) ---
print("--- STARTING INCATA MARKET ANALYSIS DASHBOARD ---")

import os
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State, MATCH

# ------------------------------------------------------------------------------
# 1) APP INIT
# ------------------------------------------------------------------------------
app = dash.Dash(__name__, assets_folder='assets', title="INCATA Market Analysis Dashboard")
server = app.server

# Minimal, embedded CSS
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root{
        --brand:#004085; --brand-soft:#e7f3ff; --ink:#333; --muted:#6c757d;
        --card:#f0f8ff; --border:#cce5ff; --shadow:0 4px 18px rgba(0,0,0,0.08);
        --panel-bg:#fff; --panel-header:#e9f5ff;
      }
      body{font-family:"Segoe UI","Roboto",Arial,sans-serif;background:#f8f9fa;}
      .muted{color:var(--muted);}
      .section-card{
        background:var(--card); border:1px solid var(--border); border-radius:12px;
        padding:24px; box-shadow:var(--shadow); margin-bottom:38px;
      }
      .floating-controls{
        position:absolute; top:16px; left:8px; width:280px; background:var(--panel-bg);
        border:1px solid #e9ecef; border-radius:10px; box-shadow:var(--shadow);
        overflow:visible !important; z-index:1000; transition:width .2s ease;
      }
      .floating-controls.icon-only{ width:46px; overflow:hidden !important; }
      .control-panel-header{
        display:flex; align-items:center; gap:8px; padding:10px 12px;
        background:var(--panel-header); border-bottom:1px solid #d6ebff; cursor:pointer;
      }
      .control-panel-header .header-icon{font-size:16px;}
      .control-panel-header .header-text{font-weight:600;color:var(--brand);}
      .toggle-btn{
        margin-left:auto; border:1px solid var(--brand); background:var(--brand-soft);
        color:var(--brand); border-radius:6px; font-size:12px; padding:2px 8px; cursor:pointer;
      }
      .controls-content{ padding:12px; max-height:70vh; overflow:visible !important; }
      .controls-content.hidden{display:none;}
      #network-map-title{
        position:absolute; top:10px; left:50%; transform:translateX(-50%);
        background:rgba(255,255,255,.95); padding:8px 20px; border-radius:20px;
        font-size:14px; font-weight:700; color:var(--brand);
        box-shadow:0 2px 8px rgba(0,0,0,.12); border:1px solid rgba(0,64,133,.2); z-index:999;
      }
      #network-info-collapse{ backdrop-filter:saturate(140%) blur(2px); }
      @keyframes fadeIn{ from{opacity:0} to{opacity:1} }
      .lifted-dropdown, .dash-dropdown, .Select{ z-index:3000 !important; }
      .lifted-dropdown .Select-menu-outer,
      .dash-dropdown .Select-menu-outer,
      .Select-menu-outer{ z-index:5000 !important; overflow:visible !important; }
      .lifted-dropdown .Select-menu, .Select-menu{ z-index:5000 !important; }
      .lifted-dropdown .Select__menu{ z-index:5000 !important; }
      @media (max-width:768px){
        .floating-controls{ top:56px; left:6px; width:88vw; }
        .floating-controls.icon-only{ width:44px; }
        .controls-content{ overflow-y:auto !important; }
      }
      .app-hero h1{ text-align:center; color:var(--brand); margin-bottom:4px; }
      .app-hero h4{ text-align:center; font-weight:normal; margin-top:2px; }

      :root{ --title-top: 14px; }
      .title-wrap{
        position: absolute; top: var(--title-top); left: 50%;
        transform: translateX(-50%); z-index: 1200; text-align: center;
      }
      .title-chip{
        background: rgba(255,255,255,.95); padding: 8px 20px; border-radius: 20px;
        font-size: 14px; font-weight: 700; color: #004085;
        box-shadow: 0 2px 8px rgba(0,0,0,.12);
        border: 1px solid rgba(0,64,133,.2); display: inline-block;
      }
      .title-spinner{ margin-top: 18px; }
      .title-spinner .dash-spinner{ transform: scale(.9); opacity:.9; }
      #network-map-title, #combined-map-title{
        position: static !important; top: auto !important; left: auto !important;
        transform: none !important; display: inline-block !important;
      }
      .floating-controls .Select__menu {
      top: auto !important;
      bottom: calc(100% + 6px) !important;
      max-height: 240px !important;
      overflow-y: auto !important;
      }
      /* Open the Trade Routes opacity dropdown upward (old + new dash) */
      #opacity-dropdown .Select { position: relative !important; }

      #opacity-dropdown .Select-menu-outer,
      #opacity-dropdown .Select__menu {
      top: auto !important;
      bottom: calc(100% + 6px) !important;
      max-height: 240px !important;
      overflow-y: auto !important;
      transform: none !important; /* cancel any JS transform */
      }

    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# ------------------------------------------------------------------------------
# 2) DATA LOADING
# ------------------------------------------------------------------------------
print("--- Loading Pre-Processed Data ---")
PROCESSED_DATA_FOLDER = Path(__file__).parent / "processed_data"

network_df = None
market_volume_df = None
trader_df = None
business_df = None  # NEW
roads_data = {}
nightlights_data = {}
data_load_success = False

def _ensure_data_uri(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    return s if s.startswith("data:image") else f"data:image/png;base64,{s}"

def _normalize_coords(coords):
    """Return corners as [[lonW,latN],[lonE,latN],[lonE,latS],[lonW,latS]]; fix [lat,lon] if needed."""
    if not coords or len(coords) != 4:
        return None
    fixed = []
    for c in coords:
        if not isinstance(c, (list, tuple)) or len(c) != 2: return None
        lon, lat = c
        if abs(lon) <= 90 and abs(lat) > 90:
            lon, lat = lat, lon
        fixed.append([float(lon), float(lat)])
    pts = [{"i": i, "lon": p[0], "lat": p[1]} for i, p in enumerate(fixed)]
    top2 = sorted(sorted(pts, key=lambda x: x["lat"], reverse=True)[:2], key=lambda x: x["lon"])
    bot2 = sorted(sorted(pts, key=lambda x: x["lat"])[:2], key=lambda x: x["lon"])
    return [[top2[0]["lon"], top2[0]["lat"]], [top2[1]["lon"], top2[1]["lat"]],
            [bot2[1]["lon"], bot2[1]["lat"]], [bot2[0]["lon"], bot2[0]["lat"]]]

def _default_bbox_coords():
    """If no coords in JSON, fit to data extent (with padding)."""
    lats, lons = [], []
    if network_df is not None:
        for c in ("origin_lat", "market_lat"):
            if c in network_df.columns: lats += pd.to_numeric(network_df[c], errors="coerce").dropna().tolist()
        for c in ("origin_lon", "market_lon"):
            if c in network_df.columns: lons += pd.to_numeric(network_df[c], errors="coerce").dropna().tolist()
    if trader_df is not None:
        if "lat" in trader_df.columns: lats += pd.to_numeric(trader_df["lat"], errors="coerce").dropna().tolist()
        if "lon" in trader_df.columns: lons += pd.to_numeric(trader_df["lon"], errors="coerce").dropna().tolist()
    if business_df is not None:
        if "lat" in business_df.columns: lats += pd.to_numeric(business_df["lat"], errors="coerce").dropna().tolist()
        if "lon" in business_df.columns: lons += pd.to_numeric(business_df["lon"], errors="coerce").dropna().tolist()
    if lats and lons:
        latS, latN = min(lats), max(lats)
        lonW, lonE = min(lons), max(lons)
        pad_lat = max(0.1, (latN - latS) * 0.05)
        pad_lon = max(0.1, (lonE - lonW) * 0.05)
        return [[lonW - pad_lon, latN + pad_lat], [lonE + pad_lon, latN + pad_lat],
                [lonE + pad_lon, latS - pad_lat], [lonW - pad_lon, latS - pad_lat]]
    return [[33.9, 5.2], [41.9, 5.2], [41.9, -4.9], [33.9, -4.9]]

def _norm_key(s: str) -> str:
    return str(s).strip().lower().replace("_", "").replace(" ", "")

def _get_nl_image_and_coords(store, selected_time):
    """
    Accepts flexible shapes; returns (image_data_uri, coords_list)
    """
    if isinstance(store, str):
        return _ensure_data_uri(store), _default_bbox_coords()

    if isinstance(store, dict):
        pretty = selected_time
        alt = {"Now": "now", "5 Yrs Ago": "5_yrs_ago", "10 Yrs Ago": "10_yrs_ago"}.get(selected_time, selected_time)
        targets = {_norm_key(pretty), _norm_key(alt)}

        chosen_val = None
        for k, v in store.items():
            if _norm_key(k) in targets:
                chosen_val = v
                break
        if chosen_val is None and len(store) == 1:
            chosen_val = next(iter(store.values()))
        if chosen_val is None:
            return None, None

        if isinstance(chosen_val, str):
            return _ensure_data_uri(chosen_val), _default_bbox_coords()
        if isinstance(chosen_val, (list, tuple)) and len(chosen_val) == 2:
            img, coords = chosen_val
            return _ensure_data_uri(img), (_normalize_coords(coords) or _default_bbox_coords())
        if isinstance(chosen_val, dict):
            img = chosen_val.get("image") or chosen_val.get("img")
            coords = chosen_val.get("coordinates") or chosen_val.get("coords")
            return _ensure_data_uri(img), (_normalize_coords(coords) or _default_bbox_coords())

    return None, None

try:
    # Core tabular data
    network_df = pd.read_parquet(PROCESSED_DATA_FOLDER / "network_df.parquet")
    market_volume_df = pd.read_parquet(PROCESSED_DATA_FOLDER / "market_volume_df.parquet")
    trader_df = pd.read_parquet(PROCESSED_DATA_FOLDER / "trader_df.parquet")

    # NEW: businesses
    business_path = PROCESSED_DATA_FOLDER / "business_df.parquet"
    if business_path.exists():
        business_df = pd.read_parquet(business_path)
        print("SUCCESS: Business table loaded.")
        print("Business types:", sorted([x for x in business_df["business_label"].dropna().unique()]))

    print("SUCCESS: Parquet tables loaded.")

    # Preprocessed geospatial overlays
    time_period_map = {"10 Yrs Ago": "10_yrs_ago", "5 Yrs Ago": "5_yrs_ago", "Now": "now"}
    for pretty, suffix in time_period_map.items():
        road_path = PROCESSED_DATA_FOLDER / f"roads_{suffix}_processed.geojson"
        if road_path.exists():
            with open(road_path, "r", encoding="utf-8") as f:
                roads_data[pretty] = json.load(f)
        else:
            print(f"Warning: Road file not found at {road_path}")
    if roads_data:
        print("SUCCESS: Roads GeoJSON overlays loaded.")

    # Nightlights: support a few filenames
    nl_candidates = [
        PROCESSED_DATA_FOLDER / "nightlights_data_.json",
        PROCESSED_DATA_FOLDER / "nightlights_data.json",
        PROCESSED_DATA_FOLDER / "nightlights.json",
    ]
    for nl_path in nl_candidates:
        if nl_path.exists():
            with open(nl_path, "r", encoding="utf-8") as f:
                nightlights_data = json.load(f)
            print(f"SUCCESS: Nightlights overlay loaded from {nl_path.name}")
            break
    else:
        print("Warning: No nightlights JSON found (tried nightlights_data_.json, nightlights_data.json, nightlights.json)")

    print("--- All data loaded successfully ---")
    print("Unique trader types:", sorted([x for x in trader_df["trader_id"].dropna().unique()]))
    data_load_success = True

except FileNotFoundError as e:
    print(f"---! FATAL ERROR !---: {e}")
    print("Please upload the 'processed_data' folder with required files.")
except Exception as e:
    print(f"---! UNEXPECTED ERROR !---: {e}")

# ------------------------------------------------------------------------------
# 3) CONSTANTS & SMALL HELPERS
# ------------------------------------------------------------------------------
section_style = {
    "background-color": "#f0f8ff",
    "border": "1px solid #cce5ff",
    "border-radius": "12px",
    "padding": "25px",
    "box-shadow": "0 4px 18px rgba(0,0,0,0.08)",
    "margin-bottom": "38px",
}
title_style = {"textAlign": "center", "color": "#333333", "marginBottom": "16px"}

MAPBOX_STYLE_LIGHT = "carto-positron"
MAPBOX_STYLE_DARK = "carto-darkmatter"

GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d", "zoomIn2d", "zoomOut2d",
        "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"
    ],
}

def _safe_bool_contains(container, value):
    return bool(container) and (value in container)

# ------------------------------------------------------------------------------
# 4) LAYOUT
# ------------------------------------------------------------------------------
if data_load_success:
    # business UI options
    biz_options = [{"label": "All Businesses", "value": "All"}]
    biz_disabled = business_df is None or business_df.empty
    if not biz_disabled:
        biz_options += [{"label": b, "value": b} for b in sorted(business_df["business_label"].dropna().unique())]

    app.layout = html.Div(
        style={"padding": "2% 5%"},
        children=[
            # HERO
            html.Div(
                className="app-hero",
                children=[
                    html.H1("INCATA Market Analysis Dashboard"),
                    html.H4("Markets studied under Project INCATA"),
                    html.P(
                        "INCATA: Linked Farms and Enterprises for Inclusive Agricultural Transformation in Africa and Asia",
                        className="muted",
                        style={"textAlign": "center", "marginTop": "2px"},
                    ),
                ],
                style={"marginBottom": "28px"},
            ),

            # GLOBAL FILTER
            html.Div(
                style={
                    "background-color": "#e2e3e5",
                    "padding": "14px",
                    "border-radius": "10px",
                    "margin-bottom": "46px",
                },
                children=[
                    html.Label(
                        "Global Filter: Select Market Type",
                        style={"fontWeight": "bold", "display": "block", "color": "#495057", "marginBottom": "8px"},
                    ),
                    dcc.Dropdown(
                        id="master-market-type-filter",
                        className="lifted-dropdown",
                        options=[{"label": "All Markets", "value": "All Markets"}]
                        + [{"label": m, "value": m} for m in sorted(network_df["mkt_type"].dropna().unique())],
                        value="All Markets",
                    ),
                ],
            ),

            # PRODUCE FLOW NETWORK MAP
            html.Div(
                className="section-card",
                children=[
                    html.H2("Produce Flow Network", style={"color": "#004085", "border-bottom": "2px solid #b8daff", "padding-bottom": "10px"}),
                    html.P("Map shows tomato flows from origins to markets. Origins (red) are approximate locations (often county-level centroids).", style={"marginBottom": "16px"}),
                    html.Div(
                        style={"position": "relative", "width": "100%"},
                        children=[
                            # Floating controls
                            html.Div(
                                id={"type": "floating-panel-wrapper", "index": "network"},
                                className="floating-controls",
                                children=[
                                    html.Div(
                                        id={"type": "panel-header", "index": "network"},
                                        className="control-panel-header",
                                        n_clicks=0,
                                        children=[
                                            html.Span("⚙️", className="header-icon"),
                                            html.Span("Map Controls", className="header-text"),
                                            html.Button("−", id="network-toggle-controls-btn", className="toggle-btn"),
                                        ],
                                    ),
                                    html.Div(
                                        id={"type": "panel-content", "index": "network"},
                                        className="controls-content",
                                        children=[
                                            # TIME + SEASON
                                            html.Div(
                                                style={"marginBottom": "16px"},
                                                children=[
                                                    html.Label("📅 Time Period", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Slider(id="network-time-slider", min=0, max=2, step=None, included=False, value=2,
                                                               marks={0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"}),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "16px"},
                                                children=[
                                                    html.Label("🌱 Season", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.RadioItems(
                                                        id="season-toggle",
                                                        options=[{"label": " High Season", "value": "High Season"}, {"label": " Low Season", "value": "Low Season"}],
                                                        value="High Season",
                                                        labelStyle={"display": "block", "marginBottom": "5px", "fontSize": "12px"},
                                                    ),
                                                ],
                                            ),
                                            # LAYERS
                                            html.Div(
                                                style={"marginBottom": "16px"},
                                                children=[
                                                    html.Label("🗺 Map Layers", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Checklist(
                                                        id="layer-toggles",
                                                        options=[
                                                            {"label": " Markets & Origins", "value": "show_markers"},
                                                            {"label": " Roads", "value": "show_roads"},
                                                            {"label": " Nightlights", "value": "show_nightlights"},
                                                        ],
                                                        value=["show_markers"],
                                                        labelStyle={"display": "block", "marginBottom": "5px", "fontSize": "12px"},
                                                    ),
                                                ],
                                            ),
                                            # ROUTES
                                            html.Div(
                                                style={"marginBottom": "12px"},
                                                children=[
                                                    html.Label("🔗 Trade Routes", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Checklist(id="toggle-routes", options=[{"label": " Show Trade Routes", "value": "show"}],
                                                                  value=["show"], labelStyle={"fontSize": "12px", "marginBottom": "8px"}),
                                                    html.Div(
                                                        style={"display": "flex", "alignItems": "center", "gap": "10px", "marginTop": "10px"},
                                                        children=[
                                                            html.Label("Opacity:", style={"fontSize": "12px", "minWidth": "50px"}),
                                                            dcc.Dropdown(
                                                                id="opacity-dropdown", className="lifted-dropdown",
                                                                options=[{"label": f"{i}%", "value": i} for i in range(0, 101, 10)],
                                                                value=70, clearable=False, searchable=False, style={"width": "80px", "fontSize": "11px"},
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                            # SIMPLIFY BUTTON
                                            html.Div(
                                                style={"marginBottom": "14px"},
                                                children=[
                                                    html.Button(
                                                        "Simplify Map: Off",
                                                        id="simplify-map-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "width": "100%",
                                                            "cursor": "pointer",
                                                            "border": "1px solid #004085",
                                                            "backgroundColor": "#f8f9fa",
                                                            "padding": "7px 10px",
                                                            "borderRadius": "6px",
                                                            "fontSize": "12px",
                                                            "color": "#004085",
                                                            "fontWeight": "600",
                                                        },
                                                    )
                                                ],
                                            ),
                                            # INFO BUTTON
                                            html.Div(
                                                style={"borderTop": "1px solid #e0e0e0", "paddingTop": "10px", "marginTop": "10px"},
                                                children=[html.Button("ℹ How to Read This Map", id="network-info-button", n_clicks=0,
                                                                      style={"width": "100%", "cursor": "pointer", "border": "1px solid #004085",
                                                                             "backgroundColor": "#e7f3ff", "padding": "6px 10px", "borderRadius": "6px", "fontSize": "12px"})],
                                            ),
                                            # --- NEW: BUSINESSES LAYER ---
                                            html.Div(
                                                style={"borderTop": "1px solid #e0e0e0", "marginTop": "14px", "paddingTop": "12px"},
                                                children=[
                                                    html.Label("🏬 Businesses", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.Checklist(
                                                        id="biz-toggle",
                                                        options=[{"label": " Show Businesses", "value": "show"}],
                                                        value=[],  # off by default to keep the map clean
                                                        labelStyle={"fontSize": "12px", "marginBottom": "8px"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="biz-type-dropdown", className="lifted-dropdown",
                                                        options=biz_options, value="All", clearable=False,
                                                        placeholder="Select business type", disabled=biz_disabled,
                                                    ),
                                                    html.Div(style={"height": "8px"}),
                                                    dcc.RadioItems(
                                                        id="biz-view-mode",
                                                        options=[{"label": " Points", "value": "points"}, {"label": " Heatmap", "value": "heatmap"}],
                                                        value="points", inline=True,
                                                    ),
                                                    html.Div(style={"height": "8px"}),
                                                    html.Label("Opacity", style={"fontSize": "12px"}),
                                                    dcc.Slider(id="biz-opacity", min=10, max=100, step=5, value=70,
                                                               marks={10: "10%", 70: "70%", 100: "100%"}),
                                                    html.Div(className="muted", style={"fontSize": "11px", "marginTop": "6px"},
                                                             children="Size = number of businesses (by selected time). Color = nearest distance (km)."),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            # Centered title + tiny spinner
                            html.Div(
                                className="title-wrap",
                                children=[
                                    html.Div(id="network-map-title", className="title-chip"),
                                    html.Div(
                                        className="title-spinner",
                                        children=dcc.Loading(
                                            id="network-loading",
                                            type="dot",
                                            children=html.Div(id="network-loading-sentinel", style={"width": 1, "height": 1}),
                                        ),
                                    ),
                                ],
                            ),

                            # Map
                            dcc.Graph(id="network-map", style={"height": "85vh", "width": "100%"}, config=GRAPH_CONFIG),

                            # Info bubble
                            html.Div(
                                id="network-info-collapse",
                                style={"display": "none", "position": "absolute", "bottom": "10px", "right": "10px",
                                       "background-color": "rgba(248, 249, 250, 0.95)", "padding": "15px",
                                       "border": "1px dashed #cce5ff", "borderRadius": "6px", "zIndex": "998", "maxWidth": "420px"},
                                children=[
                                    html.Button("✕", id="close-info-btn", n_clicks=0,
                                                style={"position": "absolute", "top": "6px", "right": "10px",
                                                       "background": "transparent", "border": "none", "fontSize": "18px", "cursor": "pointer"}),
                                    dcc.Markdown(
                                        "* **Red Dots (Produce Origins):** County/area where tomatoes are sourced.\n"
                                        "* **Blue Dots (Markets):** Markets where tomatoes are sold.\n"
                                        "* **Lines (Trade Routes):** Connections from origin to market.\n"
                                        "* **Line Thickness:** Represents the share of produce from that origin.\n"
                                        "* **Business Halos (optional):** Circle size shows number of businesses; color shows nearest distance (km).",
                                        style={"fontSize": "12px", "margin": "0"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # MARKET CONCENTRATION / TRADERS (unchanged)
            html.Div(
                className="section-card",
                children=[
                    html.H2("Market Concentration Analysis", style={"color": "#004085", "border-bottom": "2px solid #b8daff", "padding-bottom": "10px"}),
                    html.P("Analyze tomato trade volume or trader concentration. Switch type and view to explore patterns.", style={"marginBottom": "20px"}),

                    html.Div(
                        style={"position": "relative", "width": "100%"},
                        children=[
                            html.Div(
                                id={"type": "floating-panel-wrapper", "index": "combined"},
                                className="floating-controls",
                                children=[
                                    html.Div(
                                        id={"type": "panel-header", "index": "combined"},
                                        className="control-panel-header",
                                        n_clicks=0,
                                        children=[
                                            html.Span("⚙️", className="header-icon"),
                                            html.Span("Analysis Controls", className="header-text"),
                                            html.Button("−", id="combined-toggle-controls-btn", className="toggle-btn"),
                                        ],
                                    ),
                                    html.Div(
                                        id={"type": "panel-content", "index": "combined"},
                                        className="controls-content",
                                        children=[
                                            html.Div(
                                                style={"marginBottom": "18px"},
                                                children=[
                                                    html.Label("📊 Analysis Type", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.RadioItems(
                                                        id="data-type-toggle",
                                                        options=[{"label": " Tomatoes", "value": "tomatoes"}, {"label": " Traders", "value": "traders"}],
                                                        value="tomatoes", inline=True, labelStyle={"marginRight": "14px"},
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "18px"},
                                                children=[
                                                    html.Label("🎨 View Style", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.RadioItems(
                                                        id="view-type-toggle",
                                                        options=[{"label": " Points", "value": "points"}, {"label": " Heatmap", "value": "heatmap"}],
                                                        value="points", inline=True, labelStyle={"marginRight": "14px"},
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="season-control-div", className="conditional-control", style={"marginBottom": "18px"},
                                                children=[
                                                    html.Label("🌱 Season", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.RadioItems(
                                                        id="combined-season-toggle",
                                                        options=[{"label": " High", "value": "High Season"}, {"label": " Low", "value": "Low Season"}],
                                                        value="High Season", inline=True,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="trader-control-div", className="conditional-control", style={"marginBottom": "18px"},
                                                children=[
                                                    html.Label("👤 Trader Type", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.Dropdown(
                                                        id="combined-trader-type-dropdown", className="lifted-dropdown",
                                                        options=[{"label": "All Traders", "value": "All"}] + [
                                                            {"label": t, "value": t} for t in sorted(trader_df["trader_id"].dropna().unique())
                                                        ],
                                                        value="All", placeholder="Select...",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "6px"},
                                                children=[
                                                    html.Label("📅 Time Period", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Slider(id="combined-time-slider", min=0, max=2, step=None, included=False, value=2,
                                                               marks={0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"}),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            html.Div(
                                className="title-wrap",
                                children=[
                                    html.Div(id="combined-map-title", className="title-chip"),
                                    html.Div(
                                        className="title-spinner",
                                        children=dcc.Loading(
                                            id="combined-loading",
                                            type="dot",
                                            children=html.Div(id="combined-loading-sentinel", style={"width": 1, "height": 1}),
                                        ),
                                    ),
                                ],
                            ),

                            dcc.Graph(id="combined-map", style={"height": "85vh"}, config=GRAPH_CONFIG),
                        ],
                    ),
                ],
            ),

            # FOOTER
            html.Footer(
                [
                    html.P(
                        "The INCATA project is funded by the Gates Foundation.",
                        className="muted",
                        style={"fontSize": "0.9em"},
                    ),
                ],
                style={"textAlign": "center", "padding": "18px 0", "marginTop": "24px", "borderTop": "1px solid #dee2e6"},
            ),
        ],
    )

# ------------------------------------------------------------------------------
# 5) CALLBACKS
# ------------------------------------------------------------------------------
if data_load_success:

    @app.callback(
        [Output({"type": "floating-panel-wrapper", "index": MATCH}, "className"),
         Output({"type": "panel-content", "index": MATCH}, "className")],
        Input({"type": "panel-header", "index": MATCH}, "n_clicks"),
        State({"type": "floating-panel-wrapper", "index": MATCH}, "className"),
        prevent_initial_call=True,
    )
    def toggle_panel_animation(n, current_class):
        if n and n > 0:
            if current_class and "icon-only" in current_class:
                return "floating-controls", "controls-content"
            else:
                return "floating-controls icon-only", "controls-content hidden"
        return no_update, no_update

    @app.callback(
        Output("network-info-collapse", "style"),
        [Input("network-info-button", "n_clicks"), Input("close-info-btn", "n_clicks")],
        State("network-info-collapse", "style"),
        prevent_initial_call=True,
    )
    def toggle_network_info(info_clicks, close_clicks, current_style):
        ctx = callback_context
        if not ctx.triggered:
            return current_style
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "network-info-button" and (not current_style or current_style.get("display") == "none"):
            base = current_style.copy() if current_style else {}
            base.update({"display": "block", "animation": "fadeIn 0.3s"})
            return base
        base = current_style.copy() if current_style else {}
        base.update({"display": "none"})
        return base

# --------------------------- NETWORK MAP -----------------------------------
    @app.callback(
    [
        Output("network-map", "figure"),
        Output("network-map-title", "children"),
        Output("network-loading-sentinel", "children"),  # drives the spinner
        Output("simplify-map-btn", "children"),
    ],
    [
        Input("master-market-type-filter", "value"),
        Input("season-toggle", "value"),
        Input("network-time-slider", "value"),
        Input("opacity-dropdown", "value"),
        Input("toggle-routes", "value"),
        Input("layer-toggles", "value"),
        Input("simplify-map-btn", "n_clicks"),
        # Businesses UI (ensure these exist in your layout)
        Input("biz-toggle", "value"),            # checklist ['show'] or []
        Input("biz-type-dropdown", "value"),     # 'All' or a specific business label
        Input("biz-view-mode", "value"),         # 'points' or 'heatmap'
        Input("biz-opacity", "value"),           # 10..100 slider
    ],
    [State("network-map", "relayoutData")],
    )
    def update_network_map(selected_market_type, selected_season, time_value, opacity_percent,
                        toggle_value, layer_toggles, simplify_clicks, biz_toggle, biz_type, biz_view_mode, biz_opacity,
                        relayout_data):

        # ---------------- selections / defaults ----------------
        time_map = {0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"}
        selected_time = time_map.get(time_value, "Now")
        layer_toggles = layer_toggles or []
        if opacity_percent is None:
            opacity_percent = 70
        if biz_opacity is None:
            biz_opacity = 70
        show_businesses = bool(biz_toggle) and ("show" in biz_toggle)
        biz_label_for_hover = biz_type if (biz_type and biz_type != "All") else "All businesses"
        simplify_clicks = simplify_clicks or 0
        simplify_mode = (simplify_clicks % 2 == 1)
        simplify_label = "Simplify Map: On" if simplify_mode else "Simplify Map: Off"

        # ---------------- flows / volume slices ----------------
        df_flow = network_df[(network_df["season"] == selected_season) & (network_df["Time Period"] == selected_time)]
        if selected_market_type and selected_market_type != "All Markets":
            df_flow = df_flow[df_flow["mkt_type"] == selected_market_type]
        df_map = df_flow[df_flow["share"] > 0].copy()

        df_vol = market_volume_df[(market_volume_df["season"] == selected_season) & (market_volume_df["Time Period"] == selected_time)]
        if selected_market_type and selected_market_type != "All Markets":
            df_vol = df_vol[df_vol["mkt_type"] == selected_market_type]

        # ---------------- basemap + layers ----------------
        mapbox_style = MAPBOX_STYLE_DARK if "show_nightlights" in layer_toggles else MAPBOX_STYLE_LIGHT
        layers = []

        # Nightlights under everything
        if "show_nightlights" in layer_toggles and nightlights_data:
            try:
                img, coords = _get_nl_image_and_coords(nightlights_data, selected_time)
                if img:
                    coords = coords or _default_bbox_coords()
                    layers.append({
                        "sourcetype": "image",
                        "type": "raster",              # <- required for image in MapLibre
                        "source": img,        # your base64 data URI
                        "coordinates": coords,         # [[lonW,latN],[lonE,latN],[lonE,latS],[lonW,latS]]
                        "opacity": 0.70,
                        "below": "traces"              # keep it under your markers/lines
                    })
            except Exception as e:
                print(f"ERROR: nightlights layer: {e}")

        # Roads above nightlights
        if "show_roads" in layer_toggles and selected_time in roads_data:
            road_color = "rgba(255,255,255,0.45)" if "show_nightlights" in layer_toggles else "rgba(100,100,100,0.7)"
            road_width = 1.3 if "show_nightlights" in layer_toggles else 0.9
            layers.append({
                "sourcetype": "geojson",
                "source": roads_data[selected_time],
                "type": "line",
                "color": road_color,
                "line": {"width": road_width},
                "below": "traces"
            })

        # Preserve user pan/zoom
        zoom, center = 5.5, {"lat": 0.5, "lon": 37.5}
        if relayout_data and "map.center" in relayout_data:
            zoom = relayout_data.get("map.zoom", zoom)
            center = relayout_data.get("map.center", center)

        fig = go.Figure()

        # ---------------- Businesses overlay + metrics for hover ----------------
        # We'll also compute per-market business metrics to append to market hover text.
        biz_metrics_for_hover = None
        if show_businesses and (business_df is not None) and (not business_df.empty):
            df_biz = business_df.copy()
            if selected_market_type and selected_market_type != "All Markets":
                df_biz = df_biz[df_biz["mkt_type"] == selected_market_type]
            if biz_type and biz_type != "All":
                df_biz = df_biz[df_biz["business_label"] == biz_type]

            # Aggregate for overlay: total count at selected_time AND median km (valid distances only)
            g = (df_biz.groupby(["mkt_id", "mkt_name", "mkt_type", "lat", "lon"], observed=True)
                    .agg(
                            count=(selected_time, "sum"),
                            median_km=("nearest_km", lambda s: s.dropna().median())
                    )
                    .reset_index())
            g = g[g["count"] > 0]

            # This will be merged into the market markers for unified hover
            biz_metrics_for_hover = g[["mkt_id", "count", "median_km"]].rename(columns={"count": "biz_count", "median_km": "biz_median_km"})

            if not g.empty:
                # Bigger, clearer halos
                q95 = g["count"].quantile(0.95) if len(g) > 1 else g["count"].max()
                base = (q95 ** 0.5) if (pd.notna(q95) and q95 > 0) else 1.0
                scale = 32.0 / base  # exaggerated scale
                g["size"] = 8 + (g["count"].clip(lower=0) ** 0.5) * scale
                g["size"] = g["size"].clip(lower=8, upper=70)

                if biz_view_mode == "heatmap":
                    fig.add_trace(go.Densitymap(
                        lat=g["lat"], lon=g["lon"], z=g["count"],
                        radius=34, colorscale="Turbo",
                        colorbar=dict(title="Businesses")
                    ))
                    # Empty markers just to stabilize hover if you ever want it
                    fig.add_trace(go.Scattermap(
                        lat=g["lat"], lon=g["lon"], mode="markers",
                        marker=dict(size=10, color="rgba(0,0,0,0)"),
                        hoverinfo="skip", showlegend=False
                    ))
                else:
                    # Two white underlays → glow
                    outer = (g["size"] + 12).clip(upper=80)
                    inner = (g["size"] + 6).clip(upper=76)
                    for sz, alpha in [(outer, 0.35), (inner, 0.90)]:
                        fig.add_trace(go.Scattermap(
                            lat=g["lat"], lon=g["lon"], mode="markers",
                            marker=dict(size=sz, color=f"rgba(255,255,255,{alpha})", opacity=(biz_opacity/100.0)),
                            hoverinfo="skip", showlegend=False
                        ))

                    # Main colored fill — color by median distance if we have any, else by count
                    has_dist = g["median_km"].notna().any()
                    color_values = g["median_km"] if has_dist else g["count"]
                    cscale = "YlOrRd" if has_dist else "Blues"
                    cmax = float(color_values.quantile(0.95)) if len(g) > 1 else None
                    ctitle = "Median dist (km)" if has_dist else "Businesses"

                    fig.add_trace(go.Scattermap(
                        lat=g["lat"], lon=g["lon"], mode="markers",
                        marker=dict(
                            size=g["size"],
                            color=color_values,
                            colorscale=cscale, cmin=0, cmax=cmax,
                            opacity=(biz_opacity/100.0),
                            showscale=True, colorbar=dict(title=ctitle),
                        ),
                        name="Businesses",
                        hoverinfo="skip",  # keep a single tooltip (on markets)
                        showlegend=False
                    ))

        # ---------------- Build MARKET dots from a base union ----------------
        # (so fish-only markets still appear as blue dots)
        # From df_map (tomato flows)
        mkts_from_flows = pd.DataFrame(columns=["mkt_id","mkt_name","mkt_type","lat","lon"])
        if not df_map.empty:
            mkts_from_flows = (
                df_map[["mkt_id","mkt_name","mkt_type","market_lat","market_lon"]]
                .drop_duplicates()
                .rename(columns={"market_lat":"lat","market_lon":"lon"})
            )

        # From market_volume_df (has lat/lon)
        mkts_from_volume = df_vol[["mkt_id","mkt_name","mkt_type","lat","lon"]].drop_duplicates()

        # From businesses (covers fish-only or others without tomato flows)
        mkts_from_business = pd.DataFrame(columns=["mkt_id","mkt_name","mkt_type","lat","lon"])
        if (business_df is not None) and (not business_df.empty):
            mkts_from_business = business_df[["mkt_id","mkt_name","mkt_type","lat","lon"]].drop_duplicates()
            if selected_market_type and selected_market_type != "All Markets":
                mkts_from_business = mkts_from_business[mkts_from_business["mkt_type"] == selected_market_type]

        markets_base = pd.concat([mkts_from_flows, mkts_from_volume, mkts_from_business], ignore_index=True)
        markets_base = markets_base.dropna(subset=["lat","lon"]).drop_duplicates(subset=["mkt_id"])

        # ---------------- Trade routes + markers ----------------
        if "show_markers" in layer_toggles:
            opacity = opacity_percent / 100.0
            routes_visible = bool(toggle_value) and ("show" in toggle_value)

            # Routes (if flows exist)
            if not df_map.empty:
                if simplify_mode:
                    lats = [item for _, row in df_map.iterrows() for item in (row["origin_lat"], row["market_lat"], None)]
                    lons = [item for _, row in df_map.iterrows() for item in (row["origin_lon"], row["market_lon"], None)]
                    fig.add_trace(go.Scattermap(
                        lat=lats, lon=lons, mode="lines",
                        line=dict(width=3, color=f"rgba(0, 64, 133, {opacity})"),
                        name="Trade Routes", hoverinfo="none", visible=routes_visible))
                else:
                    share_bins = [
                        {"name": "High Share (>75%)", "data": df_map[df_map["share"] >= 75], "width": 4, "color": f"rgba(217, 95, 2, {opacity})"},
                        {"name": "Medium Share (25-75%)", "data": df_map[(df_map["share"] < 75) & (df_map["share"] >= 25)], "width": 2, "color": f"rgba(117, 112, 179, {opacity})"},
                        {"name": "Low Share (<25%)", "data": df_map[df_map["share"] < 25], "width": 1, "color": f"rgba(102, 166, 30, {opacity})"},
                    ]
                    for s_bin in share_bins:
                        if not s_bin["data"].empty:
                            lats = [item for _, row in s_bin["data"].iterrows() for item in (row["origin_lat"], row["market_lat"], None)]
                            lons = [item for _, row in s_bin["data"].iterrows() for item in (row["origin_lon"], row["market_lon"], None)]
                            fig.add_trace(go.Scattermap(
                                lat=lats, lon=lons, mode="lines",
                                line=dict(width=s_bin["width"], color=s_bin["color"]),
                                name=s_bin["name"], hoverinfo="none", visible=routes_visible))

            # ---- ORIGINS (always draw; fallback if no flows) ----
            origins_src = network_df[
                (network_df["season"] == selected_season) &
                (network_df["Time Period"] == selected_time)
            ]
            if selected_market_type and selected_market_type != "All Markets":
                origins_src = origins_src[origins_src["mkt_type"] == selected_market_type]

            origins_unique = (
                origins_src[["origin_name", "origin_lat", "origin_lon"]]
                .drop_duplicates()
                .dropna(subset=["origin_lat", "origin_lon"])
            )

            counts = (
                origins_src[origins_src["share"] > 0]
                .groupby("origin_name", observed=True)["mkt_name"]
                .nunique()
                .reset_index(name="market_count")
            )

            origins = origins_unique.merge(counts, on="origin_name", how="left").fillna({"market_count": 0})
            origins["size"] = 4 + origins["market_count"].astype(float)
            if simplify_mode:
                origins["size"] = 12
            origins["hover_text"] = (
                origins["origin_name"] + "<br>Supplies " +
                origins["market_count"].astype(int).astype(str) + " market(s)"
            )

            fig.add_trace(go.Scattermap(
                lat=origins["origin_lat"],
                lon=origins["origin_lon"],
                mode="markers",
                marker=dict(size=origins["size"], color="#a50f15", opacity=0.9),
                name="Produce Origin",
                text=origins["hover_text"],
                hoverinfo="text"
            ))
            # ---- END ORIGINS ----

            # Markets (build from union; then merge volumes, flows hover, and businesses hover)
            if not markets_base.empty:
                # Flow details per market (only where flows exist)
                market_hover_info = pd.DataFrame(columns=["mkt_name","details"])
                if not df_map.empty:
                    market_hover_info = (
                        df_map.assign(origin_share_str=df_map["origin_name"].astype(str) + ": " + df_map["share"].astype(int).astype(str) + "%")
                        .groupby("mkt_name", observed=True)["origin_share_str"].apply("<br>".join).reset_index(name="details")
                    )

                markets = (
                    markets_base
                    .merge(df_vol[["mkt_id","Total Volume"]], on="mkt_id", how="left")
                    .merge(market_hover_info, on="mkt_name", how="left")
                ).fillna({"Total Volume":0, "details":"n/a"})

                # Append businesses (if enabled) to the market hover
                if show_businesses and (biz_metrics_for_hover is not None):
                    markets = markets.merge(biz_metrics_for_hover, on="mkt_id", how="left")
                else:
                    markets["biz_count"] = pd.NA
                    markets["biz_median_km"] = pd.NA

                def _fmt_int(x): 
                    return "n/a" if pd.isna(x) else f"{int(round(float(x))):,}"
                def _fmt_km(x): 
                    return "NA" if pd.isna(x) else f"{float(x):.2f} km"

                base_text = (
                    "<b>" + markets["mkt_name"] + "</b><br><i>" + markets["mkt_type"].fillna("") + "</i><br>"
                    + "Trade Quantity: " + markets["Total Volume"].round(0).astype(int).apply(lambda x: f"{x:,}") + " units<br>"
                    + "--- Origins ---<br>" + markets["details"].fillna("n/a")
                )

                if show_businesses:
                    biz_block = (
                        "<br>--- Businesses ---<br>"
                        + biz_label_for_hover + f" ({selected_time}): "
                        + markets["biz_count"].apply(_fmt_int)
                        + "<br>Median distance: "
                        + markets["biz_median_km"].apply(_fmt_km)
                    )
                    markets["hover_text"] = base_text + biz_block
                else:
                    markets["hover_text"] = base_text

                if simplify_mode:
                    markets["size"] = 14
                else:
                    markets["size"] = 4 + (markets["Total Volume"].clip(lower=0) ** 0.5) * 0.08
                fig.add_trace(go.Scattermap(
                    lat=markets["lat"], lon=markets["lon"], mode="markers",
                    marker=dict(size=markets["size"], color="blue", opacity=0.9),
                    name="Market", text=markets["hover_text"], hoverinfo="text"))
            else:
                fig.add_annotation(text="No markets found for this selection.", showarrow=False,
                                font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"))

        # Invisible anchor (stabilizes mapbox render)
        fig.add_trace(go.Scattermap(
            lat=[0], lon=[37.5], mode="markers",
            marker=dict(size=0.01, color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="none"))

        fig.update_layout(
            margin=dict(r=0, l=0, b=0, t=0),
            uirevision="keep",
            showlegend=True,
            legend=dict(yanchor="top", y=0.92, xanchor="right", x=0.99,
                        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
                        traceorder="normal", itemsizing="constant", font=dict(size=11)),
            map=dict(style=mapbox_style, layers=layers, zoom=zoom, center=center),
            transition={"duration": 300},
        )

        map_title = f"{selected_season} - {selected_time}"
        return fig, map_title, "", simplify_label  # ping spinner


    # --------------------------- COMBINED MAP ----------------------------------
    @app.callback(
        [
            Output("combined-map", "figure"),
            Output("combined-map-title", "children"),
            Output("combined-loading-sentinel", "children"),
        ],
        [
            Input("master-market-type-filter", "value"),
            Input("data-type-toggle", "value"),
            Input("view-type-toggle", "value"),
            Input("combined-trader-type-dropdown", "value"),
            Input("combined-season-toggle", "value"),
            Input("combined-time-slider", "value"),
        ],
        [State("combined-map", "relayoutData")],
    )
    def update_combined_map(market_type, data_type, view_type, selected_trader, selected_season, time_value, relayout_data):
        time_map = {0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"}
        selected_time = time_map.get(time_value, "Now")

        fig = go.Figure()

        if data_type == "tomatoes":
            df = market_volume_df.copy()
            if market_type and market_type != "All Markets":
                df = df[df["mkt_type"] == market_type]
            df = df[(df["season"] == selected_season) & (df["Time Period"] == selected_time) & (df["Total Volume"] > 0)]
            title_parts = ["Tomato Trade Volume", selected_season, selected_time]
            z_value_col = "Total Volume"
            colorscale = "Viridis"
            colorbar_title = "Trade Volume"
        else:
            df = trader_df.copy()
            if market_type and market_type != "All Markets":
                df = df[df["mkt_type"] == market_type]
            if selected_trader and selected_trader != "All":
                df = df[df["trader_id"] == selected_trader]
            df = df.groupby(["mkt_name", "lat", "lon"], observed=True)[selected_time].sum().reset_index()
            df = df[df[selected_time] > 0]
            title_parts = [f"{selected_trader} Traders" if selected_trader and selected_trader != "All" else "All Traders", selected_time]
            z_value_col = selected_time
            colorscale = "Plasma"
            colorbar_title = "No. of Traders"

        map_title = " - ".join(title_parts)

        if df.empty:
            fig.add_annotation(text="No data available for this selection.", showarrow=False)
        else:
            df["hover_text"] = "<b>" + df["mkt_name"] + "</b><br>" + colorbar_title + ": " + df[z_value_col].round(0).astype(int).apply(lambda x: f"{x:,}")
            if view_type == "points":
                df["size"] = 5 + (df[z_value_col].clip(lower=0) ** 0.5) * (0.1 if data_type == "tomatoes" else 0.8)
                fig.add_trace(
                    go.Scattermap(
                        lat=df["lat"], lon=df["lon"], mode="markers",
                        marker=dict(
                            size=df["size"], color=df[z_value_col], colorscale=colorscale,
                            cmin=0, cmax=float(df[z_value_col].quantile(0.95)) if len(df) > 1 else None,
                            showscale=True, colorbar=dict(title=colorbar_title),
                        ),
                        text=df["hover_text"], hoverinfo="text",
                    )
                )
            else:
                heatmap_radius = 30 if data_type == "traders" else 20
                fig.add_trace(go.Densitymap(lat=df["lat"], lon=df["lon"], z=df[z_value_col],
                                               radius=heatmap_radius, colorscale=colorscale,
                                               colorbar=dict(title=colorbar_title)))
                fig.add_trace(go.Scattermap(lat=df["lat"], lon=df["lon"], mode="markers",
                                               marker=dict(size=10, color="rgba(0,0,0,0)"),
                                               text=df["hover_text"], hoverinfo="text", showlegend=False))

        zoom, center = 5.5, {"lat": 0.5, "lon": 37.5}
        if relayout_data and "map.center" in relayout_data:
            zoom = relayout_data.get("map.zoom", zoom)
            center = relayout_data.get("map.center", center)

        fig.update_layout(
            margin=dict(r=0, l=0, b=0, t=0),
            uirevision="keep",
            map=dict(style=MAPBOX_STYLE_LIGHT, zoom=zoom, center=center),
            transition={"duration": 300},
        )
        return fig, map_title, ""  # ping spinner


# ------------------------------------------------------------------------------
# 6) RUN
# ------------------------------------------------------------------------------
# app.py (at bottom)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    debug_flag = os.environ.get("DASH_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_flag)







