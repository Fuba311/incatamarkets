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
# Expose `server` for Gunicorn/Render. Works locally with `python app.py`.
app = dash.Dash(__name__, assets_folder='assets', title="INCATA Market Analysis Dashboard")
server = app.server

# Minimal, embedded CSS (keeps things pretty even without an /assets folder)
# You can still add/override with assets/style.css later if you want.
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

      /* Section cards */
      .section-card{
        background:var(--card); border:1px solid var(--border); border-radius:12px;
        padding:24px; box-shadow:var(--shadow); margin-bottom:38px;
      }

      /* Floating control panel */
      .floating-controls{
        position:absolute; top:64px; left:10px; width:280px; background:var(--panel-bg);
        border:1px solid #e9ecef; border-radius:10px; box-shadow:var(--shadow);
        overflow:visible !important;              /* KEY: allow menus to escape */
        z-index:1000; transition:width .2s ease;
      }
      .floating-controls.icon-only{
        width:46px; overflow:hidden !important;  /* collapse keeps tidy */
      }
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
      .controls-content{
        padding:12px; max-height:70vh;
        overflow:visible !important;             /* KEY: prevent clipping */
      }
      .controls-content.hidden{display:none;}

      /* Title bubble and info box */
      #network-map-title{
        position:absolute; top:10px; left:50%; transform:translateX(-50%);
        background:rgba(255,255,255,.95); padding:8px 20px; border-radius:20px;
        font-size:14px; font-weight:700; color:var(--brand);
        box-shadow:0 2px 8px rgba(0,0,0,.12); border:1px solid rgba(0,64,133,.2); z-index:999;
      }
      #network-info-collapse{ backdrop-filter:saturate(140%) blur(2px); }
      @keyframes fadeIn{ from{opacity:0} to{opacity:1} }

      /* Dropdown overflow fix + stacking above map */
      .lifted-dropdown, .dash-dropdown, .Select{ z-index:3000 !important; }
      .lifted-dropdown .Select-menu-outer,
      .dash-dropdown .Select-menu-outer,
      .Select-menu-outer{
        z-index:5000 !important;                 /* above Plotly/Mapbox canvas */
        overflow:visible !important;
      }
      /* react-select v2+ fallbacks (if present) */
      .lifted-dropdown .Select-menu, .Select-menu{ z-index:5000 !important; }
      .lifted-dropdown .Select__menu{ z-index:5000 !important; }

      /* Responsive */
      @media (max-width:768px){
        .floating-controls{ top:56px; left:6px; width:88vw; }
        .floating-controls.icon-only{ width:44px; }
        .controls-content{ overflow-y:auto !important; } /* allow scroll on phones */
      }

      /* Hero */
      .app-hero h1{ text-align:center; color:var(--brand); margin-bottom:4px; }
      .app-hero h4{ text-align:center; font-weight:normal; margin-top:2px; }
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
# 2) DATA LOADING (optimized, no GeoPandas dependency)
# ------------------------------------------------------------------------------
print("--- Loading Pre-Processed Data ---")
PROCESSED_DATA_FOLDER = Path(__file__).parent / "processed_data"

network_df = None
market_volume_df = None
trader_df = None
roads_data = {}
nightlights_data = {}
data_load_success = False

try:
    # Core tabular data
    network_df = pd.read_parquet(PROCESSED_DATA_FOLDER / "network_df.parquet")
    market_volume_df = pd.read_parquet(PROCESSED_DATA_FOLDER / "market_volume_df.parquet")
    trader_df = pd.read_parquet(PROCESSED_DATA_FOLDER / "trader_df.parquet")
    print("SUCCESS: Parquet tables loaded.")

    # Preprocessed geospatial overlays (read raw GeoJSON, avoid GeoPandas)
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

    # Nightlights (expects: { 'now' | '5_yrs_ago' | '10_yrs_ago': [<b64_img>, [[lonW,latN],[lonE,latN],[lonE,latS],[lonW,latS]]] })
    nl_path = PROCESSED_DATA_FOLDER / "nightlights_data.json"
    if nl_path.exists():
        with open(nl_path, "r", encoding="utf-8") as f:
            nightlights_data = json.load(f)
        print("SUCCESS: Nightlights overlay loaded.")
    else:
        print(f"Warning: Nightlights JSON not found at {nl_path}")

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

# Basemap styles – token-free
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
    app.layout = html.Div(
        style={"padding": "2% 5%"},
        children=[
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

            # Global filter
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
                    # Global filter
                    dcc.Dropdown(
                        id='master-market-type-filter',
                        className='lifted-dropdown',
                        options=[{'label': 'All Markets', 'value': 'All Markets'}] + [
                            {'label': m, 'value': m} for m in sorted(network_df['mkt_type'].dropna().unique())
                        ],
                        value='All Markets'
                    ),
                ],
            ),

            # --- NETWORK MAP SECTION ---
            html.Div(
                className="section-card",
                children=[
                    html.H2("Produce Flow Network", style={"color": "#004085", "border-bottom": "2px solid #b8daff", "padding-bottom": "10px"}),
                    html.P(
                        "Map shows tomato flows from origins to markets. Origins (red) are approximate locations (often county-level centroids).",
                        style={"marginBottom": "16px"},
                    ),
                    html.Div(
                        style={"position": "relative", "width": "100%"},
                        children=[
                            # Floating panel
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
                                            html.Div(
                                                style={"marginBottom": "16px"},
                                                children=[
                                                    html.Label("📅 Time Period", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Slider(
                                                        id="network-time-slider",
                                                        min=0, max=2, step=None, included=False, value=2,
                                                        marks={0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"},
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "16px"},
                                                children=[
                                                    html.Label("🌱 Season", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.RadioItems(
                                                        id="season-toggle",
                                                        options=[
                                                            {"label": " High Season", "value": "High Season"},
                                                            {"label": " Low Season", "value": "Low Season"},
                                                        ],
                                                        value="High Season",
                                                        labelStyle={"display": "block", "marginBottom": "5px", "fontSize": "12px"},
                                                    ),
                                                ],
                                            ),
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
                                            html.Div(
                                                style={"marginBottom": "12px"},
                                                children=[
                                                    html.Label("🔗 Trade Routes", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Checklist(
                                                        id="toggle-routes",
                                                        options=[{"label": " Show Trade Routes", "value": "show"}],
                                                        value=["show"],
                                                        labelStyle={"fontSize": "12px", "marginBottom": "8px"},
                                                    ),
                                                    html.Div(
                                                        style={"display": "flex", "alignItems": "center", "gap": "10px", "marginTop": "10px"},
                                                        children=[
                                                            html.Label("Opacity:", style={"fontSize": "12px", "minWidth": "50px"}),
                                                            dcc.Dropdown(
                                                                id='opacity-dropdown',
                                                                className='lifted-dropdown',
                                                                options=[{'label': f'{i}%', 'value': i} for i in range(0, 101, 10)],
                                                                value=70, clearable=False, searchable=False,
                                                                style={'width': '80px', 'fontSize': '11px'}
                                                            ),

                                                        ],
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"borderTop": "1px solid #e0e0e0", "paddingTop": "10px", "marginTop": "10px"},
                                                children=[
                                                    html.Button(
                                                        "ℹ How to Read This Map",
                                                        id="network-info-button",
                                                        n_clicks=0,
                                                        style={"width": "100%", "cursor": "pointer", "border": "1px solid #004085", "backgroundColor": "#e7f3ff", "padding": "6px 10px", "borderRadius": "6px", "fontSize": "12px"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id="network-map-title"),
                            dcc.Loading(
                                type="dot",
                                children=dcc.Graph(
                                    id="network-map",
                                    style={"height": "85vh", "width": "100%"},
                                    config=GRAPH_CONFIG,
                                ),
                            ),
                            html.Div(
                                id="network-info-collapse",
                                style={
                                    "display": "none", "position": "absolute", "bottom": "10px", "right": "10px",
                                    "background-color": "rgba(248, 249, 250, 0.95)", "padding": "15px",
                                    "border": "1px dashed #cce5ff", "borderRadius": "6px", "zIndex": "998", "maxWidth": "420px"
                                },
                                children=[
                                    html.Button("✕", id="close-info-btn", n_clicks=0, style={"position": "absolute", "top": "6px", "right": "10px", "background": "transparent", "border": "none", "fontSize": "18px", "cursor": "pointer"}),
                                    dcc.Markdown(
                                        "* **Red Dots (Produce Origins):** County/area where tomatoes are sourced.\n"
                                        "* **Blue Dots (Markets):** Markets where tomatoes are sold.\n"
                                        "* **Lines (Trade Routes):** Connections from origin to market.\n"
                                        "* **Line Thickness:** Represents the share of produce from that origin.",
                                        style={"fontSize": "12px", "margin": "0"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # --- MARKET CONCENTRATION SECTION ---
            html.Div(
                className="section-card",
                children=[
                    html.H2("Market Concentration Analysis", style={"color": "#004085", "border-bottom": "2px solid #b8daff", "padding-bottom": "10px"}),
                    html.P("Analyze tomato trade volume or trader concentration. Switch type and view to explore patterns."),
                    html.H3(id="combined-map-title", style=title_style),
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
                                                        value="tomatoes",
                                                        inline=True,
                                                        labelStyle={"marginRight": "14px"},
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
                                                        value="points",
                                                        inline=True,
                                                        labelStyle={"marginRight": "14px"},
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="season-control-div",
                                                className="conditional-control",
                                                style={"marginBottom": "18px"},
                                                children=[
                                                    html.Label("🌱 Season", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.RadioItems(
                                                        id="combined-season-toggle",
                                                        options=[{"label": " High", "value": "High Season"}, {"label": " Low", "value": "Low Season"}],
                                                        value="High Season",
                                                        inline=True,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="trader-control-div",
                                                className="conditional-control",
                                                style={"marginBottom": "18px"},
                                                children=[
                                                    html.Label("👤 Trader Type", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"}),
                                                    dcc.Dropdown(
                                                        id='combined-trader-type-dropdown',
                                                        className='lifted-dropdown',
                                                        options=[{'label': 'All Traders', 'value': 'All'}] + [
                                                            {'label': t, 'value': t} for t in sorted(trader_df['trader_id'].dropna().unique())
                                                        ],
                                                        value='All', placeholder="Select..."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "6px"},
                                                children=[
                                                    html.Label("📅 Time Period", style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"}),
                                                    dcc.Slider(
                                                        id="combined-time-slider",
                                                        min=0, max=2, step=None, included=False, value=2,
                                                        marks={0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Loading(
                                type="dot",
                                children=dcc.Graph(id="combined-map", style={"height": "85vh"}, config=GRAPH_CONFIG),
                            ),
                        ],
                    ),
                ],
            ),

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
else:
    # Error layout if data fails to load
    app.layout = html.Div(
        [
            html.H1("INCATA Dashboard - Application Error", style={"textAlign": "center", "color": "#dc3545", "marginTop": "50px"}),
            html.P("Could not load necessary data files.", style={"textAlign": "center", "fontSize": "18px"}),
            html.P("Please ensure the 'processed_data' folder is deployed with the following files:", style={"textAlign": "center"}),
            html.Ul(
                [
                    html.Li("network_df.parquet"),
                    html.Li("market_volume_df.parquet"),
                    html.Li("trader_df.parquet"),
                    html.Li("roads_*_processed.geojson"),
                    html.Li("nightlights_data.json"),
                ],
                style={"maxWidth": "440px", "margin": "0 auto"},
            ),
            html.P("Check application logs for details.", style={"textAlign": "center", "marginTop": "16px"}),
        ],
        style={"padding": "2% 5%"},
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

    @app.callback(
        [Output("network-map", "figure"), Output("network-map-title", "children")],
        [
            Input("master-market-type-filter", "value"),
            Input("season-toggle", "value"),
            Input("network-time-slider", "value"),
            Input("opacity-dropdown", "value"),
            Input("toggle-routes", "value"),
            Input("layer-toggles", "value"),
        ],
        [State("network-map", "relayoutData")],
    )
    def update_network_map(selected_market_type, selected_season, time_value, opacity_percent, toggle_value, layer_toggles, relayout_data):
        time_map = {0: "10 Yrs Ago", 1: "5 Yrs Ago", 2: "Now"}
        selected_time = time_map.get(time_value, "Now")
        layer_toggles = layer_toggles or []
        if opacity_percent is None:
            opacity_percent = 70

        # Filter data
        df_flow = network_df[(network_df["season"] == selected_season) & (network_df["Time Period"] == selected_time)]
        if selected_market_type and selected_market_type != "All Markets":
            df_flow = df_flow[df_flow["mkt_type"] == selected_market_type]

        df_map = df_flow[df_flow["share"] > 0].copy()
        df_vol = market_volume_df[(market_volume_df["season"] == selected_season) & (market_volume_df["Time Period"] == selected_time)]

        # Basemap style & overlays
        mapbox_style = MAPBOX_STYLE_DARK if "show_nightlights" in layer_toggles else MAPBOX_STYLE_LIGHT
        layers = []

        # Nightlights image overlay
        key = {"10 Yrs Ago": "10_yrs_ago", "5 Yrs Ago": "5_yrs_ago", "Now": "now"}[selected_time]
        if "show_nightlights" in layer_toggles and key in nightlights_data:
            try:
                b64_img, coords = nightlights_data[key]
                layers.append({"sourcetype": "image", "source": b64_img, "coordinates": coords, "opacity": 0.8, "below": ""})
            except Exception as _:
                pass

        # Roads GeoJSON overlay
        if "show_roads" in layer_toggles and selected_time in roads_data:
            road_color = "rgba(211, 211, 211, 0.6)" if "show_nightlights" in layer_toggles else "rgba(100, 100, 100, 0.7)"
            layers.append({
                "sourcetype": "geojson",
                "source": roads_data[selected_time],
                "type": "line",
                "color": road_color,
                "line": {"width": 0.8},
                "below": "traces",
            })

        # Honor current view, else sensible default
        zoom, center = 5.5, {"lat": 0.5, "lon": 37.5}
        if relayout_data and "mapbox.center" in relayout_data:
            zoom = relayout_data.get("mapbox.zoom", zoom)
            center = relayout_data.get("mapbox.center", center)

        fig = go.Figure()

        if "show_markers" in layer_toggles:
            opacity = opacity_percent / 100.0
            routes_visible = _safe_bool_contains(toggle_value, "show")

            share_bins = [
                {"name": "High Share (>75%)",   "data": df_map[df_map["share"] >= 75],                          "width": 4, "color": f"rgba(217, 95, 2, {opacity})"},
                {"name": "Medium Share (25-75%)","data": df_map[(df_map["share"] < 75) & (df_map["share"] >= 25)], "width": 2, "color": f"rgba(117, 112, 179, {opacity})"},
                {"name": "Low Share (<25%)",    "data": df_map[df_map["share"] < 25],                           "width": 1, "color": f"rgba(102, 166, 30, {opacity})"},
            ]

            # Trade routes
            for s_bin in share_bins:
                if not s_bin["data"].empty:
                    lats = [item for _, row in s_bin["data"].iterrows() for item in (row["origin_lat"], row["market_lat"], None)]
                    lons = [item for _, row in s_bin["data"].iterrows() for item in (row["origin_lon"], row["market_lon"], None)]
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=lats, lon=lons, mode="lines",
                            line=dict(width=s_bin["width"], color=s_bin["color"]),
                            name=s_bin["name"], hoverinfo="none", visible=routes_visible,
                        )
                    )

            # Origins & markets
            if not df_map.empty:
                # Origins grouped by number of markets they supply
                origins = (
                    df_map[["origin_name", "origin_lat", "origin_lon"]].drop_duplicates()
                    .merge(
                        df_map.groupby("origin_name", observed=True)["mkt_name"].nunique().reset_index(name="market_count"),
                        on="origin_name",
                    )
                )
                origins["hover_text"] = origins["origin_name"] + "<br>Supplies " + origins["market_count"].astype(int).astype(str) + " market(s)"
                fig.add_trace(
                    go.Scattermapbox(
                        lat=origins["origin_lat"],
                        lon=origins["origin_lon"],
                        mode="markers",
                        marker=dict(size=(5 + origins["market_count"]), color="#a50f15", opacity=0.9),
                        name="Produce Origin",
                        text=origins["hover_text"],
                        hoverinfo="text",
                    )
                )

                # Markets sized by volume, hover shows origin breakdown
                markets = (
                    df_map[["mkt_id", "mkt_name", "market_lat", "market_lon", "mkt_type"]].drop_duplicates()
                    .merge(df_vol[["mkt_id", "Total Volume"]], on="mkt_id", how="left")
                    .fillna(0)
                )
                market_hover_info = (
                    df_map.assign(origin_share_str=df_map["origin_name"].astype(str) + ": " + df_map["share"].astype(int).astype(str) + "%")
                    .groupby("mkt_name", observed=True)["origin_share_str"].apply("<br>".join)
                    .reset_index(name="details")
                )
                markets = markets.merge(market_hover_info, on="mkt_name", how="left")
                markets["hover_text"] = (
                    "<b>" + markets["mkt_name"] + "</b><br><i>" + markets["mkt_type"].fillna("") + "</i><br>"
                    + "Trade Quantity: " + markets["Total Volume"].round(0).astype(int).apply(lambda x: f"{x:,}") + " units<br>"
                    + "--- Origins ---<br>" + markets["details"].fillna("n/a")
                )
                markets["size"] = 4 + (markets["Total Volume"].clip(lower=0) ** 0.5) * 0.08
                fig.add_trace(
                    go.Scattermapbox(
                        lat=markets["market_lat"],
                        lon=markets["market_lon"],
                        mode="markers",
                        marker=dict(size=markets["size"], color="blue", opacity=0.9),
                        name="Market",
                        text=markets["hover_text"],
                        hoverinfo="text",
                    )
                )
            else:
                fig.add_annotation(
                    text="No trade flow data for this selection.",
                    showarrow=False,
                    font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"),
                )

        # Invisible anchor for initial viewport stability
        fig.add_trace(
            go.Scattermapbox(
                lat=[0], lon=[37.5], mode="markers",
                marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="none",
            )
        )

        fig.update_layout(
            margin=dict(r=0, l=0, b=0, t=0),
            uirevision="keep",  # preserve zoom/position across updates
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.92, xanchor="right", x=0.99,
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
                traceorder="normal", itemsizing="constant", font=dict(size=11),
            ),
            mapbox=dict(style=mapbox_style, layers=layers, zoom=zoom, center=center),
        )
        map_title = f"{selected_season} - {selected_time}"
        return fig, map_title

    @app.callback(
        [Output("trader-control-div", "className"), Output("season-control-div", "className")],
        Input("data-type-toggle", "value"),
    )
    def update_combined_map_controls(data_type):
        if data_type == "traders":
            return "conditional-control", "conditional-control hidden"
        else:
            return "conditional-control hidden", "conditional-control"

    @app.callback(
        [Output("combined-map", "figure"), Output("combined-map-title", "children")],
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
                if data_type == "tomatoes":
                    df["size"] = 5 + (df[z_value_col].clip(lower=0) ** 0.5) * 0.1
                else:
                    df["size"] = 5 + (df[z_value_col].clip(lower=0) ** 0.5) * 0.8

                fig.add_trace(
                    go.Scattermapbox(
                        lat=df["lat"], lon=df["lon"], mode="markers",
                        marker=dict(
                            size=df["size"],
                            color=df[z_value_col],
                            colorscale=colorscale,
                            cmin=0,
                            cmax=float(df[z_value_col].quantile(0.95)) if len(df) > 1 else None,
                            showscale=True,
                            colorbar=dict(title=colorbar_title),
                        ),
                        text=df["hover_text"],
                        hoverinfo="text",
                    )
                )
            else:
                heatmap_radius = 30 if data_type == "traders" else 20

                fig.add_trace(
                    go.Densitymapbox(
                        lat=df["lat"], lon=df["lon"], z=df[z_value_col],
                        radius=heatmap_radius, colorscale=colorscale,
                        colorbar=dict(title=colorbar_title),
                    )
                )
                # Invisible points to preserve hover info on heatmap
                fig.add_trace(
                    go.Scattermapbox(
                        lat=df["lat"], lon=df["lon"], mode="markers",
                        marker=dict(size=10, color="rgba(0,0,0,0)"),
                        text=df["hover_text"], hoverinfo="text", showlegend=False,
                    )
                )

        zoom, center = 5.5, {"lat": 0.5, "lon": 37.5}
        if relayout_data and "mapbox.center" in relayout_data:
            zoom = relayout_data.get("mapbox.zoom", zoom)
            center = relayout_data.get("mapbox.center", center)

        fig.update_layout(
            margin=dict(r=0, l=0, b=0, t=0),
            uirevision="keep",
            mapbox=dict(style=MAPBOX_STYLE_LIGHT, zoom=zoom, center=center),
        )
        return fig, map_title

# ------------------------------------------------------------------------------
# 6) RUN (local) / SERVE (Render)
# ------------------------------------------------------------------------------
# Render uses Gunicorn and `server` above. Running locally: python app.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    debug_flag = os.environ.get("DASH_DEBUG", "1") == "1"  # set DASH_DEBUG=0 on Render if you run this directly
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
