# app.py
# --- Lightweight production-ready Dash app (local + Render) ---
print("--- STARTING INCATA MARKET ANALYSIS DASHBOARD ---")

import os
import itertools
from pathlib import Path
import json
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
import dash
from dash import dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State, MATCH
from plotly.colors import qualitative as qual_colors

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
      .floating-controls.network-controls{ width:360px; }
      .floating-controls.network-controls.icon-only{ width:56px; }
      .floating-controls.region-controls{ width:220px; }
      .floating-controls.region-controls.icon-only{ width:56px; }
      .floating-panel-container{
        position:absolute; top:16px; left:8px; display:flex; gap:12px; z-index:1000;
        align-items:flex-start; pointer-events:none;
      }
      .floating-panel-container .floating-controls{
        position:relative; top:auto; left:auto; pointer-events:auto;
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
      .controls-content{ padding:14px 12px 26px; max-height:70vh; overflow:visible !important; }
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
        .floating-panel-container{
          top:56px; left:6px; flex-direction:column; gap:8px;
        }
        .floating-panel-container .floating-controls{
          width:88vw;
        }
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
      .map-floating-legend{
        position:absolute; right:18px; bottom:18px; background:rgba(255,255,255,0.92);
        border:1px solid rgba(0,64,133,0.2); box-shadow:0 4px 12px rgba(0,0,0,0.08);
        border-radius:10px; padding:10px 14px; font-size:11px; color:#495057;
        display:none; pointer-events:none;
      }
      .map-floating-legend.visible{ display:block; }
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
_candidate_processed_roots = [
    Path(__file__).parent / "processed_data",
    Path(r"C:\Users\andre\OneDrive\Desktop\BMGF\Tegemeo map\data\processed_data"),
]
PROCESSED_DATA_FOLDER = next((p for p in _candidate_processed_roots if p.exists()), _candidate_processed_roots[0])
print(f"Using processed data folder: {PROCESSED_DATA_FOLDER}")


TIME_PERIOD_SUFFIX = {
    "10 Yrs Ago": "10_yrs_ago",
    "5 Yrs Ago": "5_yrs_ago",
    "Now": "now",
}

REGION_SPECS = {
    "kenya": {
        "label": "Kenya",
        "data_dir": PROCESSED_DATA_FOLDER,
        "network_file": "network_df.parquet",
        "market_volume_file": "market_volume_df.parquet",
        "trader_file": "trader_df.parquet",
        "business_file": "business_df.parquet",
        "nightlights_candidates": [
            "nightlights_data_.json",
            "nightlights_data.json",
            "nightlights.json",
        ],
        "roads_pattern": "roads_{suffix}_processed.geojson",
        "time_periods": ["10 Yrs Ago", "5 Yrs Ago", "Now"],
        "map_defaults": {"center": {"lat": 0.5, "lon": 37.5}, "zoom": 5.5},
        "bbox_fallback": [[33.9, 5.2], [41.9, 5.2], [41.9, -4.9], [33.9, -4.9]],
    },
    "odisha": {
        "label": "Odisha (India)",
        "data_dir": PROCESSED_DATA_FOLDER,
        "network_file": "network_df_odisha.parquet",
        "market_volume_file": "market_volume_df_odisha.parquet",
        "trader_file": "trader_df_odisha.parquet",
        "business_file": None,
        "nightlights_candidates": [
            "nightlights_data_odisha_.json",
            "nightlights_data_odisha.json",
        ],
        "roads_pattern": "roads_{suffix}_odisha_processed.geojson",
        "time_periods": ["10 Yrs Ago", "Now"],
        "map_defaults": {"center": {"lat": 20.3, "lon": 85.8}, "zoom": 6.2},
        "bbox_fallback": [[81.4, 22.7], [87.6, 22.7], [87.6, 18.8], [81.4, 18.8]],
        "odisha_outside_offset": 1.9,
        "odisha_within_offset": 0.7,
    },
}

_region_cache = {}
_region_errors = {}

def _canonical_region(region_key: str) -> str:
    return str(region_key or "").strip().lower()

def _clean_unique(series):
    if series is None:
        return []
    try:
        values = [v for v in series.dropna().unique() if str(v).strip() and str(v).lower() != "nan"]
    except Exception:
        values = []
    return values

def _augment_region_scaling(region_data: dict) -> dict:
    market_volume_df = region_data.get("market_volume_df")
    trader_df = region_data.get("trader_df")
    business_df = region_data.get("business_df")
    network_df = region_data.get("network_df")
    time_periods = list(region_data.get("time_periods") or [])

    # ---- Market volumes ----
    volume_max_global = 0.0
    volume_max_by_type = {}
    if market_volume_df is not None and "Total Volume" in market_volume_df.columns:
        all_volumes = pd.to_numeric(market_volume_df["Total Volume"], errors="coerce")
        if not all_volumes.empty:
            volume_max_global = float(all_volumes.max(skipna=True) or 0.0)
        if "mkt_type" in market_volume_df.columns:
            for mkt_value, group in market_volume_df.groupby("mkt_type", observed=True):
                if pd.isna(mkt_value):
                    continue
                group_volumes = pd.to_numeric(group["Total Volume"], errors="coerce")
                if not group_volumes.empty:
                    vmax = group_volumes.max(skipna=True)
                    if pd.notna(vmax):
                        volume_max_by_type[mkt_value] = float(vmax)
    region_data["volume_max_global"] = volume_max_global
    region_data["volume_max_by_type"] = volume_max_by_type

    # ---- Trader maxima ----
    trader_time_cols = [col for col in time_periods if trader_df is not None and col in trader_df.columns]
    trader_max_global = 0.0
    trader_max_by_market_type = {}
    trader_max_by_trader = {}
    trader_max_by_market_and_trader = {}

    def _max_trader_value(df_subset):
        if df_subset is None or df_subset.empty or not trader_time_cols:
            return 0.0
        group_cols = [col for col in ["mkt_name", "lat", "lon"] if col in df_subset.columns]
        if not group_cols:
            return 0.0
        grouped = df_subset.groupby(group_cols, observed=True)[trader_time_cols].sum()
        if grouped.empty:
            return 0.0
        max_val = pd.to_numeric(grouped[trader_time_cols].stack(), errors="coerce").max(skipna=True)
        return float(max_val) if pd.notna(max_val) else 0.0

    if trader_df is not None:
        trader_max_global = _max_trader_value(trader_df)
        if "mkt_type" in trader_df.columns:
            for mkt_value, subset in trader_df.groupby("mkt_type", observed=True):
                if pd.isna(mkt_value):
                    continue
                trader_max_by_market_type[mkt_value] = _max_trader_value(subset)
        if "trader_id" in trader_df.columns:
            for trader_id, subset in trader_df.groupby("trader_id", observed=True):
                if pd.isna(trader_id):
                    continue
                trader_max_by_trader[trader_id] = _max_trader_value(subset)
                if "mkt_type" in subset.columns:
                    for mkt_value, combo_subset in subset.groupby("mkt_type", observed=True):
                        if pd.isna(mkt_value):
                            continue
                        trader_max_by_market_and_trader[(mkt_value, trader_id)] = _max_trader_value(combo_subset)

    region_data["trader_time_columns"] = trader_time_cols
    region_data["trader_max_global"] = trader_max_global
    region_data["trader_max_by_market_type"] = trader_max_by_market_type
    region_data["trader_max_by_trader"] = trader_max_by_trader
    region_data["trader_max_by_market_and_trader"] = trader_max_by_market_and_trader

    # ---- Business maxima ----
    business_max_by_time = {}
    business_max_global = 0.0
    if business_df is not None and not business_df.empty:
        biz_group_cols = [col for col in ["mkt_id", "mkt_name", "mkt_type", "lat", "lon"] if col in business_df.columns]
        if biz_group_cols:
            candidate_time_cols = [col for col in time_periods if col in business_df.columns]
            for time_label in candidate_time_cols:
                totals = business_df.groupby(biz_group_cols, observed=True)[time_label].sum(min_count=1)
                max_val = pd.to_numeric(totals, errors="coerce").max(skipna=True)
                if pd.notna(max_val):
                    business_max_by_time[time_label] = float(max_val)
                    business_max_global = max(business_max_global, float(max_val))
    region_data["business_max_by_time"] = business_max_by_time
    region_data["business_max_global"] = business_max_global

    # ---- Origin supply maxima ----
    origin_supply_max = 0.0
    if network_df is not None and not network_df.empty:
        supply_col = next(
            (col for col in ["Trade Quantity", "trade_quantity", "Volume", "volume", "quantity"] if col in network_df.columns),
            None,
        )
        if supply_col:
            supply_series = pd.to_numeric(network_df[supply_col], errors="coerce").fillna(0.0)
        else:
            supply_series = pd.to_numeric(network_df.get("share", 0), errors="coerce").fillna(0.0)
        totals = (
            network_df.assign(_supply_metric=supply_series)
            .groupby("origin_name", observed=True)["_supply_metric"]
            .sum(min_count=1)
        )
        max_val = pd.to_numeric(totals, errors="coerce").max(skipna=True)
        if pd.notna(max_val):
            origin_supply_max = float(max_val)
    region_data["origin_supply_max"] = origin_supply_max

    return region_data

def load_region_data(region_key: str):
    region_key = _canonical_region(region_key)
    if not region_key:
        raise ValueError("Region key is required")
    if region_key in _region_cache:
        return _region_cache[region_key]
    if region_key not in REGION_SPECS:
        raise ValueError(f"Unknown region '{region_key}'. Available: {', '.join(REGION_SPECS)}")

    spec = REGION_SPECS[region_key]
    data_dir = spec["data_dir"]
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data folder for {spec['label']} not found at {data_dir}")

    print(f"Loading processed data for {spec['label']} from {data_dir}")

    def _read_parquet(filename: str) -> pd.DataFrame:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_parquet(path)

    network_df = _read_parquet(spec["network_file"])
    market_volume_df = _read_parquet(spec["market_volume_file"])
    trader_df = _read_parquet(spec["trader_file"])

    business_df = None
    if spec.get("business_file"):
        business_path = data_dir / spec["business_file"]
        if business_path.exists():
            business_df = pd.read_parquet(business_path)
            if "business_label" in business_df.columns:
                labels = sorted([x for x in business_df["business_label"].dropna().unique()])
                print(f"SUCCESS: Business table loaded. Types: {labels}")
            else:
                print("SUCCESS: Business table loaded (no label column detected).")
        else:
            print(f"Info: Business parquet not found for {spec['label']} (expected {business_path.name})")
    else:
        print(f"Info: No business dataset configured for {spec['label']}.")

    for df in (network_df, market_volume_df):
        if "Time Period" in df.columns:
            df["Time Period"] = pd.Categorical(df["Time Period"], categories=spec["time_periods"], ordered=True)

    market_types = sorted({
        val for series in [network_df.get("mkt_type"), market_volume_df.get("mkt_type")] for val in _clean_unique(series)
    })

    trader_types = sorted(_clean_unique(trader_df.get("trader_id"))) if "trader_id" in trader_df.columns else []

    roads_data = {}
    for pretty in spec["time_periods"]:
        suffix = TIME_PERIOD_SUFFIX.get(pretty)
        if not suffix:
            continue
        road_path = data_dir / spec["roads_pattern"].format(suffix=suffix)
        if road_path.exists():
            with open(road_path, "r", encoding="utf-8") as f:
                roads_data[pretty] = json.load(f)
            print(f"SUCCESS: Roads overlay loaded ({pretty})")
        else:
            print(f"Info: Road file missing for {spec['label']} ({pretty}) -> {road_path.name}")

    nightlights_data = {}
    for candidate in spec["nightlights_candidates"]:
        nl_path = data_dir / candidate
        if nl_path.exists():
            with open(nl_path, "r", encoding="utf-8") as f:
                nightlights_data = json.load(f)
            print(f"SUCCESS: Nightlights overlay loaded from {candidate}")
            break
    else:
        if spec["nightlights_candidates"]:
            print(
                "Info: Nightlights overlay not found for "
                f"{spec['label']} (tried {', '.join(spec['nightlights_candidates'])})"
            )

    business_types = []
    if business_df is not None and "business_label" in business_df.columns:
        business_types = sorted([x for x in business_df["business_label"].dropna().unique()])

    market_count = int(market_volume_df["mkt_id"].nunique()) if "mkt_id" in market_volume_df.columns else len(market_volume_df)

    region_data = {
        "key": region_key,
        "label": spec["label"],
        "network_df": network_df,
        "market_volume_df": market_volume_df,
        "trader_df": trader_df,
        "business_df": business_df,
        "business_types": business_types,
        "market_types": market_types,
        "trader_types": trader_types,
        "roads_data": roads_data,
        "nightlights_data": nightlights_data,
        "time_periods": tuple(spec["time_periods"]),
        "map_defaults": spec["map_defaults"],
        "bbox_fallback": spec["bbox_fallback"],
        "market_count": market_count,
    }
    for optional_key in ("odisha_outside_offset", "odisha_within_offset"):
        if optional_key in spec:
            region_data[optional_key] = spec[optional_key]

    region_data = _augment_region_scaling(region_data)
    _region_cache[region_key] = region_data
    print(f"--- Completed loading for {spec['label']} ---\n")
    return region_data

DEFAULT_REGION = _canonical_region(os.getenv("INCATA_DEFAULT_REGION", "kenya"))
if DEFAULT_REGION not in REGION_SPECS:
    DEFAULT_REGION = "kenya"

initial_region_data = None
data_load_success = False
try:
    initial_region_data = load_region_data(DEFAULT_REGION)
    data_load_success = True
except FileNotFoundError as e:
    print(f"---! FATAL ERROR !---: {e}")
    print("Please upload the processed data folder for the selected region.")
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

BIZ_CONTROLS_BASE_STYLE = {
    "borderTop": "1px solid #e0e0e0",
    "marginTop": "14px",
    "paddingTop": "12px",
}

def _safe_bool_contains(container, value):
    return bool(container) and (value in container)


def _unique_color_sequence(colors):
    """Return colors with preserved order but without duplicates (case-insensitive)."""
    seen = set()
    unique = []
    for color in colors:
        key = str(color).lower()
        if key not in seen:
            seen.add(key)
            unique.append(color)
    return unique


def _resolve_color_rgb(color_value):
    """Return (r,g,b) tuple for assorted Plotly color formats."""
    try:
        if isinstance(color_value, str):
            lowered = color_value.strip().lower()
            if lowered.startswith("#"):
                return pcolors.hex_to_rgb(color_value)
            if lowered.startswith("rgba"):
                r, g, b, _ = pcolors.unlabel_rgba(color_value)
                return int(r), int(g), int(b)
            r, g, b = pcolors.unlabel_rgb(color_value)
            return int(r), int(g), int(b)
        if isinstance(color_value, (tuple, list)) and len(color_value) >= 3:
            return tuple(int(float(c)) for c in color_value[:3])
    except Exception:
        pass
    return (46, 125, 50)


ORIGIN_COLOR_PALETTE = _unique_color_sequence(
    list(qual_colors.Alphabet)
    + list(qual_colors.Vivid)
    + list(qual_colors.Safe)
)

BAZAR_SAHI_COORDS = (19.37773, 84.56691)


def _format_market_type_label(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"yes", "no"}:
        return f"Wholesaler: {text.capitalize()}"
    return text


def _normalized_series(series, max_value=None):
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    cap = float(max_value) if max_value is not None else float(numeric.max()) if not numeric.empty else 0.0
    if not cap or cap <= 0:
        cap = float(numeric.max()) if not numeric.empty else 1.0
        if not cap:
            cap = 1.0
    normalized = (numeric / cap).clip(lower=0.0, upper=1.0)
    return numeric, normalized, cap


def _resolve_volume_cap(region_data, market_type):
    cap = region_data.get("volume_max_global", 0.0)
    if market_type and market_type != "All Markets":
        cap = region_data.get("volume_max_by_type", {}).get(market_type, cap)
    return cap or 0.0


def _resolve_trader_cap(region_data, market_type, trader_id):
    cap = region_data.get("trader_max_global", 0.0)
    if trader_id and trader_id != "All":
        cap = region_data.get("trader_max_by_trader", {}).get(trader_id, cap)
        if market_type and market_type != "All Markets":
            cap = region_data.get("trader_max_by_market_and_trader", {}).get((market_type, trader_id), cap)
    elif market_type and market_type != "All Markets":
        cap = region_data.get("trader_max_by_market_type", {}).get(market_type, cap)
    return cap or 0.0


def _resolve_business_cap(region_data, time_label):
    return region_data.get("business_max_by_time", {}).get(time_label, region_data.get("business_max_global", 0.0)) or 0.0


def _extract_map_view(relayout_data):
    if not isinstance(relayout_data, dict):
        return None, None
    center = None
    zoom = None
    for key in ("mapbox.center", "map.center"):
        candidate = relayout_data.get(key)
        if isinstance(candidate, dict) and "lat" in candidate and "lon" in candidate:
            try:
                center = {"lat": float(candidate["lat"]), "lon": float(candidate["lon"])}
            except (TypeError, ValueError):
                center = None
            break
    for key in ("mapbox.zoom", "map.zoom"):
        candidate = relayout_data.get(key)
        if candidate is not None:
            try:
                zoom = float(candidate)
            except (TypeError, ValueError):
                zoom = None
            if zoom is not None:
                break
    return center, zoom


def _center_within_region(center, region_data, margin=1.5):
    if not center or "lat" not in center or "lon" not in center:
        return False
    try:
        lat = float(center["lat"])
        lon = float(center["lon"])
    except (TypeError, ValueError):
        return False

    coords = region_data.get("bbox_fallback")
    if coords:
        lats = []
        lons = []
        for pt in coords:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                try:
                    lons.append(float(pt[0]))
                    lats.append(float(pt[1]))
                except (TypeError, ValueError):
                    continue
        if lats and lons:
            lat_min, lat_max = min(lats) - margin, max(lats) + margin
            lon_min, lon_max = min(lons) - margin, max(lons) + margin
        else:
            lat_min = lat - 10
            lat_max = lat + 10
            lon_min = lon - 10
            lon_max = lon + 10
    else:
        defaults = region_data.get("map_defaults", {})
        center_defaults = defaults.get("center", {})
        try:
            default_lat = float(center_defaults.get("lat"))
            default_lon = float(center_defaults.get("lon"))
        except (TypeError, ValueError):
            return True
        lat_min, lat_max = default_lat - 10, default_lat + 10
        lon_min, lon_max = default_lon - 10, default_lon + 10

    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

def _ensure_data_uri(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return s if s.startswith("data:image") else f"data:image/png;base64,{s}"


def _normalize_coords(coords):
    """Return corners as [[lonW,latN],[lonE,latN],[lonE,latS],[lonW,latS]]; fix [lat,lon] if needed."""
    if not coords or len(coords) != 4:
        return None
    fixed = []
    for c in coords:
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            return None
        lon, lat = c
        if abs(lon) <= 90 and abs(lat) > 90:
            lon, lat = lat, lon
        fixed.append([float(lon), float(lat)])
    pts = [{"i": i, "lon": p[0], "lat": p[1]} for i, p in enumerate(fixed)]
    top2 = sorted(sorted(pts, key=lambda x: x["lat"], reverse=True)[:2], key=lambda x: x["lon"])
    bot2 = sorted(sorted(pts, key=lambda x: x["lat"])[:2], key=lambda x: x["lon"])
    return [
        [top2[0]["lon"], top2[0]["lat"]],
        [top2[1]["lon"], top2[1]["lat"]],
        [bot2[1]["lon"], bot2[1]["lat"]],
        [bot2[0]["lon"], bot2[0]["lat"]],
    ]


def _norm_key(s: str) -> str:
    return str(s).strip().lower().replace("_", "").replace(" ", "")


def _default_bbox_coords(region_data):
    """If no coords in JSON, fit to data extent (with padding)."""
    if not region_data:
        return [[33.9, 5.2], [41.9, 5.2], [41.9, -4.9], [33.9, -4.9]]

    lats, lons = [], []
    network_df = region_data.get("network_df")
    trader_df_local = region_data.get("trader_df")
    business_df_local = region_data.get("business_df")

    if network_df is not None:
        for c in ("origin_lat", "market_lat"):
            if c in network_df.columns:
                lats += pd.to_numeric(network_df[c], errors="coerce").dropna().tolist()
        for c in ("origin_lon", "market_lon"):
            if c in network_df.columns:
                lons += pd.to_numeric(network_df[c], errors="coerce").dropna().tolist()

    if trader_df_local is not None:
        if "lat" in trader_df_local.columns:
            lats += pd.to_numeric(trader_df_local["lat"], errors="coerce").dropna().tolist()
        if "lon" in trader_df_local.columns:
            lons += pd.to_numeric(trader_df_local["lon"], errors="coerce").dropna().tolist()

    if business_df_local is not None:
        if "lat" in business_df_local.columns:
            lats += pd.to_numeric(business_df_local["lat"], errors="coerce").dropna().tolist()
        if "lon" in business_df_local.columns:
            lons += pd.to_numeric(business_df_local["lon"], errors="coerce").dropna().tolist()

    if lats and lons:
        lat_s, lat_n = min(lats), max(lats)
        lon_w, lon_e = min(lons), max(lons)
        pad_lat = max(0.1, (lat_n - lat_s) * 0.05)
        pad_lon = max(0.1, (lon_e - lon_w) * 0.05)
        return [
            [lon_w - pad_lon, lat_n + pad_lat],
            [lon_e + pad_lon, lat_n + pad_lat],
            [lon_e + pad_lon, lat_s - pad_lat],
            [lon_w - pad_lon, lat_s - pad_lat],
        ]

    return region_data.get("bbox_fallback") or [[33.9, 5.2], [41.9, 5.2], [41.9, -4.9], [33.9, -4.9]]


def _get_nl_image_and_coords(store, selected_time, region_data):
    """Accept flexible shapes; returns (image_data_uri, coords_list)."""
    default_bbox = _default_bbox_coords(region_data)

    if isinstance(store, str):
        return _ensure_data_uri(store), default_bbox

    if isinstance(store, dict):
        alt = {"Now": "now", "5 Yrs Ago": "5_yrs_ago", "10 Yrs Ago": "10_yrs_ago"}.get(selected_time, selected_time)
        targets = {_norm_key(selected_time), _norm_key(alt)}

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
            return _ensure_data_uri(chosen_val), default_bbox
        if isinstance(chosen_val, (list, tuple)) and len(chosen_val) == 2:
            img, coords = chosen_val
            return _ensure_data_uri(img), (_normalize_coords(coords) or default_bbox)
        if isinstance(chosen_val, dict):
            img = chosen_val.get("image") or chosen_val.get("img")
            coords = chosen_val.get("coordinates") or chosen_val.get("coords")
            return _ensure_data_uri(img), (_normalize_coords(coords) or default_bbox)

    return None, None

def _reposition_odisha_origins(df: pd.DataFrame, region_data: dict) -> pd.DataFrame:
    """Return copy with origin coordinates nudged for Odisha flows."""
    if df is None or df.empty:
        return df

    df = df.copy()
    map_defaults = region_data.get("map_defaults", {}) if region_data else {}
    center = map_defaults.get("center", {"lat": 20.3, "lon": 85.8})
    center_lat = float(center.get("lat", 20.3))
    center_lon = float(center.get("lon", 85.8))
    outside_anchored_lat = 21.5
    outside_anchored_lon = 82.6

    groupings = df.groupby("origin_name", observed=True)
    for origin_name, group in groupings:
        if pd.isna(origin_name):
            continue

        base_lat = float("nan")
        base_lon = float("nan")
        if "origin_lat" in group.columns and "origin_lon" in group.columns:
            base_lat = group["origin_lat"].dropna().mean()
            base_lon = group["origin_lon"].dropna().mean()
        if math.isnan(base_lat) or math.isnan(base_lon):
            base_lat = group["market_lat"].dropna().mean()
            base_lon = group["market_lon"].dropna().mean()
        if math.isnan(base_lat) or math.isnan(base_lon):
            base_lat, base_lon = center_lat, center_lon

        origin_lower = str(origin_name).strip().lower()
        if origin_lower.startswith("outside"):
            new_lat = outside_anchored_lat
            new_lon = outside_anchored_lon
        else:
            new_lat = base_lat
            new_lon = base_lon

        df.loc[group.index, "origin_lat"] = float(new_lat)
        df.loc[group.index, "origin_lon"] = float(new_lon)

    return df

# ------------------------------------------------------------------------------
# 4) LAYOUT
# ------------------------------------------------------------------------------

if data_load_success:
    default_region_key = initial_region_data["key"]
    region_options = [
        {"label": spec["label"], "value": key}
        for key, spec in REGION_SPECS.items()
    ]

    market_types_initial = sorted(initial_region_data.get("market_types") or [])
    initial_market_options = (
        [{"label": "All Markets", "value": "All Markets"}]
        + [{"label": m, "value": m} for m in market_types_initial]
    )

    business_types_initial = sorted(initial_region_data.get("business_types") or [])
    biz_options = (
        [{"label": "All Businesses", "value": "All"}]
        + [{"label": b, "value": b} for b in business_types_initial]
        if business_types_initial
        else [{"label": "All Businesses", "value": "All"}]
    )
    biz_disabled = not business_types_initial

    trader_types_initial = sorted(initial_region_data.get("trader_types") or [])
    trader_options = (
        [{"label": "All Traders", "value": "All"}]
        + [{"label": t, "value": t} for t in trader_types_initial]
    )

    time_periods_initial = list(initial_region_data.get("time_periods") or ["Now"])
    time_marks_initial = {idx: label for idx, label in enumerate(time_periods_initial)}
    time_max_initial = max(len(time_periods_initial) - 1, 0)
    time_default_value = time_max_initial
    time_period_note = (
        "" if len(time_periods_initial) == 3 else "Note: mid-point (5 Yrs Ago) data not available for this region."
    )

    region_info_text = f"Dataset: {initial_region_data['label']} ({initial_region_data['market_count']} markets)"

    hero_section = html.Div(
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
    )

    region_selector_children = [
        html.Div(
            children=[
                html.Label(
                    "Choose Dataset Region",
                    style={"fontWeight": "bold", "color": "#495057", "marginBottom": "8px"},
                ),
                dcc.RadioItems(
                    id="region-selector",
                    options=region_options,
                    value=default_region_key,
                    inline=False,
                    labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "13px"},
                    style={"marginTop": "6px"},
                ),
                dcc.Loading(
                    id="region-loading",
                    type="dot",
                    style={"marginTop": "6px"},
                    children=html.Div(id="region-loading-sentinel", style={"width": 1, "height": 1}),
                ),
            ],
        ),
    ]

    region_controls_children = [
        html.Hr(style={"border": "0", "borderTop": "1px solid #ced4da", "margin": "4px 0 12px"}),
        html.Label(
            "Global Filter: Select Market Type",
            style={"fontWeight": "bold", "display": "block", "color": "#495057", "marginBottom": "8px"},
        ),
        dcc.Dropdown(
            id="master-market-type-filter",
            className="lifted-dropdown",
            options=initial_market_options,
            value="All Markets",
        ),
        html.Div(
            time_period_note,
            id="time-period-note",
            className="muted",
            style={"fontSize": "11px", "marginTop": "8px"},
        ),
    ]

    region_controls_card = html.Div(
        style={
            "background-color": "#e2e3e5",
            "padding": "16px",
            "border-radius": "10px",
            "margin-bottom": "46px",
        },
        children=region_controls_children,
    )

    region_floating_panel = html.Div(
        id={"type": "floating-panel-wrapper", "index": "region"},
        className="floating-controls region-controls",
        children=[
            html.Div(
                id={"type": "panel-header", "index": "region"},
                className="control-panel-header",
                n_clicks=0,
                children=[
                    html.Span("🌍", className="header-icon"),
                    html.Span("Dataset Region", className="header-text"),
                    html.Button("-", id="region-toggle-controls-btn", className="toggle-btn"),
                ],
            ),
            html.Div(
                id={"type": "panel-content", "index": "region"},
                className="controls-content",
                children=region_selector_children,
            ),
        ],
    )

    biz_controls_base_style = BIZ_CONTROLS_BASE_STYLE.copy()

    network_controls_children = [
        html.Div(
            style={"marginBottom": "16px"},
            children=[
                html.Label(
                    "Time Period",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"},
                ),
                dcc.Slider(
                    id="network-time-slider",
                    min=0,
                    max=time_max_initial,
                    step=None,
                    included=False,
                    value=time_default_value,
                    marks=time_marks_initial,
                ),
            ],
        ),
        html.Div(
            id="network-time-note",
            className="muted",
            style={"fontSize": "11px", "marginTop": "-10px", "marginBottom": "12px"},
        ),
        html.Div(
            style={"marginBottom": "16px"},
            children=[
                html.Label(
                    "Season",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"},
                ),
                dcc.RadioItems(
                    id="season-toggle",
                    options=[
                        {"label": "High Season", "value": "High Season"},
                        {"label": "Low Season", "value": "Low Season"},
                    ],
                    value="High Season",
                    labelStyle={"display": "block", "marginBottom": "5px", "fontSize": "12px"},
                ),
            ],
        ),
        html.Div(
            style={"marginBottom": "16px"},
            children=[
                html.Label(
                    "Map Layers",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"},
                ),
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
                html.Label(
                    "Trade Routes",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"},
                ),
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
                            id="opacity-dropdown",
                            className="lifted-dropdown",
                            options=[{"label": f"{i}%", "value": i} for i in range(0, 101, 10)],
                            value=70,
                            clearable=False,
                            searchable=False,
                            style={"width": "80px", "fontSize": "11px"},
                        ),
                    ],
                ),
            ],
        ),
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
        html.Div(
            style={"borderTop": "1px solid #e0e0e0", "paddingTop": "10px", "marginTop": "10px"},
            children=[
                html.Button(
                    "How to Read This Map",
                    id="network-info-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "cursor": "pointer",
                        "border": "1px solid #004085",
                        "backgroundColor": "#e7f3ff",
                        "padding": "6px 10px",
                        "borderRadius": "6px",
                        "fontSize": "12px",
                    },
                )
            ],
        ),
        html.Div(
            id="biz-controls-wrapper",
            style=biz_controls_base_style,
            children=[
                html.Label(
                    "Businesses",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.Checklist(
                    id="biz-toggle",
                    options=[{"label": " Show Businesses", "value": "show"}],
                    value=[],
                    labelStyle={"fontSize": "12px", "marginBottom": "8px"},
                ),
                dcc.Dropdown(
                    id="biz-type-dropdown",
                    className="lifted-dropdown",
                    options=biz_options,
                    value="All",
                    clearable=False,
                    placeholder="Select business type",
                    disabled=biz_disabled,
                ),
                html.Div(style={"height": "8px"}),
                dcc.RadioItems(
                    id="biz-view-mode",
                    options=[
                        {"label": " Points", "value": "points"},
                        {"label": " Heatmap", "value": "heatmap"},
                    ],
                    value="points",
                    inline=True,
                ),
                html.Div(style={"height": "8px"}),
                html.Label("Opacity", style={"fontSize": "12px"}),
                dcc.Slider(
                    id="biz-opacity",
                    min=10,
                    max=100,
                    step=5,
                    value=70,
                    marks={10: "10%", 70: "70%", 100: "100%"},
                ),
            ],
        ),
    ]

    network_floating_panel = html.Div(
        id={"type": "floating-panel-wrapper", "index": "network"},
        className="floating-controls network-controls",
        children=[
            html.Div(
                id={"type": "panel-header", "index": "network"},
                className="control-panel-header",
                n_clicks=0,
                children=[
                    html.Span("⚙️", className="header-icon"),
                    html.Span("Map Controls", className="header-text"),
                    html.Button("-", id="network-toggle-controls-btn", className="toggle-btn"),
                ],
            ),
            html.Div(
                id={"type": "panel-content", "index": "network"},
                className="controls-content",
                children=network_controls_children,
            ),
        ],
    )

    floating_controls_wrapper = html.Div(
        className="floating-panel-container",
        children=[
            network_floating_panel,
            region_floating_panel,
        ],
    )

    network_title = html.Div(
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
    )

    network_info_bubble = html.Div(
        id="network-info-collapse",
        style={
            "display": "none",
            "position": "absolute",
            "bottom": "10px",
            "right": "10px",
            "background-color": "rgba(248, 249, 250, 0.95)",
            "padding": "15px",
            "border": "1px dashed #cce5ff",
            "borderRadius": "6px",
            "zIndex": "998",
            "maxWidth": "420px",
        },
        children=[
            html.Button(
                "Panel",
                id="close-info-btn",
                n_clicks=0,
                style={
                    "position": "absolute",
                    "top": "6px",
                    "right": "10px",
                    "background": "transparent",
                    "border": "none",
                    "fontSize": "12px",
                    "cursor": "pointer",
                },
            ),
            dcc.Markdown(
                """* **Red Dots (Produce Origins):** County/area where tomatoes are sourced.
* **Blue Dots (Markets):** Markets where tomatoes are sold.
* **Lines (Trade Routes):** Connections from origin to market.
* **Line Thickness:** Represents the share of produce from that origin.
* **Business Halos (optional):** Circle size shows number of businesses; color shows nearest distance (km).""",
                style={"fontSize": "12px", "margin": "0"},
            ),
        ],
    )

    network_map_wrapper = html.Div(
        style={"position": "relative", "width": "100%"},
        children=[
            floating_controls_wrapper,
            network_title,
            dcc.Graph(id="network-map", style={"height": "85vh", "width": "100%"}, config=GRAPH_CONFIG),
            html.Div(id="business-legend", className="map-floating-legend"),
            network_info_bubble,
        ],
    )

    network_section = html.Div(
        className="section-card",
        children=[
            html.H2(
                "Produce Flow Network",
                style={"color": "#004085", "border-bottom": "2px solid #b8daff", "padding-bottom": "10px"},
            ),
            html.P(
                "Map shows produce flows from origins to markets. In Kenya the focus is tomatoes; in Odisha the focus is vegetables.",
                style={"marginBottom": "16px"},
            ),
            network_map_wrapper,
        ],
    )

    combined_controls_children = [
        html.Div(
            style={"marginBottom": "18px"},
            children=[
                html.Label(
                    "Analysis Type",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.RadioItems(
                    id="data-type-toggle",
                    options=[
                        {"label": " Tomatoes", "value": "tomatoes"},
                        {"label": " Traders", "value": "traders"},
                    ],
                    value="tomatoes",
                    inline=True,
                    labelStyle={"marginRight": "14px"},
                ),
            ],
        ),
        html.Div(
            style={"marginBottom": "18px"},
            children=[
                html.Label(
                    "View Style",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.RadioItems(
                    id="view-type-toggle",
                    options=[
                        {"label": " Points", "value": "points"},
                        {"label": " Heatmap", "value": "heatmap"},
                    ],
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
                html.Label(
                    "Season",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.RadioItems(
                    id="combined-season-toggle",
                    options=[
                        {"label": "High", "value": "High Season"},
                        {"label": "Low", "value": "Low Season"},
                    ],
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
                html.Label(
                    "Trader Type",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.Dropdown(
                    id="combined-trader-type-dropdown",
                    className="lifted-dropdown",
                    options=trader_options,
                    value="All",
                    placeholder="Select...",
                ),
            ],
        ),
        html.Div(
            style={"marginBottom": "6px"},
            children=[
                html.Label(
                    "Time Period",
                    style={"fontWeight": "bold", "fontSize": "13px", "display": "block", "marginBottom": "6px"},
                ),
                dcc.Slider(
                    id="combined-time-slider",
                    min=0,
                    max=time_max_initial,
                    step=None,
                    included=False,
                    value=time_default_value,
                    marks=time_marks_initial,
                ),
            ],
        ),
    ]

    combined_floating_panel = html.Div(
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
                    html.Button("-", id="combined-toggle-controls-btn", className="toggle-btn"),
                ],
            ),
            html.Div(
                id={"type": "panel-content", "index": "combined"},
                className="controls-content",
                children=combined_controls_children,
            ),
        ],
    )

    combined_title = html.Div(
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
    )

    combined_map_wrapper = html.Div(
        style={"position": "relative", "width": "100%"},
        children=[
            combined_floating_panel,
            combined_title,
            dcc.Graph(id="combined-map", style={"height": "85vh"}, config=GRAPH_CONFIG),
        ],
    )

    combined_section = html.Div(
        className="section-card",
        children=[
            html.H2(
                "Market Concentration Analysis",
                style={"color": "#004085", "border-bottom": "2px solid #b8daff", "padding-bottom": "10px"},
            ),
            html.P(
                "Analyze tomato trade volume or trader concentration. Switch type and view to explore patterns.",
                style={"marginBottom": "20px"},
            ),
            combined_map_wrapper,
        ],
    )

    footer_section = html.Footer(
        [
            html.P(
                "The INCATA project is funded by the Gates Foundation.",
                className="muted",
                style={"fontSize": "0.9em"},
            ),
        ],
        style={"textAlign": "center", "padding": "18px 0", "marginTop": "24px", "borderTop": "1px solid #dee2e6"},
    )

    app.layout = html.Div(
        style={"padding": "2% 5%"},
        children=[
            hero_section,
            region_controls_card,
            network_section,
            combined_section,
            footer_section,
        ],
    )

# ------------------------------------------------------------------------------
# 5) CALLBACKS
# ------------------------------------------------------------------------------
@app.callback(
    [Output({"type": "floating-panel-wrapper", "index": MATCH}, "className"),
     Output({"type": "panel-content", "index": MATCH}, "className")],
    Input({"type": "panel-header", "index": MATCH}, "n_clicks"),
    State({"type": "floating-panel-wrapper", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
def toggle_panel_animation(n, current_class):
    if not n or n <= 0:
        return no_update, no_update

    tokens = (current_class or "").split()
    base_tokens = [token for token in tokens if token not in {"floating-controls", "icon-only"}]

    if "icon-only" in tokens:
        new_class = " ".join(["floating-controls"] + base_tokens) or "floating-controls"
        return new_class, "controls-content"

    new_class = " ".join(["floating-controls"] + base_tokens + ["icon-only"])
    return new_class, "controls-content hidden"


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
    [
        Output("region-loading-sentinel", "children"),
        Output("master-market-type-filter", "options"),
        Output("master-market-type-filter", "value"),
        Output("biz-type-dropdown", "options"),
        Output("biz-type-dropdown", "value"),
        Output("biz-type-dropdown", "disabled"),
        Output("biz-controls-wrapper", "style"),
        Output("combined-trader-type-dropdown", "options"),
        Output("combined-trader-type-dropdown", "value"),
        Output("network-time-slider", "marks"),
        Output("network-time-slider", "max"),
        Output("network-time-slider", "value"),
        Output("combined-time-slider", "marks"),
        Output("combined-time-slider", "max"),
        Output("combined-time-slider", "value"),
        Output("time-period-note", "children"),
        Output("network-time-note", "children"),
        Output("data-type-toggle", "options"),
        Output("data-type-toggle", "value"),
    ],
    Input("region-selector", "value"),
    State("master-market-type-filter", "value"),
    State("biz-type-dropdown", "value"),
    State("combined-trader-type-dropdown", "value"),
    State("network-time-slider", "value"),
    State("combined-time-slider", "value"),
    State("network-time-slider", "marks"),
    State("combined-time-slider", "marks"),
    State("data-type-toggle", "value"),
)
def refresh_region_controls(
    region_key,
    current_market,
    current_biz,
    current_trader,
    current_network_time,
    current_combined_time,
    network_marks_state,
    combined_marks_state,
    current_data_type,
):
    region_data = load_region_data(region_key)

    market_types = sorted(region_data.get("market_types") or [])
    market_options = (
        [{"label": "All Markets", "value": "All Markets"}]
        + [{"label": m, "value": m} for m in market_types]
    )
    market_values = {opt["value"] for opt in market_options}
    market_value = current_market if current_market in market_values else "All Markets"

    business_types = sorted(region_data.get("business_types") or [])
    biz_disabled = not business_types
    biz_options = (
        [{"label": "All Businesses", "value": "All"}]
        + [{"label": b, "value": b} for b in business_types]
        if business_types
        else [{"label": "All Businesses", "value": "All"}]
    )
    biz_values = {opt["value"] for opt in biz_options}
    biz_value = current_biz if current_biz in biz_values else "All"
    if business_types:
        biz_controls_style = BIZ_CONTROLS_BASE_STYLE.copy()
    else:
        hidden_style = BIZ_CONTROLS_BASE_STYLE.copy()
        hidden_style["display"] = "none"
        biz_controls_style = hidden_style

    trader_types = sorted(region_data.get("trader_types") or [])
    trader_options = (
        [{"label": "All Traders", "value": "All"}]
        + [{"label": t, "value": t} for t in trader_types]
    )
    trader_values = {opt["value"] for opt in trader_options}
    trader_value = current_trader if current_trader in trader_values else "All"

    periods = list(region_data.get("time_periods") or ["Now"])
    marks = {idx: label for idx, label in enumerate(periods)}
    max_idx = max(len(periods) - 1, 0)

    def _extract_label(index, marks_dict):
        if marks_dict is None or index is None:
            return None
        if isinstance(marks_dict, dict):
            if index in marks_dict:
                return marks_dict[index]
            key_str = str(index)
            if key_str in marks_dict:
                return marks_dict[key_str]
        return None

    prev_network_label = _extract_label(current_network_time, network_marks_state)
    prev_combined_label = _extract_label(current_combined_time, combined_marks_state)

    if prev_network_label in periods:
        network_value = periods.index(prev_network_label)
    else:
        network_value = max_idx

    if prev_combined_label in periods:
        combined_value = periods.index(prev_combined_label)
    else:
        combined_value = max_idx

    note_text = "" if "5 Yrs Ago" in periods else "Note: mid-point (5 Yrs Ago) data not available for this region."
    network_note = "" if not note_text else f"Available periods: {', '.join(periods)}"

    current_data_type = current_data_type or "tomatoes"
    if region_data["key"] == "odisha":
        data_type_options = [
            {"label": " Traders", "value": "traders"},
        ]
        data_type_value = "traders"
    else:
        data_type_options = [
            {"label": " Tomatoes", "value": "tomatoes"},
            {"label": " Traders", "value": "traders"},
        ]
        data_type_value = current_data_type if current_data_type in {"tomatoes", "traders"} else "tomatoes"

    return (
        "",
        market_options,
        market_value,
        biz_options,
        biz_value,
        biz_disabled,
        biz_controls_style,
        trader_options,
        trader_value,
        marks,
        max_idx,
        network_value,
        marks,
        max_idx,
        combined_value,
        note_text,
        network_note,
        data_type_options,
        data_type_value,
    )


# --------------------------- NETWORK MAP -----------------------------------
@app.callback(
    [
        Output("network-map", "figure"),
        Output("network-map-title", "children"),
        Output("network-loading-sentinel", "children"),
        Output("simplify-map-btn", "children"),
        Output("business-legend", "children"),
        Output("business-legend", "className"),
    ],
    [
        Input("region-selector", "value"),
        Input("master-market-type-filter", "value"),
        Input("season-toggle", "value"),
        Input("network-time-slider", "value"),
        Input("opacity-dropdown", "value"),
        Input("toggle-routes", "value"),
        Input("layer-toggles", "value"),
        Input("biz-toggle", "value"),
        Input("biz-type-dropdown", "value"),
        Input("biz-view-mode", "value"),
        Input("biz-opacity", "value"),
        Input("simplify-map-btn", "n_clicks"),
    ],
    [State("network-map", "relayoutData")],
)
def update_network_map(
    region_key,
    selected_market_type,
    selected_season,
    time_value,
    opacity_percent,
    toggle_value,
    layer_toggles,
    biz_toggle,
    biz_type,
    biz_view_mode,
    biz_opacity,
    simplify_clicks,
    relayout_data,
):
    region_data = load_region_data(region_key)
    region_code = region_data.get("key", str(region_key).lower())
    is_odisha = region_code == "odisha"
    network_df = region_data["network_df"]
    market_volume_df = region_data["market_volume_df"]
    trader_df_local = region_data["trader_df"]
    business_df_local = region_data.get("business_df")
    roads_data = region_data.get("roads_data") or {}
    nightlights_data = region_data.get("nightlights_data") or {}
    time_periods = list(region_data.get("time_periods") or ["Now"])

    try:
        time_index = int(time_value)
    except (TypeError, ValueError):
        time_index = len(time_periods) - 1
    if not time_periods:
        time_periods = ["Now"]
    time_index = max(0, min(time_index, len(time_periods) - 1))
    selected_time = time_periods[time_index]

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
    quantity_label = "tons of vegetables" if is_odisha else "units per day"
    legend_children = []
    legend_class = "map-floating-legend"

    df_flow = network_df[
        (network_df["season"] == selected_season)
        & (network_df["Time Period"] == selected_time)
    ]
    if selected_market_type and selected_market_type != "All Markets":
        df_flow = df_flow[df_flow["mkt_type"] == selected_market_type]
    df_flow = df_flow.copy()
    if {"mkt_id", "market_lat", "market_lon"}.issubset(df_flow.columns):
        df_flow.loc[df_flow["mkt_id"] == 4004, ["market_lat", "market_lon"]] = BAZAR_SAHI_COORDS
    if is_odisha:
        df_flow = _reposition_odisha_origins(df_flow, region_data)
    df_map = df_flow[df_flow["share"] > 0].copy()

    df_vol = market_volume_df[
        (market_volume_df["season"] == selected_season)
        & (market_volume_df["Time Period"] == selected_time)
    ]
    if selected_market_type and selected_market_type != "All Markets":
        df_vol = df_vol[df_vol["mkt_type"] == selected_market_type]
    if {"mkt_id", "lat", "lon"}.issubset(df_vol.columns):
        df_vol.loc[df_vol["mkt_id"] == 4004, ["lat", "lon"]] = BAZAR_SAHI_COORDS

    mapbox_style = MAPBOX_STYLE_DARK if "show_nightlights" in layer_toggles else MAPBOX_STYLE_LIGHT
    layers = []

    if "show_nightlights" in layer_toggles and nightlights_data:
        try:
            img, coords = _get_nl_image_and_coords(nightlights_data, selected_time, region_data)
            if img:
                coords = coords or _default_bbox_coords(region_data)
                layers.append(
                    {
                        "sourcetype": "image",
                        "type": "raster",
                        "source": img,
                        "coordinates": coords,
                        "opacity": 0.70,
                        "below": "traces",
                    }
                )
        except Exception as exc:
            print(f"ERROR: nightlights layer ({region_code}): {exc}")

    if "show_roads" in layer_toggles and selected_time in roads_data:
        road_color = "rgba(255,255,255,0.65)" if "show_nightlights" in layer_toggles else "rgba(40,40,40,0.95)"
        road_width = 1.82 if "show_nightlights" in layer_toggles else 1.4
        layers.append(
            {
                "sourcetype": "geojson",
                "source": roads_data[selected_time],
                "type": "line",
                "color": road_color,
                "line": {"width": road_width},
                "below": "traces",
            }
        )

    defaults = region_data.get("map_defaults") or {"center": {"lat": 0.5, "lon": 37.5}, "zoom": 5.5}
    zoom = float(defaults.get("zoom", 5.5))
    default_center = defaults.get("center", {"lat": 0.5, "lon": 37.5})
    try:
        center = {"lat": float(default_center.get("lat", 0.5)), "lon": float(default_center.get("lon", 37.5))}
    except (TypeError, ValueError):
        center = {"lat": 0.5, "lon": 37.5}

    triggered_props = {item["prop_id"].split(".")[0] for item in (callback_context.triggered or [])}
    region_changed = "region-selector" in triggered_props

    if relayout_data and not region_changed:
        relayout_center, relayout_zoom = _extract_map_view(relayout_data)
        if relayout_center and _center_within_region(relayout_center, region_data):
            center = relayout_center
        if relayout_zoom is not None:
            zoom = relayout_zoom

    fig = go.Figure()

    biz_metrics_for_hover = None
    if show_businesses and business_df_local is not None and selected_time in business_df_local.columns:
        df_biz = business_df_local.copy()
        if selected_market_type and selected_market_type != "All Markets" and "mkt_type" in df_biz.columns:
            df_biz = df_biz[df_biz["mkt_type"] == selected_market_type]
        if biz_type and biz_type != "All" and "business_label" in df_biz.columns:
            df_biz = df_biz[df_biz["business_label"] == biz_type]
        if {"mkt_id", "lat", "lon"}.issubset(df_biz.columns):
            df_biz.loc[df_biz["mkt_id"] == 4004, ["lat", "lon"]] = BAZAR_SAHI_COORDS

        g = (
            df_biz.groupby(["mkt_id", "mkt_name", "mkt_type", "lat", "lon"], observed=True)
            .agg(
                count=(selected_time, "sum"),
                median_km=("nearest_km", lambda s: s.dropna().median() if hasattr(s, "dropna") else pd.NA),
            )
            .reset_index()
        )
        g = g[g["count"] > 0]

        if not g.empty:
            biz_metrics_for_hover = g[["mkt_id", "count", "median_km"]].rename(
                columns={"count": "biz_count", "median_km": "biz_median_km"}
            )
            legend_children = [
                html.Div("Size = number of businesses (by selected time)."),
                html.Div("Color = nearest distance (km)."),
            ]
            legend_class = "map-floating-legend visible"

            counts = pd.to_numeric(g["count"], errors="coerce").clip(lower=0)
            q95 = counts.quantile(0.95) if len(g) > 1 else counts.max()
            base = (q95 ** 0.5) if (pd.notna(q95) and q95 > 0) else 1.0
            scale = 32.0 / base
            g["size"] = 8 + counts.pow(0.5) * scale
            g["size"] = g["size"].clip(lower=8, upper=70)

            biz_cap = _resolve_business_cap(region_data, selected_time)

            if biz_view_mode == "heatmap":
                density_kwargs = dict(
                    lat=g["lat"],
                    lon=g["lon"],
                    z=counts,
                    radius=34,
                    colorscale="Turbo",
                    colorbar=dict(title="Businesses"),
                )
                if biz_cap > 0:
                    density_kwargs["zmin"] = 0
                    density_kwargs["zmax"] = biz_cap
                fig.add_trace(go.Densitymap(**density_kwargs))
                fig.add_trace(
                    go.Scattermap(
                        lat=g["lat"],
                        lon=g["lon"],
                        mode="markers",
                        marker=dict(size=10, color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
                outer = (g["size"] + 12).clip(upper=80)
                inner = (g["size"] + 6).clip(upper=76)
                for sz, alpha in [(outer, 0.35), (inner, 0.90)]:
                    fig.add_trace(
                        go.Scattermap(
                            lat=g["lat"],
                            lon=g["lon"],
                            mode="markers",
                            marker=dict(size=sz, color=f"rgba(255,255,255,{alpha})", opacity=(biz_opacity / 100.0)),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                has_dist = g["median_km"].notna().any()
                color_values = g["median_km"] if has_dist else counts
                cscale = "YlOrRd" if has_dist else "Blues"
                ctitle = "Median dist (km)" if has_dist else "Businesses"
                marker_kwargs = dict(
                    size=g["size"],
                    color=color_values,
                    colorscale=cscale,
                    cmin=0,
                    opacity=(biz_opacity / 100.0),
                    showscale=True,
                    colorbar=dict(title=ctitle),
                )
                if not has_dist and biz_cap > 0:
                    marker_kwargs["cmax"] = biz_cap

                fig.add_trace(
                    go.Scattermap(
                        lat=g["lat"],
                        lon=g["lon"],
                        mode="markers",
                        marker=marker_kwargs,
                        name="Businesses",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    mkts_from_flows = pd.DataFrame(columns=["mkt_id", "mkt_name", "mkt_type", "lat", "lon"])
    if not df_map.empty:
        mkts_from_flows = (
            df_map[["mkt_id", "mkt_name", "mkt_type", "market_lat", "market_lon"]]
            .drop_duplicates()
            .rename(columns={"market_lat": "lat", "market_lon": "lon"})
        )

    mkts_from_volume = df_vol[["mkt_id", "mkt_name", "mkt_type", "lat", "lon"]].drop_duplicates()

    mkts_from_business = pd.DataFrame(columns=["mkt_id", "mkt_name", "mkt_type", "lat", "lon"])
    if business_df_local is not None:
        mkts_from_business = business_df_local[["mkt_id", "mkt_name", "mkt_type", "lat", "lon"]].drop_duplicates()
        if selected_market_type and selected_market_type != "All Markets" and "mkt_type" in mkts_from_business.columns:
            mkts_from_business = mkts_from_business[mkts_from_business["mkt_type"] == selected_market_type]

    pieces = [df for df in (mkts_from_flows, mkts_from_volume, mkts_from_business) if not df.empty]
    if pieces:
        markets_base = pd.concat(pieces, ignore_index=True)
        markets_base = markets_base.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["mkt_id"])
    else:
        markets_base = pd.DataFrame(columns=["mkt_id", "mkt_name", "mkt_type", "lat", "lon"])
    if not markets_base.empty and {"mkt_id", "lat", "lon"}.issubset(markets_base.columns):
        markets_base.loc[markets_base["mkt_id"] == 4004, ["lat", "lon"]] = BAZAR_SAHI_COORDS

    if "show_markers" in (layer_toggles or []):
        opacity = opacity_percent / 100.0
        routes_visible = bool(toggle_value) and ("show" in toggle_value)

        if not df_map.empty:
            if simplify_mode:
                palette = ORIGIN_COLOR_PALETTE or ["#2E7D32"]
                color_cycle = itertools.cycle(palette)
                origin_color_map = {}
                for origin_name, group in df_map.groupby("origin_name", observed=True):
                    color_value = origin_color_map.setdefault(origin_name, next(color_cycle))
                    r, g, b = _resolve_color_rgb(color_value)
                    color_rgba = f"rgba({r}, {g}, {b}, {opacity})"
                    lats = [item for _, row in group.iterrows() for item in (row["origin_lat"], row["market_lat"], None)]
                    lons = [item for _, row in group.iterrows() for item in (row["origin_lon"], row["market_lon"], None)]
                    fig.add_trace(
                        go.Scattermap(
                            lat=lats,
                            lon=lons,
                            mode="lines",
                            line=dict(width=2, color=color_rgba),
                            name=str(origin_name),
                            hoverinfo="none",
                            visible=routes_visible,
                            showlegend=False,
                        )
                    )
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
                        fig.add_trace(
                            go.Scattermap(
                                lat=lats,
                                lon=lons,
                                mode="lines",
                                line=dict(width=s_bin["width"], color=s_bin["color"]),
                                name=s_bin["name"],
                                hoverinfo="none",
                                visible=routes_visible,
                                showlegend=routes_visible,
                            )
                        )

        origins_src = df_flow
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

        supply_series = None
        if not df_map.empty:
            supply_col = next(
                (col for col in ["Trade Quantity", "trade_quantity", "Volume", "volume", "quantity"] if col in df_map.columns),
                None,
            )
            if supply_col:
                supply_series = (
                    df_map.assign(
                        _supply=pd.to_numeric(df_map[supply_col], errors="coerce").fillna(0.0)
                    )
                    .groupby("origin_name", observed=True)["_supply"]
                    .sum()
                )
            elif "share" in df_map.columns:
                supply_series = (
                    df_map.assign(
                        _supply=pd.to_numeric(df_map["share"], errors="coerce").fillna(0.0)
                    )
                    .groupby("origin_name", observed=True)["_supply"]
                    .sum()
                )
        if supply_series is not None:
            origins = origins.merge(supply_series.rename("supply_metric"), on="origin_name", how="left")
        if "supply_metric" not in origins.columns:
            origins["supply_metric"] = origins["market_count"].astype(float)
        origins["supply_metric"] = pd.to_numeric(origins["supply_metric"], errors="coerce").fillna(0.0)

        if simplify_mode:
            origins["size"] = 20.0
            fill_value = 20.0
        else:
            if is_odisha:
                min_origin_size, max_origin_size, exponent = 9.0, 26.0, 0.6
            else:
                min_origin_size, max_origin_size, exponent = 10.0, 34.0, 0.55

            supply_values = origins["supply_metric"].clip(lower=0)
            max_supply = float(supply_values.max()) if not supply_values.empty else 0.0
            if max_supply > 0:
                normalized_supply = (supply_values / max_supply).clip(lower=0.0, upper=1.0)
            else:
                normalized_supply = pd.Series(0.0, index=origins.index)

            origins["size"] = min_origin_size + normalized_supply.pow(exponent) * (max_origin_size - min_origin_size)
            fill_value = min_origin_size
        origins["size"] = origins["size"].fillna(fill_value)
        origins["hover_text"] = (
            origins["origin_name"] + "<br>Supplies " + origins["market_count"].astype(int).astype(str) + " market(s)"
        )

        if not origins.empty:
            fig.add_trace(
                go.Scattermap(
                    lat=origins["origin_lat"],
                    lon=origins["origin_lon"],
                    mode="markers",
                    marker=dict(size=origins["size"], color="#a50f15", opacity=0.9),
                    name="Produce Origin",
                    text=origins["hover_text"],
                    hoverinfo="text",
                )
            )

        if not markets_base.empty:
            market_hover_info = pd.DataFrame(columns=["mkt_name", "details"])
            if not df_map.empty:
                market_hover_info = (
                    df_map.assign(origin_share_str=df_map["origin_name"].astype(str) + ": " + df_map["share"].astype(int).astype(str) + "%")
                    .groupby("mkt_name", observed=True)["origin_share_str"].apply("<br>".join)
                    .reset_index(name="details")
                )

            markets = (
                markets_base
                .merge(df_vol[["mkt_id", "Total Volume"]], on="mkt_id", how="left")
                .merge(market_hover_info, on="mkt_name", how="left")
            ).fillna({"Total Volume": 0, "details": "n/a"})

            if show_businesses and (biz_metrics_for_hover is not None):
                markets = markets.merge(biz_metrics_for_hover, on="mkt_id", how="left")
            else:
                markets["biz_count"] = pd.NA
                markets["biz_median_km"] = pd.NA

            def _fmt_int(val):
                return "n/a" if pd.isna(val) else f"{int(round(float(val))):,}"

            def _fmt_km(val):
                return "NA" if pd.isna(val) else f"{float(val):.2f} km"

            volume_formatted = markets["Total Volume"].round(0).astype(int).apply(lambda x: f"{x:,}")
            formatted_types = markets["mkt_type"].apply(_format_market_type_label)
            base_text = (
                "<b>" + markets["mkt_name"] + "</b><br><i>" + formatted_types + "</i><br>"
                + "Trade Quantity: " + volume_formatted + f" {quantity_label}<br>"
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
                markets["size"] = 22.0
                fill_value = 22.0
            else:
                if is_odisha:
                    min_market_size, max_market_size, vol_exponent = 9.0, 30.0, 0.85
                else:
                    min_market_size, max_market_size, vol_exponent = 10.0, 38.0, 0.75
                volumes = markets["Total Volume"].clip(lower=0).astype(float)
                q95 = float(volumes.quantile(0.95)) if len(volumes) > 1 else float(volumes.max())
                denom = q95 if (not math.isnan(q95) and q95 > 0) else 1.0
                normalized_volume = (volumes / denom).clip(lower=0.0, upper=1.0)
                markets["size"] = min_market_size + normalized_volume.pow(vol_exponent) * (max_market_size - min_market_size)
                fill_value = min_market_size
            markets["size"] = markets["size"].fillna(fill_value)
            fig.add_trace(
                go.Scattermap(
                    lat=markets["lat"],
                    lon=markets["lon"],
                    mode="markers",
                    marker=dict(size=markets["size"], color="blue", opacity=0.9),
                    name="Market",
                    text=markets["hover_text"],
                    hoverinfo="text",
                )
            )
        else:
            fig.add_annotation(
                text="No markets found for this selection.",
                showarrow=False,
                font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"),
            )
    fig.add_trace(
        go.Scattermap(
            lat=[center.get("lat", 0.0)],
            lon=[center.get("lon", 0.0)],
            mode="markers",
            marker=dict(size=0.01, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="none",
        )
    )

    fig.update_layout(
        margin=dict(r=0, l=0, b=0, t=0),
        uirevision=f"network-{region_code}",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.92,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=11),
        ),
        map=dict(style=mapbox_style, layers=layers, zoom=zoom, center=center),
        transition={"duration": 650, "easing": "cubic-in-out"},
    )

    map_title = f"{region_data['label']}: {selected_season} - {selected_time}"
    return fig, map_title, "", simplify_label, legend_children, legend_class


# --------------------------- COMBINED MAP ----------------------------------
@app.callback(
    [
        Output("combined-map", "figure"),
        Output("combined-map-title", "children"),
        Output("combined-loading-sentinel", "children"),
    ],
    [
        Input("region-selector", "value"),
        Input("master-market-type-filter", "value"),
        Input("data-type-toggle", "value"),
        Input("view-type-toggle", "value"),
        Input("combined-trader-type-dropdown", "value"),
        Input("combined-season-toggle", "value"),
        Input("combined-time-slider", "value"),
    ],
    [State("combined-map", "relayoutData")],
)
def update_combined_map(
    region_key,
    market_type,
    data_type,
    view_type,
    selected_trader,
    selected_season,
    time_value,
    relayout_data,
):
    region_data = load_region_data(region_key)
    region_code = region_data.get("key", str(region_key).lower())
    market_volume_df = region_data["market_volume_df"]
    trader_df_local = region_data["trader_df"]
    time_periods = list(region_data.get("time_periods") or ["Now"])

    try:
        time_index = int(time_value)
    except (TypeError, ValueError):
        time_index = len(time_periods) - 1
    if not time_periods:
        time_periods = ["Now"]
    time_index = max(0, min(time_index, len(time_periods) - 1))
    selected_time = time_periods[time_index]

    fig = go.Figure()

    if data_type == "tomatoes":
        df = market_volume_df.copy()
        if market_type and market_type != "All Markets":
            df = df[df["mkt_type"] == market_type]
        df = df[(df["season"] == selected_season) & (df["Time Period"] == selected_time) & (df["Total Volume"] > 0)]
        if {"mkt_id", "lat", "lon"}.issubset(df.columns):
            df.loc[df["mkt_id"] == 4004, ["lat", "lon"]] = BAZAR_SAHI_COORDS
        if region_code == "odisha":
            title_parts = [region_data['label'], "Vegetable Trade Volume", selected_season, selected_time]
            colorbar_title = "Vegetable Volume (tons)"
        else:
            title_parts = [region_data['label'], "Tomato Trade Volume", selected_season, selected_time]
            colorbar_title = "Tomato Volume (units/day)"
        z_value_col = "Total Volume"
        colorscale = "Viridis"
        value_cap = _resolve_volume_cap(region_data, market_type)
    else:
        df = trader_df_local.copy()
        if market_type and market_type != "All Markets" and "mkt_type" in df.columns:
            df = df[df["mkt_type"] == market_type]
        if "trader_id" in df.columns and selected_trader and selected_trader != "All":
            df = df[df["trader_id"] == selected_trader]
        df = df.groupby(["mkt_name", "lat", "lon"], observed=True)[selected_time].sum().reset_index()
        df = df[df[selected_time] > 0]
        if "mkt_name" in df.columns and {"lat", "lon"}.issubset(df.columns):
            df.loc[df["mkt_name"] == "Bazar Sahi Hat", ["lat", "lon"]] = BAZAR_SAHI_COORDS
        trader_label = selected_trader if (selected_trader and selected_trader != "All") else "All Traders"
        title_parts = [region_data['label'], trader_label, selected_time]
        z_value_col = selected_time
        colorscale = "Plasma"
        colorbar_title = "No. of Traders"
        value_cap = _resolve_trader_cap(region_data, market_type, selected_trader)

    map_title = " - ".join([part for part in title_parts if part])

    if df.empty:
        fig.add_annotation(text="No data available for this selection.", showarrow=False)
    else:
        values_numeric, values_normalized, value_cap = _normalized_series(
            df[z_value_col],
            value_cap if value_cap else None,
        )
        value_strings = values_numeric.clip(lower=0).round(0).astype(int).apply(lambda x: f"{x:,}")
        df["hover_text"] = "<b>" + df["mkt_name"] + "</b><br>" + colorbar_title + ": " + value_strings
        if view_type == "points":
            if data_type == "tomatoes":
                df["size"] = 5 + values_numeric.clip(lower=0).pow(0.5) * 0.1
            else:
                min_size, max_size = (6.0, 30.0) if region_code == "odisha" else (5.0, 22.0)
                values = values_numeric.clip(lower=0)
                q95 = float(values.quantile(0.95)) if len(values) > 1 else float(values.max())
                denom = q95 if (not math.isnan(q95) and q95 > 0) else 1.0
                normalized = (values / denom).clip(lower=0.0, upper=1.0)
                df["size"] = min_size + normalized.pow(0.7) * (max_size - min_size)
            marker_kwargs = dict(
                size=df["size"],
                color=values_numeric,
                colorscale=colorscale,
                cmin=0,
                showscale=True,
                colorbar=dict(title=colorbar_title),
            )
            if value_cap and value_cap > 0:
                marker_kwargs["cmax"] = value_cap
            fig.add_trace(
                go.Scattermap(
                    lat=df["lat"],
                    lon=df["lon"],
                    mode="markers",
                    marker=marker_kwargs,
                    text=df["hover_text"],
                    hoverinfo="text",
                )
            )
        else:
            heatmap_radius = 30 if data_type == "traders" else 20
            density_kwargs = dict(
                lat=df["lat"],
                lon=df["lon"],
                z=values_numeric,
                radius=heatmap_radius,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
            )
            if value_cap and value_cap > 0:
                density_kwargs["zmin"] = 0
                density_kwargs["zmax"] = value_cap
            fig.add_trace(go.Densitymap(**density_kwargs))
            fig.add_trace(
                go.Scattermap(
                    lat=df["lat"],
                    lon=df["lon"],
                    mode="markers",
                    marker=dict(size=10, color="rgba(0,0,0,0)"),
                    text=df["hover_text"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    defaults = region_data.get("map_defaults") or {"center": {"lat": 0.5, "lon": 37.5}, "zoom": 5.5}
    zoom = float(defaults.get("zoom", 5.5))
    default_center = defaults.get("center", {"lat": 0.5, "lon": 37.5})
    try:
        center = {"lat": float(default_center.get("lat", 0.5)), "lon": float(default_center.get("lon", 37.5))}
    except (TypeError, ValueError):
        center = {"lat": 0.5, "lon": 37.5}

    triggered_props = {item["prop_id"].split(".")[0] for item in (callback_context.triggered or [])}
    region_changed = "region-selector" in triggered_props

    if relayout_data and not region_changed:
        relayout_center, relayout_zoom = _extract_map_view(relayout_data)
        if relayout_center and _center_within_region(relayout_center, region_data):
            center = relayout_center
        if relayout_zoom is not None:
            zoom = relayout_zoom

    fig.update_layout(
        margin=dict(r=0, l=0, b=0, t=0),
        uirevision=f"combined-{region_code}",
        map=dict(style=MAPBOX_STYLE_LIGHT, zoom=zoom, center=center),
        transition={"duration": 650, "easing": "cubic-in-out"},
    )
    return fig, map_title, ""

# ------------------------------------------------------------------------------
# 6) RUN


# ------------------------------------------------------------------------------
# 6) RUN
# ------------------------------------------------------------------------------
# app.py (at bottom)
if __name__ == "__main__":
    is_render = bool(os.getenv("RENDER") or os.getenv("PORT"))
    host = "0.0.0.0" if is_render else "127.0.0.1"
    port = int(os.getenv("PORT", "8050"))
    debug_flag = os.getenv("DASH_DEBUG", "1") == "1"
    print(f"→ Open http://localhost:{port}")
    app.run(host=host, port=port, debug=debug_flag)





















