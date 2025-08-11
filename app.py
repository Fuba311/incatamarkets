# app.py
# --- PRODUCTION VERSION FOR RENDER DEPLOYMENT ---
print("--- STARTING INCATA MARKET ANALYSIS DASHBOARD ---")
import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import os

# --- 1. APP INITIALIZATION ---
app = dash.Dash(__name__, assets_folder='assets')
server = app.server  # CRITICAL: Expose server for Gunicorn/Render
app.title = "INCATA Market Analysis Dashboard"

# --- 2. DATA LOADING (OPTIMIZED) ---
print("--- Loading Pre-Processed Data ---")
PROCESSED_DATA_FOLDER = Path(__file__).parent / "processed_data"

# Initialize empty containers
network_df = None
market_volume_df = None
trader_df = None
roads_data = {}
nightlights_data = {}
data_load_success = False

try:
    # Load parquet files
    network_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'network_df.parquet')
    market_volume_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'market_volume_df.parquet')
    trader_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'trader_df.parquet')
    print("SUCCESS: Tabular data (.parquet) loaded.")
    
    # Load geospatial data
    time_period_map = {"10 Yrs Ago": "10_yrs_ago", "5 Yrs Ago": "5_yrs_ago", "Now": "now"}
    for key, file_suffix in time_period_map.items():
        road_path = PROCESSED_DATA_FOLDER / f"roads_{file_suffix}_processed.geojson"
        if road_path.exists():
            roads_gdf = gpd.read_file(road_path)
            roads_data[key] = roads_gdf.__geo_interface__
        else:
            print(f"Warning: Road file not found at {road_path}")
    if roads_data:
        print("SUCCESS: Roads data (.geojson) loaded.")
    
    # Load nightlights data (can be: dict of strings, dict of [img, coords], or just a string)
    nightlight_path = PROCESSED_DATA_FOLDER / "nightlights_data.json"
    if nightlight_path.exists():
        with open(nightlight_path, 'r') as f:
            nightlights_data = json.load(f)
        print("SUCCESS: Nightlights data (.json) loaded.")
    else:
        print(f"Warning: Nightlight JSON file not found at {nightlight_path}")
    
    print("--- All Data Loaded Successfully ---")
    print(f"Unique trader types found: {trader_df['trader_id'].unique()}")
    data_load_success = True
    
except FileNotFoundError as e:
    print(f"---! FATAL ERROR !---: {e}")
    print("Please ensure the 'processed_data' folder is uploaded to Render.")
except Exception as e:
    print(f"---! UNEXPECTED ERROR !---: {e}")

# --- 3. ENHANCED STYLES ---
colors = {
    'primary': '#2563eb',
    'primary_dark': '#1e40af',
    'secondary': '#10b981',
    'accent': '#f59e0b',
    'danger': '#ef4444',
    'background': '#f8fafc',
    'surface': '#ffffff',
    'text': '#1e293b',
    'text_light': '#64748b',
    'border': '#e2e8f0',
}

section_style = {
    'background': f'linear-gradient(135deg, {colors["surface"]} 0%, #f1f5f9 100%)',
    'border': f'1px solid {colors["border"]}',
    'border-radius': '16px',
    'padding': '32px',
    'box-shadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    'margin-bottom': '48px',
    'position': 'relative',
    'overflow': 'hidden'
}

main_title_style = {
    'textAlign': 'center', 
    'color': colors['primary'], 
    'fontSize': '2.5rem',
    'fontWeight': '700',
    'letterSpacing': '-0.025em',
    'marginBottom': '8px'
}

subtitle_style = {
    'textAlign': 'center',
    'color': colors['text'],
    'fontSize': '1.125rem',
    'fontWeight': '400',
    'marginBottom': '4px'
}

# Map styles (Carto GL JSON styles)
map_style_light = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
map_style_dark = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

button_style = {
    'backgroundColor': colors['primary'],
    'color': 'white',
    'border': 'none',
    'borderRadius': '8px',
    'padding': '10px 20px',
    'fontSize': '14px',
    'fontWeight': '500',
    'cursor': 'pointer',
    'transition': 'all 0.2s ease',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
}

info_button_style = {
    **button_style,
    'backgroundColor': colors['surface'],
    'color': colors['primary'],
    'border': f'2px solid {colors["primary"]}',
    'width': '100%'
}

# --- 3.5 HELPERS FOR NIGHTLIGHTS (image-only support) -------------------------
def _to_data_uri(b64_or_uri: str, mime="image/png") -> str:
    """Ensure the image is a proper data URI for Map's image source."""
    if not isinstance(b64_or_uri, str):
        return ""
    s = b64_or_uri.strip()
    return s if s.startswith("data:image") else f"data:{mime};base64,{s}"

def _default_nl_bbox():
    """
    Compute a reasonable bbox from available points (markets, origins, traders).
    Falls back to a Kenya-wide bbox if nothing is found.
    Returns (lonW, lonE, latS, latN).
    """
    lats, lons = [], []
    # network_df origins/markets
    if network_df is not None:
        for col in ("origin_lat", "market_lat"):
            if col in network_df.columns:
                lats += list(pd.to_numeric(network_df[col], errors="coerce").dropna().values)
        for col in ("origin_lon", "market_lon"):
            if col in network_df.columns:
                lons += list(pd.to_numeric(network_df[col], errors="coerce").dropna().values)
    # trader_df points
    if trader_df is not None:
        if "lat" in trader_df.columns:
            lats += list(pd.to_numeric(trader_df["lat"], errors="coerce").dropna().values)
        if "lon" in trader_df.columns:
            lons += list(pd.to_numeric(trader_df["lon"], errors="coerce").dropna().values)
    if lats and lons:
        latS, latN = float(min(lats)), float(max(lats))
        lonW, lonE = float(min(lons)), float(max(lons))
        # pad by 5% so the image extends a bit beyond points
        pad_lon = max(0.1, (lonE - lonW) * 0.05)
        pad_lat = max(0.1, (latN - latS) * 0.05)
        return (lonW - pad_lon, lonE + pad_lon, latS - pad_lat, latN + pad_lat)
    # Kenya fallback
    return (33.9, 41.9, -4.9, 5.2)

def _normalize_image_coords(coords):
    """
    Expect 4 corners in order [TL, TR, BR, BL] as [lon, lat].
    If coords is None (image-only), we build from default bbox.
    Also accepts coords as [[lon,lat]...], [[lat,lon]...], any order.
    """
    if coords is None:
        lonW, lonE, latS, latN = _default_nl_bbox()
        return [[lonW, latN], [lonE, latN], [lonE, latS], [lonW, latS]]
    # sanitize possible lat/lon order + ordering
    def _maybe_swap(pt):
        a, b = float(pt[0]), float(pt[1])
        return [b, a] if abs(a) <= 90 and abs(b) > 90 else [a, b]
    pts = [_maybe_swap(p) for p in coords]
    try:
        enrich = [{"lon": p[0], "lat": p[1], "raw": p} for p in pts]
        top2 = sorted(sorted(enrich, key=lambda x: x["lat"], reverse=True)[:2], key=lambda x: x["lon"])
        bot2 = sorted(sorted(enrich, key=lambda x: x["lat"])[:2], key=lambda x: x["lon"])
        return [top2[0]["raw"], top2[1]["raw"], bot2[1]["raw"], bot2[0]["raw"]]
    except Exception:
        return pts

def _get_nl_image_and_coords(nl_store, selected_time):
    """
    Flexible getter:
      - dict with key == selected_time -> value
      - dict with 1 entry -> use that
      - plain string -> use as image
      - value can be string (image only) or [image, coords]
    Returns (image_str, coords_or_None) or (None, None) if not found.
    """
    # direct string (entire JSON is just the image)
    if isinstance(nl_store, str):
        return nl_store, None
    # dictionary
    if isinstance(nl_store, dict):
        if selected_time in nl_store:
            val = nl_store[selected_time]
        elif len(nl_store) == 1:
            val = next(iter(nl_store.values()))
        else:
            # try lowercase keys
            key_map = {"Now": "now", "5 Yrs Ago": "5_yrs_ago", "10 Yrs Ago": "10_yrs_ago"}
            want = key_map.get(selected_time, selected_time).lower()
            # find any key that matches lower
            match = [nl_store[k] for k in nl_store if k.lower() == want]
            val = match[0] if match else None
        if val is None:
            return None, None
        if isinstance(val, str):
            return val, None
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return val[0], val[1]
        if isinstance(val, dict) and "image" in val:
            return val.get("image"), val.get("coords")
    return None, None

# --- 4. APP LAYOUT ---
if data_load_success:
    app.layout = html.Div(style={
        'fontFamily': "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif",
        'padding': '0',
        'margin': '0',
        'background': f'linear-gradient(180deg, {colors["background"]} 0%, #e0e7ff 100%)',
        'minHeight': '100vh'
    }, children=[
        
        # Hero Header Section
        html.Div(style={
            'background': f'linear-gradient(135deg, {colors["primary"]} 0%, {colors["primary_dark"]} 100%)',
            'color': 'white',
            'padding': '48px 24px',
            'marginBottom': '48px',
            'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            'position': 'relative',
            'overflow': 'hidden'
        }, children=[
            # Decorative background pattern
            html.Div(style={
                'position': 'absolute',
                'top': '0',
                'right': '-100px',
                'width': '300px',
                'height': '300px',
                'background': 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
                'borderRadius': '50%'
            }),
            html.Div(style={'position': 'relative', 'zIndex': '1'}, children=[
                html.H1("INCATA Market Analysis Dashboard", style={
                    'textAlign': 'center',
                    'fontSize': '3rem',
                    'fontWeight': '800',
                    'marginBottom': '16px',
                    'textShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                html.H2("Transforming Agricultural Markets in Kenya", style={
                    'textAlign': 'center',
                    'fontSize': '1.5rem',
                    'fontWeight': '300',
                    'marginBottom': '24px',
                    'opacity': '0.95'
                }),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.Span("In Partnership with ", style={'fontSize': '14px', 'opacity': '0.9'}),
                    html.Span("RIMISP • MSU • IFPRI • Tegemeo Institute", style={
                        'fontSize': '14px',
                        'fontWeight': '600',
                        'letterSpacing': '0.05em'
                    })
                ]),
                html.P("Linked Farms and Enterprises for Inclusive Agricultural Transformation", style={
                    'textAlign': 'center',
                    'fontSize': '13px',
                    'fontStyle': 'italic',
                    'marginTop': '12px',
                    'opacity': '0.85'
                })
            ])
        ]),
        
        # Main Content Container
        html.Div(style={'padding': '0 5%', 'maxWidth': '1600px', 'margin': '0 auto'}, children=[
            
            # Global Filter Card
            html.Div(style={
                'background': colors['surface'],
                'padding': '24px',
                'borderRadius': '12px',
                'marginBottom': '48px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'border': f'1px solid {colors["border"]}'
            }, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}, children=[
                    html.Span("🎯", style={'fontSize': '24px', 'marginRight': '12px'}),
                    html.Label("Filter by Market Type", style={
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'color': colors['text']
                    })
                ]),
                dcc.Dropdown(
                    id='master-market-type-filter',
                    options=[{'label': 'All Markets', 'value': 'All Markets'}] + 
                            [{'label': mtype, 'value': mtype} for mtype in sorted(network_df['mkt_type'].unique())],
                    value='All Markets',
                    style={'fontSize': '14px'}
                )
            ]),
            
            # NETWORK MAP SECTION
            html.Div(style={**section_style}, children=[
                # Section Header
                html.Div(style={'marginBottom': '24px'}, children=[
                    html.H2("🌍 Produce Flow Network", style={
                        'color': colors['primary'],
                        'fontSize': '1.875rem',
                        'fontWeight': '700',
                        'marginBottom': '12px'
                    }),
                    html.P("Visualize the origin and flow patterns of tomato trade across Kenya. Hover over markers for detailed information.", 
                          style={'color': colors['text_light'], 'fontSize': '15px', 'lineHeight': '1.6'}),
                    html.Div(style={
                        'background': f'linear-gradient(90deg, {colors["accent"]} 0%, {colors["secondary"]} 100%)',
                        'height': '3px',
                        'width': '80px',
                        'borderRadius': '2px',
                        'marginTop': '12px'
                    })
                ]),
                
                # Map Container
                html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                    # Enhanced Floating Panel
                    html.Div(id={'type': 'floating-panel-wrapper', 'index': 'network'}, 
                            className='floating-controls enhanced', children=[
                        html.Div(id={'type': 'panel-header', 'index': 'network'}, 
                                className='control-panel-header', n_clicks=0, children=[
                            html.Span("⚙️", className='header-icon'),
                            html.Span("Map Controls", className='header-text'),
                            html.Button('−', id='network-toggle-controls-btn', className='toggle-btn')
                        ]),
                        html.Div(id={'type': 'panel-content', 'index': 'network'}, 
                                className='controls-content', children=[
                            # Time Period Control
                            html.Div(className='control-group', children=[
                                html.Label("📅 Time Period", className='control-label'),
                                dcc.Slider(
                                    id='network-time-slider',
                                    min=0, max=2,
                                    marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'},
                                    value=2, step=None, included=False,
                                    className='custom-slider'
                                ),
                            ]),
                            # Season Control
                            html.Div(className='control-group', children=[
                                html.Label("🌱 Season", className='control-label'),
                                dcc.RadioItems(
                                    id='season-toggle',
                                    options=[
                                        {'label': ' High Season', 'value': 'High Season'},
                                        {'label': ' Low Season', 'value': 'Low Season'}
                                    ],
                                    value='High Season',
                                    className='custom-radio'
                                ),
                            ]),
                            # Map Layers Control
                            html.Div(className='control-group', children=[
                                html.Label("🗺 Map Layers", className='control-label'),
                                dcc.Checklist(
                                    id='layer-toggles',
                                    options=[
                                        {'label': ' Markets & Origins', 'value': 'show_markers'},
                                        {'label': ' Roads', 'value': 'show_roads'},
                                        {'label': ' Nightlights', 'value': 'show_nightlights'}
                                    ],
                                    value=['show_markers'],
                                    className='custom-checklist'
                                ),
                            ]),
                            # Trade Routes Control
                            html.Div(className='control-group', children=[
                                html.Label("🔗 Trade Routes", className='control-label'),
                                dcc.Checklist(
                                    id='toggle-routes',
                                    options=[{'label': ' Show Trade Routes', 'value': 'show'}],
                                    value=['show'],
                                    className='custom-checklist'
                                ),
                                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginTop': '12px'}, children=[
                                    html.Label("Opacity:", style={'fontSize': '13px', 'color': colors['text_light']}),
                                    dcc.Dropdown(
                                        id='opacity-dropdown',
                                        options=[{'label': f'{i}%', 'value': i} for i in range(0, 101, 10)],
                                        value=70,
                                        clearable=False,
                                        searchable=False,
                                        style={'width': '90px', 'fontSize': '12px'}
                                    ),
                                ]),
                            ]),
                            # Info Button
                            html.Button('ℹ️ How to Read This Map', 
                                      id='network-info-button',
                                      n_clicks=0,
                                      style=info_button_style),
                        ]),
                    ]),
                    
                    # Map Title Badge
                    html.Div(id='network-map-title', style={
                        'position': 'absolute',
                        'top': '16px',
                        'left': '50%',
                        'transform': 'translateX(-50%)',
                        'background': 'rgba(255, 255, 255, 0.98)',
                        'padding': '10px 24px',
                        'borderRadius': '24px',
                        'fontSize': '14px',
                        'fontWeight': '600',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
                        'zIndex': '999',
                        'color': colors['primary'],
                        'border': f'2px solid {colors["primary"]}',
                        'backdropFilter': 'blur(10px)'
                    }),
                    
                    # Map Graph
                    dcc.Graph(
                        id='network-map',
                        style={'height': '85vh', 'width': '100%', 'borderRadius': '12px', 'overflow': 'hidden'},
                        config={'scrollZoom': True, 'displayModeBar': False}
                    ),
                    
                    # Info Collapse Panel
                    html.Div(id='network-info-collapse', style={
                        'display': 'none',
                        'position': 'absolute',
                        'bottom': '20px',
                        'right': '20px',
                        'background': 'rgba(255, 255, 255, 0.98)',
                        'padding': '20px',
                        'borderRadius': '12px',
                        'zIndex': '998',
                        'maxWidth': '400px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                        'backdropFilter': 'blur(10px)'
                    }, children=[
                        html.Button('✕', id='close-info-btn', n_clicks=0, style={
                            'position': 'absolute',
                            'top': '8px',
                            'right': '12px',
                            'background': 'transparent',
                            'border': 'none',
                            'fontSize': '20px',
                            'cursor': 'pointer',
                            'color': colors['text_light']
                        }),
                        html.H4("Map Legend", style={'color': colors['primary'], 'marginBottom': '12px', 'fontSize': '16px'}),
                        dcc.Markdown('''
**🔴 Red Markers** - Produce origins (counties/areas)  
**🔵 Blue Markers** - Market locations  
**📍 Lines** - Trade routes from origin to market  
**Line Thickness** - Volume/share of produce flow
                        ''', style={'fontSize': '13px', 'lineHeight': '1.8', 'color': colors['text']})
                    ])
                ]),
            ]),
            
            # MARKET CONCENTRATION SECTION
            html.Div(style={**section_style}, children=[
                html.Div(style={'marginBottom': '24px'}, children=[
                    html.H2("📊 Market Concentration Analysis", style={
                        'color': colors['primary'],
                        'fontSize': '1.875rem',
                        'fontWeight': '700',
                        'marginBottom': '12px'
                    }),
                    html.P("Analyze tomato trade volume and trader concentration patterns across different markets.", 
                          style={'color': colors['text_light'], 'fontSize': '15px', 'lineHeight': '1.6'}),
                    html.Div(style={
                        'background': f'linear-gradient(90deg, {colors["secondary"]} 0%, {colors["accent"]} 100%)',
                        'height': '3px',
                        'width': '80px',
                        'borderRadius': '2px',
                        'marginTop': '12px'
                    })
                ]),
                
                html.H3(id='combined-map-title', style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'fontSize': '1.25rem',
                    'marginBottom': '20px'
                }),
                
                html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                    html.Div(id={'type': 'floating-panel-wrapper', 'index': 'combined'}, 
                            className='floating-controls enhanced', children=[
                        html.Div(id={'type': 'panel-header', 'index': 'combined'}, 
                                className='control-panel-header', n_clicks=0, children=[
                            html.Span("⚙️", className='header-icon'),
                            html.Span("Analysis Controls", className='header-text'),
                            html.Button('−', id='combined-toggle-controls-btn', className='toggle-btn')
                        ]),
                        html.Div(id={'type': 'panel-content', 'index': 'combined'}, 
                                className='controls-content', children=[
                            html.Div(className='control-group', children=[
                                html.Label("📊 Analysis Type", className='control-label'),
                                dcc.RadioItems(
                                    id='data-type-toggle',
                                    options=[
                                        {'label': ' Tomatoes', 'value': 'tomatoes'},
                                        {'label': ' Traders', 'value': 'traders'}
                                    ],
                                    value='tomatoes',
                                    inline=True,
                                    className='custom-radio-inline'
                                ),
                            ]),
                            html.Div(className='control-group', children=[
                                html.Label("🎨 View Style", className='control-label'),
                                dcc.RadioItems(
                                    id='view-type-toggle',
                                    options=[
                                        {'label': ' Points', 'value': 'points'},
                                        {'label': ' Heatmap', 'value': 'heatmap'}
                                    ],
                                    value='points',
                                    inline=True,
                                    className='custom-radio-inline'
                                ),
                            ]),
                            html.Div(id='season-control-div', className='control-group conditional-control', children=[
                                html.Label("🌱 Season", className='control-label'),
                                dcc.RadioItems(
                                    id='combined-season-toggle',
                                    options=[
                                        {'label': ' High', 'value': 'High Season'},
                                        {'label': ' Low', 'value': 'Low Season'}
                                    ],
                                    value='High Season',
                                    inline=True,
                                    className='custom-radio-inline'
                                ),
                            ]),
                            html.Div(id='trader-control-div', className='control-group conditional-control', children=[
                                html.Label("👤 Trader Type", className='control-label'),
                                dcc.Dropdown(
                                    id='combined-trader-type-dropdown',
                                    options=[{'label': 'All Traders', 'value': 'All'}] + 
                                           [{'label': t, 'value': t} for t in sorted(trader_df['trader_id'].unique()) if pd.notna(t)],
                                    value='All',
                                    placeholder="Select trader type...",
                                    style={'fontSize': '13px'}
                                ),
                            ]),
                            html.Div(className='control-group', children=[
                                html.Label("📅 Time Period", className='control-label'),
                                dcc.Slider(
                                    id='combined-time-slider',
                                    min=0, max=2,
                                    marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'},
                                    value=2, step=None, included=False,
                                    className='custom-slider'
                                ),
                            ]),
                        ]),
                    ]),
                    
                    dcc.Graph(
                        id='combined-map',
                        style={'height': '85vh', 'borderRadius': '12px', 'overflow': 'hidden'},
                        config={'scrollZoom': True, 'displayModeBar': False}
                    ),
                ]),
            ]),
        ]),
        
        # Enhanced Footer
        html.Footer(style={
            'background': f'linear-gradient(135deg, {colors["primary"]} 0%, {colors["primary_dark"]} 100%)',
            'color': 'white',
            'padding': '48px 24px',
            'marginTop': '64px',
            'textAlign': 'center'
        }, children=[
            html.Div(style={'maxWidth': '800px', 'margin': '0 auto'}, children=[
                html.P("The INCATA project is funded by", style={
                    'fontSize': '14px',
                    'opacity': '0.9',
                    'marginBottom': '12px'
                }),
                html.H3("Bill & Melinda Gates Foundation", style={
                    'fontSize': '1.5rem',
                    'fontWeight': '600',
                    'marginBottom': '24px'
                }),
                html.P("© 2025 INCATA Project. All rights reserved.", style={
                    'fontSize': '13px',
                    'opacity': '0.8',
                    'marginTop': '24px',
                    'paddingTop': '24px',
                    'borderTop': '1px solid rgba(255,255,255,0.2)'
                })
            ])
        ])
    ])
else:
    # Error layout if data fails to load
    app.layout = html.Div(style={
        'fontFamily': "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
        'padding': '48px 24px',
        'background': f'linear-gradient(180deg, {colors["background"]} 0%, #fee2e2 100%)',
        'minHeight': '100vh',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    }, children=[
        html.Div(style={
            'background': 'white',
            'borderRadius': '16px',
            'padding': '48px',
            'maxWidth': '600px',
            'boxShadow': '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
            'textAlign': 'center'
        }, children=[
            html.Div("⚠️", style={'fontSize': '64px', 'marginBottom': '24px'}),
            html.H1("Data Loading Error", style={
                'color': colors['danger'],
                'marginBottom': '16px',
                'fontSize': '2rem',
                'fontWeight': '700'
            }),
            html.P("Could not load the required data files.", style={
                'fontSize': '18px',
                'color': colors['text'],
                'marginBottom': '32px'
            }),
            html.Div(style={
                'background': colors['background'],
                'borderRadius': '8px',
                'padding': '24px',
                'textAlign': 'left'
            }, children=[
                html.P("Please ensure the following files are in the 'processed_data' folder:", style={
                    'fontWeight': '600',
                    'marginBottom': '16px',
                    'color': colors['text']
                }),
                html.Ul([
                    html.Li("✓ network_df.parquet", style={'marginBottom': '8px'}),
                    html.Li("✓ market_volume_df.parquet", style={'marginBottom': '8px'}),
                    html.Li("✓ trader_df.parquet", style={'marginBottom': '8px'}),
                    html.Li("✓ roads_*_processed.geojson files", style={'marginBottom': '8px'}),
                    html.Li("✓ nightlights_data.json")
                ], style={'listStyle': 'none', 'padding': '0', 'color': colors['text_light']})
            ]),
            html.P("Check the application logs for detailed error information.", style={
                'marginTop': '24px',
                'color': colors['text_light'],
                'fontSize': '14px'
            })
        ])
    ])

# --- 5. CALLBACKS (only if data loaded successfully) ---
if data_load_success:
    
    @app.callback(
        [Output({'type': 'floating-panel-wrapper', 'index': MATCH}, 'className'),
         Output({'type': 'panel-content', 'index': MATCH}, 'className')],
        Input({'type': 'panel-header', 'index': MATCH}, 'n_clicks'),
        State({'type': 'floating-panel-wrapper', 'index': MATCH}, 'className'),
        prevent_initial_call=True
    )
    def toggle_panel_animation(n, current_class):
        if n and n > 0:
            if 'icon-only' in current_class:
                return 'floating-controls enhanced', 'controls-content'
            else:
                return 'floating-controls enhanced icon-only', 'controls-content hidden'
        return no_update, no_update

    @app.callback(
        Output('network-info-collapse', 'style'),
        [Input('network-info-button', 'n_clicks'), Input('close-info-btn', 'n_clicks')],
        State('network-info-collapse', 'style'),
        prevent_initial_call=True
    )
    def toggle_network_info(info_clicks, close_clicks, current_style):
        ctx = callback_context
        if not ctx.triggered: return current_style
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'network-info-button' and current_style.get('display') == 'none':
            return {**current_style, 'display': 'block', 'animation': 'fadeIn 0.3s'}
        return {**current_style, 'display': 'none'}

    @app.callback(
        [Output('network-map', 'figure'), Output('network-map-title', 'children')],
        [Input('master-market-type-filter', 'value'), Input('season-toggle', 'value'),
         Input('network-time-slider', 'value'), Input('opacity-dropdown', 'value'),
         Input('toggle-routes', 'value'), Input('layer-toggles', 'value')],
        [State('network-map', 'relayoutData')]
    )
    def update_network_map(selected_market_type, selected_season, time_value, opacity_percent, toggle_value, layer_toggles, relayout_data):
        time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
        selected_time = time_map[time_value]
        layer_toggles = layer_toggles or []
        if opacity_percent is None: opacity_percent = 70

        df_flow = network_df[(network_df['season'] == selected_season) & (network_df['Time Period'] == selected_time)]
        if selected_market_type != 'All Markets':
            df_flow = df_flow[df_flow['mkt_type'] == selected_market_type]
        df_map = df_flow[df_flow['share'] > 0].copy()
        df_vol = market_volume_df[(market_volume_df['season'] == selected_season) & (market_volume_df['Time Period'] == selected_time)]
        
        map_title = f'{selected_season} - {selected_time}'
        map_style = map_style_dark if 'show_nightlights' in layer_toggles else map_style_light
        layers = []

        # --- NIGHTLIGHTS: accept image-only or [image, coords] ------------------
        if 'show_nightlights' in layer_toggles:
            img, coords = _get_nl_image_and_coords(nightlights_data, selected_time)
            if img:
                try:
                    data_uri = _to_data_uri(img)
                    coords = _normalize_image_coords(coords)  # build bbox if None
                    layers.append({
                        "sourcetype": "image",
                        "source": data_uri,
                        "coordinates": coords,   # [TL, TR, BR, BL] in [lon,lat]
                        "opacity": 0.85,
                        "below": "traces"        # render under all traces
                    })
                except Exception as e:
                    print(f"Nightlights overlay error for '{selected_time}': {e}")

        # --- ROADS ABOVE NIGHTLIGHTS, under traces ------------------------------
        if 'show_roads' in layer_toggles and selected_time in roads_data:
            road_color = 'rgba(211, 211, 211, 0.75)' if 'show_nightlights' in layer_toggles else 'rgba(100, 100, 100, 0.7)'
            layers.append({
                "sourcetype": "geojson",
                "source": roads_data[selected_time],
                "type": "line",
                "color": road_color,
                "line": {"width": 0.9},
                "below": "traces"
            })

        # Viewport persistence
        zoom, center = (5.5, {"lat": 0.5, "lon": 37.5})
        if relayout_data and 'map.center' in relayout_data:
            zoom = relayout_data.get('map.zoom', zoom)
            center = relayout_data.get('map.center', center)

        fig = go.Figure()
        
        # Markers / routes
        if 'show_markers' in layer_toggles:
            opacity = opacity_percent / 100.0
            routes_visible = isinstance(toggle_value, (list, tuple)) and ('show' in toggle_value)
            share_bins = [
                {'name': 'High Share (>75%)', 'data': df_map[df_map['share'] >= 75], 'width': 4, 'color': f'rgba(217, 95, 2, {opacity})'},
                {'name': 'Medium Share (25-75%)', 'data': df_map[(df_map['share'] < 75) & (df_map['share'] >= 25)], 'width': 2, 'color': f'rgba(117, 112, 179, {opacity})'},
                {'name': 'Low Share (<25%)', 'data': df_map[df_map['share'] < 25], 'width': 1, 'color': f'rgba(102, 166, 30, {opacity})'}
            ]
            for s_bin in share_bins:
                if not s_bin['data'].empty:
                    lats = [item for _, row in s_bin['data'].iterrows() for item in (row['origin_lat'], row['market_lat'], None)]
                    lons = [item for _, row in s_bin['data'].iterrows() for item in (row['origin_lon'], row['market_lon'], None)]
                    fig.add_trace(go.Scattermap(
                        lat=lats, lon=lons, mode='lines',
                        line=dict(width=s_bin['width'], color=s_bin['color']),
                        name=s_bin['name'], hoverinfo='none', visible=routes_visible))

            if not df_map.empty:
                origins = df_map[['origin_name', 'origin_lat', 'origin_lon']].drop_duplicates() \
                    .merge(df_map.groupby('origin_name', observed=True)['mkt_name'].nunique().reset_index(name='market_count'), on='origin_name')
                origins['hover_text'] = origins['origin_name'] + '<br>Supplies ' + origins['market_count'].astype(str) + ' market(s)'
                fig.add_trace(go.Scattermap(
                    lat=origins['origin_lat'], lon=origins['origin_lon'], mode='markers',
                    marker=dict(size=(5 + origins['market_count']), color='#ef4444', opacity=0.9),
                    name='Produce Origin', text=origins['hover_text'], hoverinfo='text'))

                markets = df_map[['mkt_id', 'mkt_name', 'market_lat', 'market_lon', 'mkt_type']].drop_duplicates() \
                    .merge(df_vol[['mkt_id', 'Total Volume']], on='mkt_id', how='left').fillna(0)
                market_hover_info = df_map.assign(origin_share_str=df_map['origin_name'].astype(str) + ': ' + df_map['share'].astype(int).astype(str) + '%') \
                    .groupby('mkt_name', observed=True)['origin_share_str'].apply('<br>'.join).reset_index(name='details')
                markets = markets.merge(market_hover_info, on='mkt_name')
                markets['hover_text'] = '<b>' + markets['mkt_name'] + '</b><br><i>' + markets['mkt_type'] + '</i><br>' \
                    + 'Trade Quantity: ' + markets['Total Volume'].round(0).astype(int).apply(lambda x: f'{x:,}') + ' units<br>' \
                    + '--- Origins ---<br>' + markets['details']
                markets['size'] = 4 + (markets['Total Volume'] ** 0.5) * 0.08
                fig.add_trace(go.Scattermap(
                    lat=markets['market_lat'], lon=markets['market_lon'], mode='markers',
                    marker=dict(size=markets['size'], color='#2563eb', opacity=0.9),
                    name='Market', text=markets['hover_text'], hoverinfo='text'))
            else:
                fig.add_annotation(text="No trade flow data for this selection.", showarrow=False,
                                   font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"))

        # keep Map happy even with only layers
        fig.add_trace(go.Scattermap(lat=[0], lon=[37.5], mode='markers',
                                    marker=dict(size=0.1, color='rgba(0,0,0,0)'),
                                    showlegend=False, hoverinfo='none'))
        
        fig.update_layout(
            margin={"r":0, "t":0, "l":0, "b":0},
            showlegend=True,
            legend=dict(yanchor="top", y=0.92, xanchor="right", x=0.99,
                        bgcolor='rgba(255,255,255,0.95)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1,
                        traceorder='normal', itemsizing='constant', font=dict(size=11)),
            map=dict(style=map_style, layers=layers, zoom=zoom, center=center),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        if 'show_roads' in layer_toggles:
            fig.add_trace(go.Scattermap(mode='lines', lon=[None], lat=[None],
                                        line=dict(color='rgba(100, 100, 100, 0.7)', width=2),
                                        name='Road Network'))
        return fig, map_title

    @app.callback(
        [Output('trader-control-div', 'className'),
         Output('season-control-div', 'className')],
        Input('data-type-toggle', 'value')
    )
    def update_combined_map_controls(data_type):
        if data_type == 'traders':
            return 'control-group conditional-control', 'control-group conditional-control hidden'
        else:
            return 'control-group conditional-control hidden', 'control-group conditional-control'

    @app.callback(
        [Output('combined-map', 'figure'), Output('combined-map-title', 'children')],
        [Input('master-market-type-filter', 'value'),
         Input('data-type-toggle', 'value'),
         Input('view-type-toggle', 'value'),
         Input('combined-trader-type-dropdown', 'value'),
         Input('combined-season-toggle', 'value'),
         Input('combined-time-slider', 'value')],
        [State('combined-map', 'relayoutData')]
    )
    def update_combined_map(market_type, data_type, view_type, selected_trader, selected_season, time_value, relayout_data):
        time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
        selected_time = time_map[time_value]
        
        fig = go.Figure()

        if data_type == 'tomatoes':
            df = market_volume_df.copy()
            if market_type != 'All Markets': df = df[df['mkt_type'] == market_type]
            df = df[(df['season'] == selected_season) & (df['Time Period'] == selected_time) & (df['Total Volume'] > 0)]
            title_parts = ['Tomato Trade Volume', selected_season, selected_time]
            z_value_col = 'Total Volume'
            colorscale = 'Viridis'
            colorbar_title = 'Trade Volume'
        else:
            df = trader_df.copy()
            if market_type != 'All Markets': df = df[df['mkt_type'] == market_type]
            if selected_trader != 'All': df = df[df['trader_id'] == selected_trader]
            df = df.groupby(['mkt_name', 'lat', 'lon'], observed=True)[selected_time].sum().reset_index()
            df = df[df[selected_time] > 0]
            title_parts = [f'{selected_trader} Traders' if selected_trader != 'All' else 'All Traders', selected_time]
            z_value_col = selected_time
            colorscale = 'Plasma'
            colorbar_title = 'No. of Traders'
        
        map_title = ' - '.join(title_parts)

        if df.empty:
            fig.add_annotation(text=f"No data available for this selection.", showarrow=False)
        else:
            df['hover_text'] = '<b>' + df['mkt_name'] + '</b><br>' + colorbar_title + ': ' + df[z_value_col].round(0).astype(int).apply(lambda x: f'{x:,}')
            if view_type == 'points':
                df['size'] = 5 + (df[z_value_col] ** 0.5) * (0.1 if data_type == 'tomatoes' else 0.8)
                fig.add_trace(go.Scattermap(
                    lat=df['lat'], lon=df['lon'], mode='markers',
                    marker=dict(
                        size=df['size'],
                        color=df[z_value_col],
                        colorscale=colorscale, cmin=0, cmax=df[z_value_col].quantile(0.95),
                        showscale=True, colorbar_title_text=colorbar_title
                    ),
                    text=df['hover_text'], hoverinfo='text'
                ))
            else:
                heatmap_radius = 30 if data_type == 'traders' else 20
                fig.add_trace(go.Densitymap(
                    lat=df['lat'], lon=df['lon'], z=df[z_value_col],
                    radius=heatmap_radius, colorscale=colorscale,
                    colorbar_title_text=colorbar_title
                ))
                fig.add_trace(go.Scattermap(
                    lat=df['lat'], lon=df['lon'], mode='markers',
                    marker=dict(size=10, color='rgba(0,0,0,0)'),
                    text=df['hover_text'], hoverinfo='text', showlegend=False
                ))

        zoom, center = (5.5, {"lat": 0.5, "lon": 37.5})
        if relayout_data and 'map.center' in relayout_data:
            zoom = relayout_data['map.zoom']
            center = relayout_data['map.center']

        fig.update_layout(
            margin={"r":0, "t":0, "l":0, "b":0},
            map=dict(style=map_style_light, zoom=zoom, center=center),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig, map_title

# --- 6. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=False)
