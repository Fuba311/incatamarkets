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
    
    # Load nightlights data
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
# Modern gradient backgrounds and improved color scheme
app_background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
section_background = 'rgba(255, 255, 255, 0.98)'
header_gradient = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'

section_style = {
    'background': section_background,
    'border': 'none',
    'border-radius': '20px',
    'padding': '35px',
    'box-shadow': '0 10px 40px rgba(0, 0, 0, 0.1)',
    'margin-bottom': '40px',
    'position': 'relative',
    'overflow': 'hidden'
}

title_style = {
    'textAlign': 'center',
    'color': '#2c3e50',
    'marginBottom': '20px',
    'fontSize': '1.5rem',
    'fontWeight': '600'
}

map_style_light = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
map_style_dark = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# Custom CSS as a string to inject
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif !important;
}

/* Animated gradient background */
.main-container {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass morphism effect for sections */
.glass-section {
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* Enhanced floating controls */
.floating-controls {
    position: absolute;
    top: 70px;
    right: 20px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    width: 280px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.floating-controls.icon-only {
    width: 50px;
    height: 50px;
    overflow: hidden;
}

.control-panel-header {
    padding: 12px 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px 15px 0 0;
    font-weight: 500;
    transition: all 0.3s ease;
}

.control-panel-header:hover {
    filter: brightness(1.1);
}

.controls-content {
    padding: 20px;
    max-height: 500px;
    overflow-y: auto;
    transition: all 0.3s ease;
}

.controls-content.hidden {
    display: none;
}

/* Beautiful buttons */
.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

/* Dropdown styling */
.Select-control {
    border-radius: 10px !important;
    border: 1px solid #e0e0e0 !important;
    transition: all 0.3s ease !important;
}

.Select-control:hover {
    border-color: #667eea !important;
}

/* Slider enhancements */
.rc-slider-track {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.rc-slider-handle {
    border: 3px solid #667eea !important;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
}

/* Info panel styling */
.info-panel {
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

/* Map title badge */
.map-title-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px 24px;
    border-radius: 30px;
    font-weight: 600;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    letter-spacing: 0.5px;
}

/* Footer enhancement */
.footer-section {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border-radius: 20px;
    padding: 30px;
    margin-top: 50px;
}

/* Smooth animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}
</style>
"""

# --- 4. APP LAYOUT ---
if data_load_success:
    app.layout = html.Div([
        # Inject custom CSS
        html.Div(custom_css, dangerously_allow_html=True),
        
        html.Div(className='main-container', style={'padding': '0'}, children=[
            html.Div(style={'padding': '2% 5%', 'background': 'rgba(255, 255, 255, 0.02)'}, children=[
                
                # Header Section with gradient
                html.Div(style={
                    'background': 'white',
                    'border-radius': '20px',
                    'padding': '40px',
                    'margin-bottom': '40px',
                    'box-shadow': '0 10px 40px rgba(0, 0, 0, 0.1)',
                    'text-align': 'center'
                }, children=[
                    html.H1("INCATA Market Analysis Dashboard", 
                           style={
                               'background': header_gradient,
                               'background-clip': 'text',
                               '-webkit-background-clip': 'text',
                               'color': 'transparent',
                               'font-size': '2.5rem',
                               'font-weight': '700',
                               'margin-bottom': '10px'
                           }),
                    html.H4("A Public Dashboard for Markets Studied under Project INCATA", 
                           style={'color': '#5a6c7d', 'font-weight': '400', 'margin-bottom': '15px'}),
                    html.Div(style={'display': 'flex', 'justify-content': 'center', 'gap': '20px', 'flex-wrap': 'wrap', 'margin-top': '20px'}, children=[
                        html.Span("RIMISP", style={'background': 'rgba(102, 126, 234, 0.1)', 'padding': '5px 15px', 'border-radius': '20px', 'color': '#667eea', 'font-weight': '500'}),
                        html.Span("Michigan State University", style={'background': 'rgba(118, 75, 162, 0.1)', 'padding': '5px 15px', 'border-radius': '20px', 'color': '#764ba2', 'font-weight': '500'}),
                        html.Span("IFPRI", style={'background': 'rgba(102, 126, 234, 0.1)', 'padding': '5px 15px', 'border-radius': '20px', 'color': '#667eea', 'font-weight': '500'}),
                        html.Span("Tegemeo Institute", style={'background': 'rgba(118, 75, 162, 0.1)', 'padding': '5px 15px', 'border-radius': '20px', 'color': '#764ba2', 'font-weight': '500'})
                    ]),
                    html.P("Linked Farms and Enterprises for Inclusive Agricultural Transformation in Africa and Asia", 
                          style={'font-style': 'italic', 'color': '#8795a1', 'margin-top': '15px', 'font-size': '0.95rem'})
                ]),
                
                # Global Filter with enhanced styling
                html.Div(style={
                    'background': 'white',
                    'padding': '25px',
                    'border-radius': '15px',
                    'margin-bottom': '50px',
                    'box-shadow': '0 5px 20px rgba(0, 0, 0, 0.08)'
                }, children=[
                    html.Label("🎯 Global Filter: Select Market Type", 
                              style={'font-weight': '600', 'display': 'block', 'color': '#2c3e50', 'margin-bottom': '12px', 'font-size': '1.1rem'}),
                    dcc.Dropdown(
                        id='master-market-type-filter',
                        options=[{'label': 'All Markets', 'value': 'All Markets'}] + [{'label': mtype, 'value': mtype} for mtype in sorted(network_df['mkt_type'].unique())],
                        value='All Markets',
                        style={'border-radius': '10px'}
                    )
                ]),
                
                # NETWORK MAP SECTION with glass morphism
                html.Div(className='glass-section', style={**section_style, 'position': 'relative'}, children=[
                    html.Div(style={
                        'position': 'absolute',
                        'top': '0',
                        'left': '0',
                        'right': '0',
                        'height': '5px',
                        'background': header_gradient,
                        'border-radius': '20px 20px 0 0'
                    }),
                    html.H2("🗺️ Produce Flow Network", 
                           style={'color': '#2c3e50', 'font-weight': '600', 'margin-bottom': '15px', 'margin-top': '10px'}),
                    html.P("Visualize the origin and flow of tomatoes across Kenya. Trade routes show connections between production areas and markets.", 
                          style={'color': '#5a6c7d', 'margin-bottom': '10px', 'font-size': '0.95rem'}),
                    html.P("📍 Note: Origin locations (red dots) represent approximate county positions.", 
                          style={'background': 'rgba(102, 126, 234, 0.1)', 'padding': '10px 15px', 'border-radius': '10px', 'color': '#667eea', 'font-size': '0.9rem', 'margin-bottom': '20px'}),
                    
                    html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                        html.Div(id={'type': 'floating-panel-wrapper', 'index': 'network'}, className='floating-controls fade-in', children=[
                            html.Div(id={'type': 'panel-header', 'index': 'network'}, className='control-panel-header', n_clicks=0, children=[
                                html.Span("⚙️", className='header-icon'),
                                html.Span("Map Controls", className='header-text'),
                                html.Button('−', id='network-toggle-controls-btn', className='toggle-btn', style={'background': 'transparent', 'border': 'none', 'color': 'white', 'font-size': '20px', 'margin-left': 'auto'})
                            ]),
                            html.Div(id={'type': 'panel-content', 'index': 'network'}, className='controls-content', children=[
                                html.Div(style={'margin-bottom': '25px'}, children=[
                                    html.Label("📅 Time Period", style={'font-weight': '600', 'font-size': '13px', 'margin-bottom': '12px', 'display': 'block', 'color': '#2c3e50'}),
                                    dcc.Slider(id='network-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False),
                                ]),
                                html.Div(style={'margin-bottom': '25px'}, children=[
                                    html.Label("🌱 Season", style={'font-weight': '600', 'font-size': '13px', 'margin-bottom': '12px', 'display': 'block', 'color': '#2c3e50'}),
                                    dcc.RadioItems(id='season-toggle', options=[{'label': ' High Season', 'value': 'High Season'}, {'label': ' Low Season', 'value': 'Low Season'}], value='High Season', labelStyle={'display': 'block', 'margin-bottom': '8px', 'font-size': '12px'}),
                                ]),
                                html.Div(style={'margin-bottom': '25px'}, children=[
                                    html.Label("🗺️ Map Layers", style={'font-weight': '600', 'font-size': '13px', 'margin-bottom': '12px', 'display': 'block', 'color': '#2c3e50'}),
                                    dcc.Checklist(id='layer-toggles', options=[{'label': ' Markets & Origins', 'value': 'show_markers'}, {'label': ' Roads', 'value': 'show_roads'}, {'label': ' Nightlights', 'value': 'show_nightlights'}], value=['show_markers'], labelStyle={'display': 'block', 'margin-bottom': '8px', 'font-size': '12px'}),
                                ]),
                                html.Div(style={'margin-bottom': '20px'}, children=[
                                    html.Label("🔗 Trade Routes", style={'font-weight': '600', 'font-size': '13px', 'margin-bottom': '12px', 'display': 'block', 'color': '#2c3e50'}),
                                    dcc.Checklist(id='toggle-routes', options=[{'label': ' Show Trade Routes', 'value': 'show'}], value=['show'], labelStyle={'font-size': '12px', 'margin-bottom': '10px'}),
                                    html.Div(style={'display': 'flex', 'align-items': 'center', 'gap': '10px', 'margin-top': '12px'}, children=[
                                        html.Label("Opacity:", style={'font-size': '12px', 'min-width': '50px', 'color': '#5a6c7d'}),
                                        dcc.Dropdown(id='opacity-dropdown', options=[{'label': f'{i}%', 'value': i} for i in range(0, 101, 10)], value=70, clearable=False, searchable=False, style={'width': '90px', 'font-size': '11px'}),
                                    ]),
                                ]),
                                html.Div(style={'border-top': '1px solid rgba(0,0,0,0.1)', 'padding-top': '15px', 'margin-top': '15px'}, children=[
                                    html.Button('ℹ️ How to Read This Map', id='network-info-button', n_clicks=0, className='btn-primary', style={'width': '100%', 'font-size': '12px'}),
                                ]),
                            ]),
                        ]),
                        html.Div(id='network-map-title', className='map-title-badge', style={'position': 'absolute', 'top': '15px', 'left': '50%', 'transform': 'translateX(-50%)', 'z-index': '999'}),
                        dcc.Graph(id='network-map', style={'height': '85vh', 'width': '100%', 'border-radius': '15px', 'overflow': 'hidden'}, config={'scrollZoom': True}),
                        html.Div(id='network-info-collapse', className='info-panel', style={'display': 'none', 'position': 'absolute', 'bottom': '20px', 'right': '20px', 'z-index': '998', 'max-width': '400px'}, children=[
                            html.Button('✕', id='close-info-btn', n_clicks=0, style={'position': 'absolute', 'top': '10px', 'right': '15px', 'background': 'transparent', 'border': 'none', 'font-size': '20px', 'cursor': 'pointer', 'color': '#5a6c7d'}),
                            html.H4("Map Legend", style={'color': '#2c3e50', 'margin-bottom': '15px', 'font-weight': '600'}),
                            dcc.Markdown('''
**🔴 Red Dots (Produce Origins)**  
Counties or areas where tomatoes are sourced

**🔵 Blue Dots (Markets)**  
Markets where tomatoes are sold

**➖ Lines (Trade Routes)**  
Connections from origin to market

**Line Thickness**  
Represents the share of produce from that origin
                            ''', style={'font-size': '13px', 'line-height': '1.6', 'color': '#5a6c7d'})
                        ])
                    ]),
                ]),
                
                # COMBINED ANALYSIS MAP SECTION
                html.Div(className='glass-section', style={**section_style, 'position': 'relative'}, children=[
                    html.Div(style={
                        'position': 'absolute',
                        'top': '0',
                        'left': '0',
                        'right': '0',
                        'height': '5px',
                        'background': header_gradient,
                        'border-radius': '20px 20px 0 0'
                    }),
                    html.H2("📊 Market Concentration Analysis", 
                           style={'color': '#2c3e50', 'font-weight': '600', 'margin-bottom': '15px', 'margin-top': '10px'}),
                    html.P("Analyze tomato trade volume and trader concentration patterns across different markets and time periods.", 
                          style={'color': '#5a6c7d', 'margin-bottom': '20px', 'font-size': '0.95rem'}),
                    html.H3(id='combined-map-title', style={**title_style, 'color': '#667eea'}),
                    html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                        html.Div(id={'type': 'floating-panel-wrapper', 'index': 'combined'}, className='floating-controls fade-in', children=[
                            html.Div(id={'type': 'panel-header', 'index': 'combined'}, className='control-panel-header', n_clicks=0, children=[
                                html.Span("⚙️", className='header-icon'),
                                html.Span("Analysis Controls", className='header-text'),
                                html.Button('−', id='combined-toggle-controls-btn', className='toggle-btn', style={'background': 'transparent', 'border': 'none', 'color': 'white', 'font-size': '20px', 'margin-left': 'auto'})
                            ]),
                            html.Div(id={'type': 'panel-content', 'index': 'combined'}, className='controls-content', children=[
                                html.Div(style={'margin-bottom': '25px'}, children=[
                                    html.Label("📊 Analysis Type", style={'font-weight': '600', 'font-size': '13px', 'display': 'block', 'margin-bottom': '12px', 'color': '#2c3e50'}),
                                    dcc.RadioItems(id='data-type-toggle', options=[{'label': ' Tomatoes', 'value': 'tomatoes'}, {'label': ' Traders', 'value': 'traders'}], value='tomatoes', inline=True, labelStyle={'margin-right': '15px', 'font-size': '12px'}),
                                ]),
                                html.Div(style={'margin-bottom': '25px'}, children=[
                                    html.Label("🎨 View Style", style={'font-weight': '600', 'font-size': '13px', 'display': 'block', 'margin-bottom': '12px', 'color': '#2c3e50'}),
                                    dcc.RadioItems(id='view-type-toggle', options=[{'label': ' Points', 'value': 'points'}, {'label': ' Heatmap', 'value': 'heatmap'}], value='points', inline=True, labelStyle={'margin-right': '15px', 'font-size': '12px'}),
                                ]),
                                html.Div(id='season-control-div', className='conditional-control', style={'margin-bottom': '25px'}, children=[
                                    html.Label("🌱 Season", style={'font-weight': '600', 'font-size': '13px', 'display': 'block', 'margin-bottom': '12px', 'color': '#2c3e50'}),
                                    dcc.RadioItems(id='combined-season-toggle', options=[{'label': ' High', 'value': 'High Season'}, {'label': ' Low', 'value': 'Low Season'}], value='High Season', inline=True, labelStyle={'font-size': '12px'}),
                                ]),
                                html.Div(id='trader-control-div', className='conditional-control', style={'margin-bottom': '25px'}, children=[
                                    html.Label("👤 Trader Type", style={'font-weight': '600', 'font-size': '13px', 'display': 'block', 'margin-bottom': '12px', 'color': '#2c3e50'}),
                                    dcc.Dropdown(id='combined-trader-type-dropdown', options=[{'label': 'All Traders', 'value': 'All'}] + [{'label': t, 'value': t} for t in sorted(trader_df['trader_id'].unique()) if pd.notna(t)], value='All', placeholder="Select trader type..."),
                                ]),
                                html.Div(style={'margin-bottom': '10px'}, children=[
                                    html.Label("📅 Time Period", style={'font-weight': '600', 'font-size': '13px', 'display': 'block', 'margin-bottom': '12px', 'color': '#2c3e50'}),
                                    dcc.Slider(id='combined-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False),
                                ]),
                            ]),
                        ]),
                        dcc.Graph(id='combined-map', style={'height': '85vh', 'border-radius': '15px', 'overflow': 'hidden'}),
                    ]),
                ]),
                
                # Enhanced Footer
                html.Footer(className='footer-section', children=[
                    html.Div(style={'text-align': 'center'}, children=[
                        html.H4("About INCATA", style={'color': '#2c3e50', 'margin-bottom': '15px', 'font-weight': '600'}),
                        html.P("The INCATA project is funded by the Bill & Melinda Gates Foundation", 
                              style={'color': '#5a6c7d', 'font-size': '0.95rem', 'margin-bottom': '10px'}),
                        html.P("© 2025 INCATA Project. All rights reserved.", 
                              style={'color': '#8795a1', 'font-size': '0.85rem'})
                    ])
                ])
            ])
        ])
    ])
else:
    # Enhanced error layout
    app.layout = html.Div([
        html.Div(custom_css, dangerously_allow_html=True),
        html.Div(style={
            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'min-height': '100vh',
            'display': 'flex',
            'align-items': 'center',
            'justify-content': 'center',
            'padding': '20px'
        }, children=[
            html.Div(style={
                'background': 'white',
                'border-radius': '20px',
                'padding': '40px',
                'max-width': '600px',
                'box-shadow': '0 20px 60px rgba(0, 0, 0, 0.2)',
                'text-align': 'center'
            }, children=[
                html.H1("⚠️ Application Error", style={'color': '#dc3545', 'margin-bottom': '20px'}),
                html.P("Could not load necessary data files.", style={'font-size': '18px', 'color': '#5a6c7d', 'margin-bottom': '20px'}),
                html.P("Please ensure the 'processed_data' folder contains:", style={'color': '#5a6c7d', 'margin-bottom': '15px'}),
                html.Ul([
                    html.Li("network_df.parquet", style={'text-align': 'left'}),
                    html.Li("market_volume_df.parquet", style={'text-align': 'left'}),
                    html.Li("trader_df.parquet", style={'text-align': 'left'}),
                    html.Li("roads_*_processed.geojson files", style={'text-align': 'left'}),
                    html.Li("nightlights_data.json", style={'text-align': 'left'})
                ], style={'max-width': '300px', 'margin': '0 auto', 'color': '#2c3e50'}),
                html.P("Check the application logs for details.", style={'margin-top': '20px', 'color': '#8795a1', 'font-size': '0.9rem'})
            ])
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
                return 'floating-controls fade-in', 'controls-content'
            else:
                return 'floating-controls icon-only', 'controls-content hidden'
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
            updated_style = current_style.copy()
            updated_style['display'] = 'block'
            return updated_style
        updated_style = current_style.copy()
        updated_style['display'] = 'none'
        return updated_style

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

        if 'show_nightlights' in layer_toggles and selected_time in nightlights_data:
            b64_img, coords = nightlights_data[selected_time]
            layers.append({"source": b64_img, "sourcetype": "image", "coordinates": coords, "opacity": 0.8, "below": ""})
        if 'show_roads' in layer_toggles and selected_time in roads_data:
            road_color = 'rgba(211, 211, 211, 0.6)' if 'show_nightlights' in layer_toggles else 'rgba(100, 100, 100, 0.7)'
            layers.append({'source': roads_data[selected_time], 'type': 'line', 'color': road_color, 'line': {'width': 0.8}, 'below': 'traces'})

        zoom, center = (5.5, {"lat": 0.5, "lon": 37.5})
        if relayout_data and 'map.center' in relayout_data:
            zoom = relayout_data['map.zoom']
            center = relayout_data['map.center']

        fig = go.Figure()
        
        if 'show_markers' in layer_toggles:
            opacity = opacity_percent / 100.0
            routes_visible = 'show' in toggle_value
            share_bins = [
                {'name': 'High Share (>75%)', 'data': df_map[df_map['share'] >= 75], 'width': 4, 'color': f'rgba(217, 95, 2, {opacity})'},
                {'name': 'Medium Share (25-75%)', 'data': df_map[(df_map['share'] < 75) & (df_map['share'] >= 25)], 'width': 2, 'color': f'rgba(117, 112, 179, {opacity})'},
                {'name': 'Low Share (<25%)', 'data': df_map[df_map['share'] < 25], 'width': 1, 'color': f'rgba(102, 166, 30, {opacity})'}
            ]
            for s_bin in share_bins:
                if not s_bin['data'].empty:
                    lats, lons = [item for _, row in s_bin['data'].iterrows() for item in (row['origin_lat'], row['market_lat'], None)], [item for _, row in s_bin['data'].iterrows() for item in (row['origin_lon'], row['market_lon'], None)]
                    fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode='lines', line=dict(width=s_bin['width'], color=s_bin['color']), name=s_bin['name'], hoverinfo='none', visible=routes_visible))

            if not df_map.empty:
                origins = df_map[['origin_name', 'origin_lat', 'origin_lon']].drop_duplicates().merge(df_map.groupby('origin_name', observed=True)['mkt_name'].nunique().reset_index(name='market_count'), on='origin_name')
                origins['hover_text'] = origins['origin_name'] + '<br>Supplies ' + origins['market_count'].astype(str) + ' market(s)'
                fig.add_trace(go.Scattermap(lat=origins['origin_lat'], lon=origins['origin_lon'], mode='markers', marker=dict(size=(5 + origins['market_count']), color='#a50f15', opacity=0.9), name='Produce Origin', text=origins['hover_text'], hoverinfo='text'))
                
                markets = df_map[['mkt_id', 'mkt_name', 'market_lat', 'market_lon', 'mkt_type']].drop_duplicates().merge(df_vol[['mkt_id', 'Total Volume']], on='mkt_id', how='left').fillna(0)
                market_hover_info = df_map.assign(origin_share_str=df_map['origin_name'].astype(str) + ': ' + df_map['share'].astype(int).astype(str) + '%').groupby('mkt_name', observed=True)['origin_share_str'].apply('<br>'.join).reset_index(name='details')
                markets = markets.merge(market_hover_info, on='mkt_name')
                markets['hover_text'] = '<b>' + markets['mkt_name'] + '</b><br><i>' + markets['mkt_type'] + '</i><br>' + 'Trade Quantity: ' + markets['Total Volume'].round(0).astype(int).apply(lambda x: f'{x:,}') + ' units<br>' + '--- Origins ---<br>' + markets['details']
                markets['size'] = 4 + (markets['Total Volume'] ** 0.5) * 0.08
                fig.add_trace(go.Scattermap(lat=markets['market_lat'], lon=markets['market_lon'], mode='markers', marker=dict(size=markets['size'], color='blue', opacity=0.9), name='Market', text=markets['hover_text'], hoverinfo='text'))
            elif 'show_markers' in layer_toggles:
                fig.add_annotation(text="No trade flow data for this selection.", showarrow=False, font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"))

        fig.add_trace(go.Scattermap(lat=[0], lon=[37.5], mode='markers', marker=dict(size=0.1, color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='none'))
        
        fig.update_layout(
            margin={"r":0, "t":0, "l":0, "b":0},
            showlegend=True,
            legend=dict(yanchor="top", y=0.92, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, traceorder='normal', itemsizing='constant', font=dict(size=11)),
            map=dict(style=map_style, layers=layers, zoom=zoom, center=center)
        )
        if 'show_roads' in layer_toggles:
            fig.add_trace(go.Scattermap(mode='lines', lon=[None], lat=[None], line=dict(color='rgba(100, 100, 100, 0.7)', width=2), name='Road Network'))
        return fig, map_title

    @app.callback(
        [Output('trader-control-div', 'className'),
         Output('season-control-div', 'className')],
        Input('data-type-toggle', 'value')
    )
    def update_combined_map_controls(data_type):
        if data_type == 'traders':
            return 'conditional-control', 'conditional-control hidden'
        else:
            return 'conditional-control hidden', 'conditional-control'

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
                if data_type == 'tomatoes':
                    df['size'] = 5 + (df[z_value_col] ** 0.5) * 0.1
                else: 
                    df['size'] = 5 + (df[z_value_col] ** 0.5) * 0.8
                
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
                    radius=heatmap_radius,
                    colorscale=colorscale,
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
            map=dict(style=map_style_light, zoom=zoom, center=center)
        )
        return fig, map_title

# --- 6. RUN THE APP ---
# For production on Render, we don't use app.run()
# Instead, Gunicorn will use the 'server' object
if __name__ == '__main__':
    app.run(debug=False)
