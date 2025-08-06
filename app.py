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

# --- 3. STYLES ---
section_style = {
    'background-color': '#f0f8ff', 'border': '1px solid #cce5ff', 'border-radius': '10px',
    'padding': '25px', 'box-shadow': '2px 2px 10px lightgrey', 'margin-bottom': '40px'
}
title_style = {'textAlign': 'center', 'color': '#333333', 'marginBottom': '20px'}
map_style_light = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
map_style_dark = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# --- 4. APP LAYOUT ---
if data_load_success:
    app.layout = html.Div(style={'fontFamily': "'Segoe UI', 'Roboto', Arial, sans-serif", 'padding': '2% 5%', 'background-color': '#f8f9fa'}, children=[
        
        html.Div(id='menu-portal-target'),
        html.Div([
            html.H1("INCATA Market Analysis Dashboard", style={'textAlign': 'center', 'color': '#004085'}),
            html.H4("A Public Dashboard for Markets Studied under Project INCATA", style={'textAlign': 'center', 'fontWeight': 'normal'}),
            html.P("In Collaboration with RIMISP, Michigan State University, IFPRI, and Tegemeo Institute", 
                   style={'textAlign': 'center', 'fontSize': '14px', 'color': '#495057', 'marginTop': '-5px', 'marginBottom': '5px'}),
            html.P("INCATA: Linked Farms and Enterprises for Inclusive Agricultural Transformation in Africa and Asia", 
                   style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#6c757d', 'fontSize': '13px'})
        ], style={'marginBottom': '40px'}),
        html.Div(style={'background-color': '#e2e3e5', 'padding': '15px', 'border-radius': '10px', 'margin-bottom': '60px'}, children=[
            html.Label("Global Filter: Select Market Type", style={'fontWeight': 'bold', 'display': 'block', 'color': '#495057', 'marginBottom': '10px'}),
            dcc.Dropdown(id='master-market-type-filter', options=[{'label': 'All Markets', 'value': 'All Markets'}] + [{'label': mtype, 'value': mtype} for mtype in sorted(network_df['mkt_type'].unique())], value='All Markets')
        ]),
        
        # NETWORK MAP SECTION
        html.Div(style=section_style, children=[
            html.H2("Produce Flow Network", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
            html.P("This map shows the origin and flow of tomatoes. Use the controls panel to adjust visualization settings. The map position of the Origins of Produce (Red dots) is an approximation only, as in many cases they refer to somewhere within that county.", style={'marginBottom': '20px'}),
            html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                html.Div(id={'type': 'floating-panel-wrapper', 'index': 'network'}, className='floating-controls', children=[
                    html.Div(id={'type': 'panel-header', 'index': 'network'}, className='control-panel-header', n_clicks=0, children=[
                        html.Span("⚙️", className='header-icon'),
                        html.Span("Map Controls", className='header-text'),
                        html.Button('−', id='network-toggle-controls-btn', className='toggle-btn')
                    ]),
                    html.Div(id={'type': 'panel-content', 'index': 'network'}, className='controls-content', children=[
                        html.Div(style={'marginBottom': '20px'}, children=[
                            html.Label("📅 Time Period", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Slider(id='network-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False),
                        ]),
                        html.Div(style={'marginBottom': '20px'}, children=[
                            html.Label("🌱 Season", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.RadioItems(id='season-toggle', options=[{'label': ' High Season', 'value': 'High Season'}, {'label': ' Low Season', 'value': 'Low Season'}], value='High Season', labelStyle={'display': 'block', 'marginBottom': '5px', 'fontSize': '12px'}),
                        ]),
                        html.Div(style={'marginBottom': '20px'}, children=[
                            html.Label("🗺 Map Layers", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Checklist(id='layer-toggles', options=[{'label': ' Markets & Origins', 'value': 'show_markers'}, {'label': ' Roads', 'value': 'show_roads'}, {'label': ' Nightlights', 'value': 'show_nightlights'}], value=['show_markers'], labelStyle={'display': 'block', 'marginBottom': '5px', 'fontSize': '12px'}),
                        ]),
                        html.Div(style={'marginBottom': '15px'}, children=[
                            html.Label("🔗 Trade Routes", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Checklist(id='toggle-routes', options=[{'label': ' Show Trade Routes', 'value': 'show'}], value=['show'], labelStyle={'fontSize': '12px', 'marginBottom': '8px'}),
                            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginTop': '10px'}, children=[
                                html.Label("Opacity:", style={'fontSize': '12px', 'minWidth': '50px'}),
                                dcc.Dropdown(id='opacity-dropdown', options=[{'label': f'{i}%', 'value': i} for i in range(0, 101, 10)], value=70, clearable=False, searchable=False, style={'width': '80px', 'fontSize': '11px'}),
                            ]),
                        ]),
                        html.Div(style={'borderTop': '1px solid #e0e0e0', 'paddingTop': '10px', 'marginTop': '10px'}, children=[
                            html.Button('ℹ How to Read This Map', id='network-info-button', n_clicks=0, style={'width': '100%', 'cursor': 'pointer', 'border': '1px solid #004085', 'backgroundColor': '#e7f3ff', 'padding': '5px 10px', 'borderRadius': '5px', 'fontSize': '12px'}),
                        ]),
                    ]),
                ]),
                html.Div(id='network-map-title', style={'position': 'absolute', 'top': '10px', 'left': '50%', 'transform': 'translateX(-50%)', 'background-color': 'rgba(255, 255, 255, 0.95)', 'padding': '8px 20px', 'border-radius': '20px', 'font-size': '14px', 'font-weight': 'bold', 'box-shadow': '0 2px 8px rgba(0,0,0,0.15)', 'z-index': '999', 'color': '#004085', 'border': '1px solid rgba(0, 64, 133, 0.2)'}),
                dcc.Graph(id='network-map', style={'height': '85vh', 'width': '100%'}, config={'scrollZoom': True}),
                html.Div(id='network-info-collapse', style={'display': 'none', 'position': 'absolute', 'bottom': '10px', 'right': '10px', 'background-color': 'rgba(248, 249, 250, 0.95)', 'padding': '15px', 'border': '1px dashed #cce5ff', 'borderRadius': '5px', 'z-index': '998', 'max-width': '400px'}, children=[
                    html.Button('✕', id='close-info-btn', n_clicks=0, style={'position': 'absolute', 'top': '5px', 'right': '10px', 'background': 'transparent', 'border': 'none', 'fontSize': '18px', 'cursor': 'pointer'}),
                    dcc.Markdown('''* **Red Dots (Produce Origins):** Counties or areas where tomatoes are sourced.\n* **Blue Dots (Markets):** Markets where tomatoes are sold.\n* **Lines (Trade Routes):** Connections from origin to market.\n* **Line Thickness:** Represents the share of produce from that origin.''', style={'fontSize': '12px', 'margin': '0'})
                ])
            ]),
        ]),
        
        # COMBINED ANALYSIS MAP SECTION
        html.Div(style=section_style, children=[
            html.H2("Market Concentration Analysis", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
            html.P("Analyze tomato trade volume or trader concentration. Use the controls to switch data types and visualization styles.", style={'marginBottom': '20px'}),
            html.H3(id='combined-map-title', style=title_style),
            html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                html.Div(id={'type': 'floating-panel-wrapper', 'index': 'combined'}, className='floating-controls', children=[
                    html.Div(id={'type': 'panel-header', 'index': 'combined'}, className='control-panel-header', n_clicks=0, children=[
                        html.Span("⚙️", className='header-icon'),
                        html.Span("Analysis Controls", className='header-text'),
                        html.Button('−', id='combined-toggle-controls-btn', className='toggle-btn')
                    ]),
                    html.Div(id={'type': 'panel-content', 'index': 'combined'}, className='controls-content', children=[
                        html.Div([
                            html.Label("📊 Analysis Type", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginBottom': '10px'}),
                            dcc.RadioItems(id='data-type-toggle', options=[{'label': ' Tomatoes', 'value': 'tomatoes'}, {'label': ' Traders', 'value': 'traders'}], value='tomatoes', inline=True, labelStyle={'marginRight': '15px'}),
                        ], style={'marginBottom': '25px'}),
                        html.Div([
                            html.Label("🎨 View Style", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginBottom': '10px'}),
                            dcc.RadioItems(id='view-type-toggle', options=[{'label': ' Points', 'value': 'points'}, {'label': ' Heatmap', 'value': 'heatmap'}], value='points', inline=True, labelStyle={'marginRight': '15px'}),
                        ], style={'marginBottom': '25px'}),
                        html.Div(id='season-control-div', className='conditional-control', children=[
                            html.Label("🌱 Season", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginBottom': '10px'}),
                            dcc.RadioItems(id='combined-season-toggle', options=[{'label': ' High', 'value': 'High Season'}, {'label': ' Low', 'value': 'Low Season'}], value='High Season', inline=True),
                        ], style={'marginBottom': '25px'}),
                        html.Div(id='trader-control-div', className='conditional-control', children=[
                            html.Label("👤 Trader Type", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginBottom': '10px'}),
                            dcc.Dropdown(id='combined-trader-type-dropdown', options=[{'label': 'All Traders', 'value': 'All'}] + [{'label': t, 'value': t} for t in sorted(trader_df['trader_id'].unique()) if pd.notna(t)], value='All', placeholder="Select..."),
                        ], style={'marginBottom': '25px'}),
                        html.Div([
                            html.Label("📅 Time Period", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginBottom': '8px'}),
                            dcc.Slider(id='combined-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False),
                        ], style={'marginBottom': '10px'}),
                    ]),
                ]),
                dcc.Graph(id='combined-map', style={'height': '85vh'}),
            ]),
        ]),
        
        html.Footer([
            html.P("The INCATA project is funded by the Gates Foundation.",
                   style={'color': '#6c757d', 'fontSize': '0.9em'}),
        ], style={'textAlign': 'center', 'padding': '20px 0', 'marginTop': '40px', 'borderTop': '1px solid #dee2e6'})
    ])
else:
    # Error layout if data fails to load
    app.layout = html.Div([
        html.H1("INCATA Dashboard - Application Error", style={'textAlign': 'center', 'color': '#dc3545', 'marginTop': '50px'}),
        html.P("Could not load necessary data files.", style={'textAlign': 'center', 'fontSize': '18px'}),
        html.P("Please ensure the 'processed_data' folder is uploaded to Render with all required files:", 
               style={'textAlign': 'center'}),
        html.Ul([
            html.Li("network_df.parquet"),
            html.Li("market_volume_df.parquet"),
            html.Li("trader_df.parquet"),
            html.Li("roads_*_processed.geojson files"),
            html.Li("nightlights_data.json")
        ], style={'maxWidth': '400px', 'margin': '0 auto'}),
        html.P("Check the application logs for more details.", style={'textAlign': 'center', 'marginTop': '20px'})
    ], style={'fontFamily': "'Segoe UI', 'Roboto', Arial, sans-serif", 'padding': '2% 5%'})

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
                return 'floating-controls', 'controls-content'
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
