# filename: app.py
# --- FINAL VERSION FOR RENDER DEPLOYMENT ---
# This version uses pre-processed data for maximum efficiency.

import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json

# --- 1. APP INITIALIZATION & DATA LOADING (OPTIMIZED) ---

app = dash.Dash(__name__, assets_folder='assets')
server = app.server # Expose server for Gunicorn
app.title = "INCATA Market Analysis"

# Load the pre-processed data from the 'processed_data' folder.
# This is much faster and more memory-efficient than processing on the fly.
print("Loading pre-processed data...")
PROCESSED_DATA_FOLDER = Path(__file__).parent / "processed_data"

try:
    network_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'network_df.parquet')
    market_volume_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'market_volume_df.parquet')
    trader_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'trader_df.parquet')
    print("SUCCESS: Tabular data (.parquet) loaded.")

    roads_data = {}
    nightlights_data = {}
    time_period_map = {"10 Yrs Ago": "10_yrs_ago", "5 Yrs Ago": "5_yrs_ago", "Now": "now"}

    # Load pre-clipped and simplified roads
    for key, file_suffix in time_period_map.items():
        road_path = PROCESSED_DATA_FOLDER / f"roads_{file_suffix}_processed.geojson"
        if road_path.exists():
            roads_gdf = gpd.read_file(road_path)
            roads_data[key] = roads_gdf.__geo_interface__
    print("SUCCESS: Roads data (.geojson) loaded.")

    # Load pre-rendered nightlight images and coordinates from JSON
    nightlight_path = PROCESSED_DATA_FOLDER / "nightlights_data.json"
    if nightlight_path.exists():
        with open(nightlight_path, 'r') as f:
            nightlights_data = json.load(f)
        print("SUCCESS: Nightlights data (.json) loaded.")

except FileNotFoundError as e:
    print(f"---! FATAL ERROR !---")
    print(f"A required data file is missing: {e}")
    print("Please run the `preprocess_data.py` script locally and ensure the `processed_data` folder is uploaded to Render.")
    # Display an error message in the app layout if data is missing
    app.layout = html.Div([
        html.H1("Application Error"),
        html.P(f"Could not load necessary data file: {e}. Please contact the administrator.")
    ])
    # Prevent further execution if critical data is missing
else:
    print("All pre-processed data loaded successfully.")


    # --- 2. DASH APP LAYOUT ---
    section_style = {
        'background-color': '#f0f8ff', 'border': '1px solid #cce5ff', 'border-radius': '10px',
        'padding': '25px', 'box-shadow': '2px 2px 10px lightgrey', 'margin-bottom': '40px'
    }
    title_style = {'textAlign': 'center', 'color': '#333333', 'marginBottom': '20px'}
    map_style_light = "carto-positron"
    map_style_dark = "carto-darkmatter"

    app.layout = html.Div(style={'fontFamily': "'Segoe UI', 'Roboto', Arial, sans-serif", 'padding': '2% 5%', 'background-color': '#f8f9fa'}, children=[
        
        # --- HEADER (FROM YOUR OLD APP) ---
        html.Div([
            html.H1("INCATA Market Analysis Dashboard", style={'textAlign': 'center', 'color': '#004085'}),
            html.H4("A Public Dashboard for Markets Studied under Project INCATA", style={'textAlign': 'center', 'fontWeight': 'normal'}),
            html.H4("In Collaboration with RIMISP, Michigan State University, International Food Policy Research Institute and Tegemeo Institute", style={'textAlign': 'center', 'fontWeight': 'normal', 'fontSize':'1em'}),
            html.P("Linked Farms and Enterprises for Inclusive Agricultural Transformation in Africa and Asia", style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#6c757d', 'marginTop': '-10px'})
        ], style={'marginBottom': '40px'}),

        html.Div(style={'background-color': '#e2e3e5', 'padding': '15px', 'border-radius': '10px', 'margin-bottom': '60px'}, children=[
            html.Label("Global Filter: Select Market Type", style={'fontWeight': 'bold', 'display': 'block', 'color': '#495057', 'marginBottom': '10px'}),
            dcc.Dropdown(id='master-market-type-filter', options=[{'label': 'All Markets', 'value': 'All Markets'}] + [{'label': mtype, 'value': mtype} for mtype in sorted(network_df['mkt_type'].unique())], value='All Markets')
        ]),
        
        # --- NEW NETWORK MAP SECTION ---
        html.Div(style=section_style, children=[
            html.H2("Produce Flow Network", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
            html.P("This map shows the origin and flow of tomatoes. Use the toggles to show/hide roads, nightlights, and the market network. The origins of tomatoes' (Red dots) position are approximations only. Please be patient when selecting any option that will dynamically update the map, as this website is hosted on a free instance and may be slow at times.", style={'marginBottom': '20px'}), # <-- YOUR TEXT
            html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                html.Div(id={'type': 'floating-panel-wrapper', 'index': 'network'}, className='floating-controls', children=[
                    html.Div(id={'type': 'panel-header', 'index': 'network'}, className='control-panel-header', n_clicks=0, children=[
                        html.Span("⚙️", className='header-icon'),
                        html.Span("Map Controls", className='header-text'),
                        html.Button('−', className='toggle-btn')
                    ]),
                    html.Div(id={'type': 'panel-content', 'index': 'network'}, className='controls-content', children=[
                        # Controls Content Here
                        html.Label("📅 Time Period", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block'}),
                        dcc.Slider(id='network-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False),
                        html.Label("🌱 Season", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginTop': '20px'}),
                        dcc.RadioItems(id='season-toggle', options=[{'label': ' High', 'value': 'High Season'}, {'label': ' Low', 'value': 'Low Season'}], value='High Season', labelStyle={'display': 'inline-block', 'marginRight':'15px'}),
                        html.Label("🗺 Map Layers", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginTop': '20px'}),
                        dcc.Checklist(id='layer-toggles', options=[{'label': ' Markets & Origins', 'value': 'show_markers'}, {'label': ' Roads', 'value': 'show_roads'}, {'label': ' Nightlights', 'value': 'show_nightlights'}], value=['show_markers'], labelStyle={'display': 'block'}),
                        html.Label("🔗 Trade Routes", style={'fontWeight': 'bold', 'fontSize': '13px', 'display': 'block', 'marginTop': '20px'}),
                        dcc.Checklist(id='toggle-routes', options=[{'label': ' Show Trade Routes', 'value': 'show'}], value=['show']),
                    ]),
                ]),
                dcc.Graph(id='network-map', style={'height': '85vh', 'width': '100%'}, config={'scrollZoom': True}),
            ]),
        ]),
        
        # --- NEW COMBINED ANALYSIS MAP SECTION ---
        html.Div(style=section_style, children=[
            html.H2("Market Concentration Analysis", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
            html.P("Analyze tomato trade volume or trader concentration. Use the controls to switch data types and visualization styles.", style={'marginBottom': '20px'}),
            html.H3(id='combined-map-title', style=title_style),
            html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                html.Div(id={'type': 'floating-panel-wrapper', 'index': 'combined'}, className='floating-controls', children=[
                    html.Div(id={'type': 'panel-header', 'index': 'combined'}, className='control-panel-header', n_clicks=0, children=[
                        html.Span("⚙️", className='header-icon'),
                        html.Span("Analysis Controls", className='header-text'),
                        html.Button('−', className='toggle-btn')
                    ]),
                    html.Div(id={'type': 'panel-content', 'index': 'combined'}, className='controls-content', children=[
                        # Controls Content Here
                        html.Label("📊 Analysis Type", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                        dcc.RadioItems(id='data-type-toggle', options=[{'label': ' Tomato Volume', 'value': 'tomatoes'}, {'label': ' Traders', 'value': 'traders'}], value='tomatoes', labelStyle={'display': 'block'}),
                        html.Label("🎨 View Style", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginTop': '20px'}),
                        dcc.RadioItems(id='view-type-toggle', options=[{'label': ' Points', 'value': 'points'}, {'label': ' Heatmap', 'value': 'heatmap'}], value='points', labelStyle={'display': 'block'}),
                        html.Div(id='season-control-div', children=[
                            html.Label("🌱 Season", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginTop': '20px'}),
                            dcc.RadioItems(id='combined-season-toggle', options=[{'label': ' High', 'value': 'High Season'}, {'label': ' Low', 'value': 'Low Season'}], value='High Season', labelStyle={'display': 'inline-block', 'marginRight':'15px'}),
                        ]),
                        html.Div(id='trader-control-div', children=[
                            html.Label("👤 Trader Type", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginTop': '20px'}),
                            dcc.Dropdown(id='combined-trader-type-dropdown', options=[{'label': 'All Traders', 'value': 'All'}] + [{'label': t, 'value': t} for t in sorted(trader_df['trader_id'].unique()) if pd.notna(t)], value='All'),
                        ]),
                        html.Label("📅 Time Period", style={'fontWeight': 'bold', 'fontSize': '13px', 'marginTop': '20px'}),
                        dcc.Slider(id='combined-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False),
                    ]),
                ]),
                dcc.Graph(id='combined-map', style={'height': '85vh'}),
            ]),
        ]),
        
        # --- FOOTER (FROM YOUR OLD APP) ---
        html.Footer([
            html.P("The INCATA project is funded by the Gates Foundation.",
                   style={'color': '#6c757d', 'fontSize': '0.9em'})
        ], style={'textAlign': 'center', 'padding': '20px 0', 'marginTop': '40px', 'borderTop': '1px solid #dee2e6'})
    ])


    # --- 3. DASH CALLBACKS ---

    # Callback for floating control panels
    @app.callback(
        [Output({'type': 'floating-panel-wrapper', 'index': MATCH}, 'className'),
         Output({'type': 'panel-content', 'index': MATCH}, 'className')],
        Input({'type': 'panel-header', 'index': MATCH}, 'n_clicks'),
        State({'type': 'floating-panel-wrapper', 'index': MATCH}, 'className'),
        prevent_initial_call=True
    )
    def toggle_panel_animation(n, current_class):
        if n and n > 0:
            return ('floating-controls icon-only', 'controls-content hidden') if 'icon-only' not in current_class else ('floating-controls', 'controls-content')
        return no_update, no_update

    # Callback for the Network Map
    @app.callback(
        Output('network-map', 'figure'),
        [Input('master-market-type-filter', 'value'), Input('season-toggle', 'value'),
         Input('network-time-slider', 'value'), Input('toggle-routes', 'value'), 
         Input('layer-toggles', 'value')],
        [State('network-map', 'relayoutData')]
    )
    def update_network_map(selected_market_type, selected_season, time_value, toggle_value, layer_toggles, relayout_data):
        time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
        selected_time = time_map[time_value]
        layer_toggles = layer_toggles or []

        df_flow = network_df[(network_df['season'] == selected_season) & (network_df['Time Period'] == selected_time)]
        if selected_market_type != 'All Markets':
            df_flow = df_flow[df_flow['mkt_type'] == selected_market_type]
        df_map = df_flow[df_flow['share'] > 0].copy()
        
        map_style = map_style_dark if 'show_nightlights' in layer_toggles else map_style_light
        mapbox_layers = []

        if 'show_nightlights' in layer_toggles and selected_time in nightlights_data:
            b64_img, coords = nightlights_data[selected_time]
            mapbox_layers.append({"source": b64_img, "sourcetype": "image", "coordinates": coords, "opacity": 0.8, "below": ""})
        if 'show_roads' in layer_toggles and selected_time in roads_data:
            road_color = 'rgba(211, 211, 211, 0.6)' if 'show_nightlights' in layer_toggles else 'rgba(100, 100, 100, 0.7)'
            mapbox_layers.append({'source': roads_data[selected_time], 'type': 'line', 'color': road_color, 'line': {'width': 0.8}, 'below': 'traces'})

        zoom, center = (5.5, {"lat": 0.5, "lon": 37.5})
        if relayout_data and 'mapbox.center' in relayout_data:
            zoom = relayout_data['mapbox.zoom']
            center = relayout_data['mapbox.center']

        fig = go.Figure()
        
        if 'show_markers' in layer_toggles:
            routes_visible = 'show' in toggle_value
            if not df_map.empty:
                # Add Lines
                lats, lons = [item for _, row in df_map.iterrows() for item in (row['origin_lat'], row['market_lat'], None)], [item for _, row in df_map.iterrows() for item in (row['origin_lon'], row['market_lon'], None)]
                fig.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode='lines', line=dict(width=1, color='grey'), hoverinfo='none', visible=routes_visible, showlegend=False))
                
                # Add Origins
                origins = df_map[['origin_name', 'origin_lat', 'origin_lon']].drop_duplicates()
                fig.add_trace(go.Scattermapbox(lat=origins['origin_lat'], lon=origins['origin_lon'], mode='markers', marker=dict(size=10, color='#a50f15'), name='Produce Origin', text=origins['origin_name'], hoverinfo='text'))
                
                # Add Markets
                markets = df_map[['mkt_name', 'market_lat', 'market_lon']].drop_duplicates()
                fig.add_trace(go.Scattermapbox(lat=markets['market_lat'], lon=markets['market_lon'], mode='markers', marker=dict(size=12, color='blue'), name='Market', text=markets['mkt_name'], hoverinfo='text'))

        fig.update_layout(
            mapbox_style=map_style, mapbox_layers=mapbox_layers,
            mapbox_zoom=zoom, mapbox_center=center,
            margin={"r":0, "t":0, "l":0, "b":0}, showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

    # Callback to show/hide controls for the combined map
    @app.callback(
        [Output('trader-control-div', 'style'), Output('season-control-div', 'style')],
        Input('data-type-toggle', 'value')
    )
    def update_combined_map_controls(data_type):
        if data_type == 'traders':
            return {'display': 'block'}, {'display': 'none'}
        else: # tomatoes
            return {'display': 'none'}, {'display': 'block'}

    # Callback for the Combined "Market Concentration" Map
    @app.callback(
        [Output('combined-map', 'figure'), Output('combined-map-title', 'children')],
        [Input('master-market-type-filter', 'value'), Input('data-type-toggle', 'value'),
         Input('view-type-toggle', 'value'), Input('combined-trader-type-dropdown', 'value'),
         Input('combined-season-toggle', 'value'), Input('combined-time-slider', 'value')],
        [State('combined-map', 'relayoutData')]
    )
    def update_combined_map(market_type, data_type, view_type, selected_trader, selected_season, time_value, relayout_data):
        time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
        selected_time = time_map[time_value]
        
        fig = go.Figure()
        df, title_parts, z_value_col, colorscale, colorbar_title, heatmap_radius = (None,)*6

        if data_type == 'tomatoes':
            df = market_volume_df.copy()
            if market_type != 'All Markets': df = df[df['mkt_type'] == market_type]
            df = df[(df['season'] == selected_season) & (df['Time Period'] == selected_time) & (df['Total Volume'] > 0)]
            title_parts, z_value_col, colorscale, colorbar_title, heatmap_radius = ['Tomato Trade Volume', selected_season, selected_time], 'Total Volume', 'Viridis', 'Trade Volume', 20
        else: # traders
            df = trader_df.copy()
            if market_type != 'All Markets': df = df[df['mkt_type'] == market_type]
            if selected_trader != 'All': df = df[df['trader_id'] == selected_trader]
            df = df.groupby(['mkt_name', 'lat', 'lon'], as_index=False)[selected_time].sum()
            df = df[df[selected_time] > 0]
            title_parts, z_value_col, colorscale, colorbar_title, heatmap_radius = [f'{selected_trader} Traders', selected_time], selected_time, 'Plasma', 'No. of Traders', 30
        
        map_title = ' - '.join(title_parts)

        if df.empty:
            fig.add_annotation(text="No data available for this selection.", showarrow=False)
        else:
            df['hover_text'] = '<b>' + df['mkt_name'] + '</b><br>' + colorbar_title + ': ' + df[z_value_col].round(0).astype(int).apply(lambda x: f'{x:,}')
            
            if view_type == 'points':
                df['size'] = 5 + (df[z_value_col] ** 0.5) * (0.1 if data_type == 'tomatoes' else 0.8)
                fig.add_trace(go.Scattermapbox(lat=df['lat'], lon=df['lon'], mode='markers', marker=dict(size=df['size'], color=df[z_value_col], colorscale=colorscale, cmin=0, cmax=df[z_value_col].quantile(0.95), showscale=True, colorbar_title_text=colorbar_title), text=df['hover_text'], hoverinfo='text'))
            else: # heatmap
                fig.add_trace(go.Densitymapbox(lat=df['lat'], lon=df['lon'], z=df[z_value_col], radius=heatmap_radius, colorscale=colorscale, colorbar_title_text=colorbar_title))
                fig.add_trace(go.Scattermapbox(lat=df['lat'], lon=df['lon'], mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'), text=df['hover_text'], hoverinfo='text', showlegend=False))

        zoom, center = (5.5, {"lat": 0.5, "lon": 37.5})
        if relayout_data and 'mapbox.center' in relayout_data:
            zoom = relayout_data['mapbox.zoom']
            center = relayout_data['mapbox.center']

        fig.update_layout(
            margin={"r":0, "t":0, "l":0, "b":0},
            mapbox_style=map_style_light, mapbox_zoom=zoom, mapbox_center=center
        )
        return fig, map_title

# --- 4. RUN THE APP ---
if __name__ == '__main__':
    # When running locally, use debug=True for hot-reloading
    # For production on Render, Gunicorn will run the app, and debug should be False
    app.run(debug=False)
