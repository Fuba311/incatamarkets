# filename: app.py
# --- FINAL VERSION FOR RENDER DEPLOYMENT WITH REDIS CACHING ---

import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from PIL import Image
import io
import base64
from shapely.geometry import Polygon
import os
from flask_caching import Cache

# --- App and Cache Initialization ---
app = dash.Dash(__name__)
server = app.server # Expose server for Gunicorn

# Configure Flask-Caching to use Redis
# It reads the REDIS_URL from the environment variable you set on Render.
cache = Cache(app.server, config={
    'CACHE_TYPE': 'RedisCache',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
})

# --- 1. DATA LOADING AND PREPARATION ---

# Set up a relative path to the 'data' directory
DATA_FOLDER = Path(__file__).parent / "data"

# Define file paths relative to the data folder
PATH_ID = DATA_FOLDER / "Module A0_Identifiers.dta"
PATH_TOMATO_HIGH = DATA_FOLDER / "Module B1_Tomato seasonality_high.dta"
PATH_TOMATO_LOW = DATA_FOLDER / "Module B1_Tomato seasonality_low.dta"
PATH_VEHICLE = DATA_FOLDER / "Module B3_Tomato vehicles.dta"
PATH_TRADER = DATA_FOLDER / "Module D1_Trader composition.dta"

@cache.memoize(timeout=7200) # Caches result for 2 hours
def load_stata_manual(path, categorical_cols_map):
    """Loads a Stata file and manually applies value labels."""
    print(f"CACHE MISS: Loading and processing {path}...")
    try:
        reader = pd.read_stata(path, iterator=True)
        df = reader.read(convert_categoricals=False)
        labels = reader.value_labels()
        for col, label_name in categorical_cols_map.items():
            if col in df.columns and label_name in labels:
                df[col] = df[col].map(labels[label_name])
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()

@cache.memoize(timeout=7200) # Caches result for 2 hours
def prepare_all_data():
    """A single function to load and process all dataframes, whose result will be cached."""
    print("CACHE MISS: Preparing all data from scratch...")
    
    df_id = load_stata_manual(PATH_ID, {'mkt_name': 'mkt_name111', 'county_id': 'a91', 'mkt_type': 'market_type111'})
    df_tomato_high_raw = load_stata_manual(PATH_TOMATO_HIGH, {'tomhigh_id': 'tomhigh_id111'})
    df_tomato_low_raw = load_stata_manual(PATH_TOMATO_LOW, {'tomlow_id': 'tomhigh_id111'})

    df_tomato_high = df_tomato_high_raw[['mkt_id', 'tomhigh_id', 'b103', 'b104', 'b105']].copy()
    df_tomato_high.rename(columns={'tomhigh_id': 'origin_name', 'b103': 'Now', 'b104': '5 Yrs Ago', 'b105': '10 Yrs Ago'}, inplace=True)
    df_tomato_high['season'] = 'High Season'
    df_tomato_low = df_tomato_low_raw[['mkt_id', 'tomlow_id', 'b107', 'b108', 'b109']].copy()
    df_tomato_low.rename(columns={'tomlow_id': 'origin_name', 'b107': 'Now', 'b108': '5 Yrs Ago', 'b109': '10 Yrs Ago'}, inplace=True)
    df_tomato_low['season'] = 'Low Season'
    df_tomato_combined = pd.concat([df_tomato_high, df_tomato_low], ignore_index=True)
    network_df_melted = df_tomato_combined.melt(id_vars=['mkt_id', 'origin_name', 'season'], value_vars=['Now', '5 Yrs Ago', '10 Yrs Ago'], var_name='Time Period', value_name='share')
    network_df = pd.merge(df_id, network_df_melted, on='mkt_id', how='inner')
    network_df.rename(columns={'n101latitude': 'market_lat', 'n101longitude': 'market_lon'}, inplace=True)
    network_df['share'] = network_df['share'].fillna(0)
    network_df.dropna(subset=['market_lat', 'market_lon', 'origin_name'], inplace=True)
    network_df['Time Period'] = pd.Categorical(network_df['Time Period'], ["10 Yrs Ago", "5 Yrs Ago", "Now"])
    origin_coords = {'Baringo': {'lat': 0.6277, 'lon': 35.9619}, 'Bomet': {'lat': -0.7833, 'lon': 35.3500}, 'Bungoma': {'lat': 0.5639, 'lon': 34.5617}, 'Busia': {'lat': 0.4607, 'lon': 34.1105}, 'Embu': {'lat': -0.5333, 'lon': 37.4500}, 'Homa Bay': {'lat': -0.5167, 'lon': 34.4500}, 'Isiolo': {'lat': 0.3528, 'lon': 37.5847}, 'Kajiado': {'lat': -1.8500, 'lon': 36.7833}, 'Kakamega': {'lat': 0.2833, 'lon': 34.7500}, 'Kiambu': {'lat': -1.1667, 'lon': 36.8333}, 'Kilifi': {'lat': -3.6333, 'lon': 39.8500}, 'Kirinyaga': {'lat': -0.5000, 'lon': 37.2667}, 'Kisii': {'lat': -0.6833, 'lon': 34.7667}, 'Kisumu': {'lat': -0.1000, 'lon': 34.7500}, 'Kitui': {'lat': -1.3667, 'lon': 38.0167}, 'Laikipia': {'lat': 0.3333, 'lon': 36.7500}, 'Machakos': {'lat': -1.5167, 'lon': 37.2667}, 'Makueni': {'lat': -1.8000, 'lon': 37.6167}, 'Meru': {'lat': 0.0500, 'lon': 37.6500}, 'Migori': {'lat': -1.0634, 'lon': 34.4731}, 'Muranga': {'lat': -0.7167, 'lon': 37.1500}, 'Nakuru': {'lat': -0.3031, 'lon': 36.0667}, 'Nandi': {'lat': 0.3333, 'lon': 35.1667}, 'Narok': {'lat': -1.0833, 'lon': 35.8667}, 'Nyahururu': {'lat': 0.0472, 'lon': 36.3683}, 'Nyamira': {'lat': -0.5667, 'lon': 35.0000}, 'Nyandarua': {'lat': -0.3333, 'lon': 36.5000}, 'Nyeri': {'lat': -0.4167, 'lon': 36.9500}, 'Samburu': {'lat': 1.2833, 'lon': 36.8333}, 'Siaya': {'lat': 0.0667, 'lon': 34.2833}, 'Taita-Taveta': {'lat': -3.4000, 'lon': 38.3333}, 'Tana River': {'lat': -1.5000, 'lon': 40.0000}, 'Tharaka-Nithi': {'lat': -0.2167, 'lon': 37.9500}, 'Trans-Nzoia': {'lat': 1.0167, 'lon': 35.0000}, 'Uasin Gishu': {'lat': 0.5167, 'lon': 35.2833}, 'West Pokot': {'lat': 1.6667, 'lon': 35.5000}, 'Uganda': {'lat': 1.0820, 'lon': 34.1759}, 'Tanzania': {'lat': -1.35, 'lon': 34.38}}
    network_df['origin_lat'] = network_df['origin_name'].map(lambda x: origin_coords.get(str(x), {}).get('lat'))
    network_df['origin_lon'] = network_df['origin_name'].map(lambda x: origin_coords.get(str(x), {}).get('lon'))
    network_df.dropna(subset=['origin_lat', 'origin_lon'], inplace=True)

    df_vehicle = load_stata_manual(PATH_VEHICLE, {'tomaveh_id': 'tomaveh_id111'})
    df_vehicle.fillna(0, inplace=True)
    df_vehicle['vol_high_now'] = df_vehicle['b301'] * df_vehicle['b300']
    df_vehicle['vol_high_5y'] = df_vehicle['b301'] * df_vehicle['b302']
    df_vehicle['vol_high_10y'] = df_vehicle['b301'] * df_vehicle['b303']
    df_vehicle['vol_low_now'] = df_vehicle['b304_1'] * df_vehicle['b304']
    df_vehicle['vol_low_5y'] = df_vehicle['b304_1'] * df_vehicle['b305']
    df_vehicle['vol_low_10y'] = df_vehicle['b304_1'] * df_vehicle['b306']
    agg_vol = df_vehicle.groupby('mkt_id').agg({'vol_high_now': 'sum', 'vol_high_5y': 'sum', 'vol_high_10y': 'sum', 'vol_low_now': 'sum', 'vol_low_5y': 'sum', 'vol_low_10y': 'sum'}).reset_index()
    df_vol_high = agg_vol[['mkt_id', 'vol_high_now', 'vol_high_5y', 'vol_high_10y']]
    df_vol_high.columns = ['mkt_id', 'Now', '5 Yrs Ago', '10 Yrs Ago']
    df_vol_high = df_vol_high.melt(id_vars='mkt_id', var_name='Time Period', value_name='Total Volume')
    df_vol_high['season'] = 'High Season'
    df_vol_low = agg_vol[['mkt_id', 'vol_low_now', 'vol_low_5y', 'vol_low_10y']]
    df_vol_low.columns = ['mkt_id', 'Now', '5 Yrs Ago', '10 Yrs Ago']
    df_vol_low = df_vol_low.melt(id_vars='mkt_id', var_name='Time Period', value_name='Total Volume')
    df_vol_low['season'] = 'Low Season'
    market_volume_df = pd.concat([df_vol_high, df_vol_low], ignore_index=True)
    market_volume_df = pd.merge(market_volume_df, df_id, on='mkt_id', how='left')
    market_volume_df.rename(columns={'n101latitude': 'lat', 'n101longitude': 'lon'}, inplace=True)

    df_id_trader = df_id[['mkt_id', 'mkt_name', 'mkt_type', 'county_id', 'n101latitude', 'n101longitude']]
    df_trader_raw = load_stata_manual(PATH_TRADER, {'trader_id': 'trader_id111'})
    df_trader_selected_cols = df_trader_raw[['mkt_id', 'trader_id', 'd105', 'd107', 'd109']].copy()
    trader_df = pd.merge(df_id_trader, df_trader_selected_cols, on='mkt_id', how='inner')
    trader_df.rename(columns={'n101latitude': 'lat', 'n101longitude': 'lon', 'd105': 'Now', 'd107': '5 Yrs Ago', 'd109': '10 Yrs Ago'}, inplace=True)
    trader_df[['Now', '5 Yrs Ago', '10 Yrs Ago']] = trader_df[['Now', '5 Yrs Ago', '10 Yrs Ago']].fillna(0)
    county_to_region = {'Mombasa': 'Coast', 'Kwale': 'Coast', 'Kilifi': 'Coast', 'Tana River': 'Coast', 'Lamu': 'Coast', 'Taita-Taveta': 'Coast', 'Garissa': 'North Eastern', 'Wajir': 'North Eastern', 'Mandera': 'North Eastern', 'Marsabit': 'Eastern', 'Isiolo': 'Eastern', 'Meru': 'Eastern', 'Tharaka-Nithi': 'Eastern', 'Embu': 'Eastern', 'Kitui': 'Eastern', 'Machakos': 'Eastern', 'Makueni': 'Eastern', 'Nyandarua': 'Central', 'Nyeri': 'Central', 'Kirinyaga': 'Central', 'Muranga': 'Central', 'Kiambu': 'Central', 'Turkana': 'Rift Valley', 'West Pokot': 'Rift Valley', 'Samburu': 'Rift Valley', 'Trans-Nzoia': 'Rift Valley', 'Uasin Gishu': 'Rift Valley', 'Elgeyo-Marakwet': 'Rift Valley', 'Nandi': 'Rift Valley', 'Baringo': 'Rift Valley', 'Laikipia': 'Rift Valley', 'Nakuru': 'Rift Valley', 'Narok': 'Rift Valley', 'Kajiado': 'Rift Valley', 'Kericho': 'Rift Valley', 'Bomet': 'Rift Valley', 'Kakamega': 'Western', 'Vihiga': 'Western', 'Bungoma': 'Western', 'Busia': 'Western', 'Siaya': 'Nyanza', 'Kisumu': 'Nyanza', 'Homa Bay': 'Nyanza', 'Migori': 'Nyanza', 'Kisii': 'Nyanza', 'Nyamira': 'Nyanza', 'Nairobi city': 'Nairobi'}
    trader_df['region'] = trader_df['county_id'].map(county_to_region)
    trends_base_df = trader_df.melt(id_vars=['county_id', 'region', 'mkt_type', 'trader_id'], value_vars=['Now', '5 Yrs Ago', '10 Yrs Ago'], var_name='Time Period', value_name='Number of Traders')
    trends_base_df['Time Period'] = pd.Categorical(trends_base_df['Time Period'], ["10 Yrs Ago", "5 Yrs Ago", "Now"])
    
    return network_df, market_volume_df, trader_df, trends_base_df

@cache.memoize(timeout=7200) # Caches result for 2 hours
def load_geospatial_data():
    """Loads and processes all geospatial files, whose result will be cached."""
    print("CACHE MISS: Loading and processing geospatial data...")
    time_period_map = {"10 Yrs Ago": "10_yrs_ago", "5 Yrs Ago": "5_yrs_ago", "Now": "now"}
    roads_data = {}
    nightlights_data = {}

    kenya_coords = [(33.89, -4.68), (34.07, -4.68), (35.03, -4.63), (36.08, -4.45), (37.7, -3.99), (37.77, -3.68), (38.23, -3.65), (38.74, -3.84), (39.2, -4.68), (39.6, -4.35), (40.12, -4.27), (40.31, -3.5), (40.98, -2.5), (41.58, -1.68), (41.91, -1.57), (41.86, -1.1), (41.91, -0.8), (41.58, 0.2), (41.58, 0.6), (41.42, 1.11), (41.28, 1.74), (40.98, 2.78), (41.86, 3.92), (41.17, 3.92), (40.77, 4.25), (40.0, 4.25), (39.86, 3.84), (39.56, 4.11), (39.2, 4.48), (38.77, 4.42), (38.43, 3.59), (38.12, 3.6), (37.94, 4.02), (37.04, 4.37), (36.84, 4.45), (36.16, 4.45), (35.82, 4.78), (35.82, 5.0), (35.3, 5.0), (34.67, 4.77), (34.48, 3.56), (34.0, 2.5), (33.89, 1.0), (33.89, -4.68)]
    kenya_polygon = Polygon(kenya_coords)
    clip_mask = gpd.GeoDataFrame([1], geometry=[kenya_polygon], crs="EPSG:4326")

    for key, file_suffix in time_period_map.items():
        road_path = DATA_FOLDER / f"roads_{file_suffix}.geojson"
        if road_path.exists():
            roads_gdf = gpd.read_file(road_path)
            if roads_gdf.crs is None: roads_gdf.set_crs("EPSG:4326", inplace=True)
            roads_gdf = roads_gdf.to_crs(clip_mask.crs)
            roads_clipped = gpd.clip(roads_gdf, clip_mask)
            if not roads_clipped.empty:
                roads_clipped.geometry = roads_clipped.geometry.simplify(tolerance=0.01)
                roads_data[key] = roads_clipped.__geo_interface__
        else:
            print(f"Warning: Road file not found at {road_path}")

        nightlight_path = DATA_FOLDER / f"nightlights_{file_suffix}.tif"
        if nightlight_path.exists():
            with rasterio.open(nightlight_path) as src:
                bounds = src.bounds
                coordinates = [[bounds.left, bounds.top], [bounds.right, bounds.top], [bounds.right, bounds.bottom], [bounds.left, bounds.bottom]]
                data = src.read(1)
                vmax = np.percentile(data[data > 0], 98) if (data > 0).any() else 1
                data_norm = np.clip(data / vmax, 0, 1) * 255
                data_norm = data_norm.astype(np.uint8)
                img = Image.fromarray(data_norm).convert("RGBA")
                alpha_data = np.where(data > 1, 200, 0).astype(np.uint8)
                alpha = Image.fromarray(alpha_data)
                img.putalpha(alpha)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                nightlights_data[key] = (f"data:image/png;base64,{encoded_image}", coordinates)
        else:
            print(f"Warning: Nightlight file not found at {nightlight_path}")

    return roads_data, nightlights_data

# --- Load all data on app start ---
network_df, market_volume_df, trader_df, trends_base_df = prepare_all_data()
roads_data, nightlights_data = load_geospatial_data()
print("All data loaded and ready.")


# --- 2. DASH APP LAYOUT ---
app.title = "Kenya Market Analysis"

section_style = {'background-color': '#f0f8ff', 'border': '1px solid #cce5ff', 'border-radius': '10px', 'padding': '25px', 'box-shadow': '2px 2px 10px lightgrey', 'margin-bottom': '40px'}
title_style = {'textAlign': 'center', 'color': '#333333', 'marginBottom': '20px'}

app.layout = html.Div(style={'fontFamily': "'Segoe UI', 'Roboto', Arial, sans-serif", 'padding': '2% 5%', 'background-color': '#f8f9fa'}, children=[
    
    html.Div([
        html.H1("Tegemeo Market Analysis Dashboard", style={'textAlign': 'center', 'color': '#004085'}),
        html.H4("A Public Dashboard for Markets Studied under Project INCATA", style={'textAlign': 'center', 'fontWeight': 'normal'}),
        html.P("Linked Farms and Enterprises for Inclusive Agricultural Transformation in Africa and Asia", style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#6c757d', 'marginTop': '-10px'})
    ], style={'marginBottom': '40px'}),

    html.Div(style={'background-color': '#e2e3e5', 'padding': '15px', 'border-radius': '10px', 'margin-bottom': '60px'}, children=[
        html.Label("Global Filter: Select Market Type", style={'fontWeight': 'bold', 'display': 'block', 'color': '#495057', 'marginBottom': '10px'}),
        dcc.Dropdown(id='master-market-type-filter', options=[{'label': 'All Markets', 'value': 'All Markets'}] + [{'label': mtype, 'value': mtype} for mtype in sorted(network_df['mkt_type'].unique())], value='All Markets')
    ]),
    
    html.Div(style=section_style, children=[
        html.H2("Produce Flow Network", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
        html.P("This map shows the origin and flow of tomatoes. Use the toggles to show/hide roads, nightlights, and the market network. The origins of tomatoes' (Red dots) position are approximations only. Please be patient when selecting any option that will dynamically update the map, as this website is hosted on a free instance and may be slow at times. ", style={'marginBottom': '20px'}),
        html.Div(style={'display': 'flex', 'flex-direction': 'column', 'gap': '25px'}, children=[
            html.Div(style={'display': 'flex', 'gap': '30px', 'align-items': 'center', 'flex-wrap': 'wrap'}, children=[
                html.Div(children=[
                    html.Label("Select Season:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(id='season-toggle', options=[{'label': 'High', 'value': 'High Season'}, {'label': 'Low', 'value': 'Low Season'}], value='High Season', inline=True, labelStyle={'margin-right': '15px'}),
                ]),
                html.Div(style={'flex': 1, 'min-width': '300px'}, children=[
                    html.Label("Select Time Period:", style={'fontWeight': 'bold'}),
                    dcc.Slider(id='network-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None),
                ]),
            ]),
            html.Div(style={'display':'flex', 'gap':'20px', 'align-items':'center', 'flex-wrap': 'wrap'}, children=[
                dcc.Checklist(id='toggle-routes', options=[{'label': 'Show Trade Routes', 'value': 'show'}], value=['show'], style={'fontWeight': 'bold', 'marginRight': '20px'}),
                html.Div(style={'display': 'flex', 'align-items': 'center', 'gap': '15px', 'flex': '1', 'min-width': '400px'}, children=[
                    html.Label("Route Opacity:", style={'fontWeight': 'bold', 'whiteSpace': 'nowrap'}),
                    html.Div(style={'flex': '1', 'min-width': '250px', 'position': 'relative', 'padding': '0 10px'}, children=[
                        dcc.Slider(id='opacity-slider', min=0, max=100, step=10, value=70, marks={i: str(i) for i in range(0, 101, 20)}, tooltip={"placement": "bottom", "always_visible": False}, updatemode='drag'),
                    ]),
                    dcc.Input(id='opacity-input', type='number', min=0, max=100, step=10, value=70, style={'width': '70px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                ]),
            ]),
            html.Div(style={'paddingTop': '15px', 'borderTop': '1px solid #ddd', 'marginTop': '15px', 'marginBottom': '25px'}, children=[
                html.Label("Map Overlays:", style={'fontWeight': 'bold', 'marginRight': '15px'}),
                dcc.Checklist(id='layer-toggles',
                              options=[
                                  {'label': 'Show Markets & Origins', 'value': 'show_markers'},
                                  {'label': 'Show Roads', 'value': 'show_roads'},
                                  {'label': 'Show Nightlights', 'value': 'show_nightlights'},
                              ],
                              value=['show_markers'],
                              inline=True, labelStyle={'margin-right': '20px'})
            ]),
        ]),
        html.Div(style={'marginTop': '20px'}, children=[
            html.Button('How to Read This Map', id='network-info-button', n_clicks=0, style={'marginBottom': '10px', 'cursor': 'pointer', 'border': '1px solid #004085', 'backgroundColor': '#e7f3ff', 'padding': '5px 10px', 'borderRadius': '5px'}),
            html.Div(id='network-info-collapse', children=[
                dcc.Markdown('''
                    * **Red Dots (Produce Origins):** These represent the counties or areas where tomatoes are sourced. <br>
                    * **Blue Dots (Markets):** These are the markets where tomatoes are sold. <br>
                    * **Lines (Trade Routes):** These connect an origin to a market. The color and thickness show the share of that market's tomatoes that come from the connected origin. <br>
                        * **Orange (Thick):** High Share - Over 75% of the market's supply comes from this origin.
                        * **Purple (Medium):** Medium Share - Between 25% and 75% of the supply comes from this origin.
                        * **Green (Thin):** Low Share - Less than 25% of the supply comes from this origin. <br>
                    * Units of tomatoes shown are the ones traded on a typical day during the selected season
                    ''', style={'padding': '15px', 'border': '1px dashed #cce5ff', 'borderRadius': '5px', 'backgroundColor': '#f8f9fa'})
            ], style={'display': 'none'})
        ]),
        html.H3(id='network-map-title', style=title_style),
        dcc.Graph(id='network-map', style={'height': '85vh'}, config={'scrollZoom': True}),
    ]),
    
    html.Div(style=section_style, children=[
        html.H2("Trader Composition Analysis", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
        html.Div(style={'display': 'flex', 'gap': '40px', 'alignItems': 'center', 'marginBottom': '20px', 'flex-wrap': 'wrap'}, children=[
            html.Div(style={'flex': 1, 'min-width': '250px'}, children=[
                html.Label("Filter by Trader Type:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Dropdown(id='trader-type-dropdown', options=[{'label': 'All Traders', 'value': 'All'}] + [{'label': t, 'value': t} for t in trader_df['trader_id'].unique() if t], value='All'),
            ]),
            html.Div(style={'flex': 2, 'min-width': '300px'}, children=[
                html.Label("Select Time Period:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Slider(id='time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None),
            ]),
        ]),
        html.H3(id='trader-map-title', style=title_style),
        dcc.Graph(id='trader-map', style={'height': '85vh'}),
    ]),

    html.Div(style=section_style, children=[
        html.H2("Tomato Trade Volume", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
        html.P("Shows the concentration of tomato trade by market. Dot size and color reflect the estimated daily quantity of tomatoes traded on a typical day during the selected season.", style={'marginBottom': '20px'}),
        html.Div(style={'display': 'flex', 'gap': '40px', 'alignItems': 'center', 'marginBottom': '20px', 'flex-wrap': 'wrap'}, children=[
             html.Div(children=[
                html.Label("Select Season:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(id='volume-season-toggle', options=[{'label': 'High', 'value': 'High Season'}, {'label': 'Low', 'value': 'Low Season'}], value='High Season', inline=True, labelStyle={'margin-right': '15px'}),
            ]),
             html.Div(style={'flex': 2, 'min-width': '300px'}, children=[
                html.Label("Select Time Period:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Slider(id='volume-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None),
            ]),
        ]),
        html.H3(id='volume-map-title', style=title_style),
        dcc.Graph(id='volume-map', style={'height': '85vh'}),
    ]),
    
    html.Div(style=section_style, children=[
        html.H2("Trader Population Trends", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
        html.P("Compares the total number of traders over time. Leaving the filter empty shows all areas.", style={'marginBottom': '20px'}),
        html.Div(style={'display':'flex', 'gap':'30px', 'flex-wrap': 'wrap'}, children=[
            html.Div(style={'flex':1, 'min-width': '250px'}, children=[
                html.Label("Filter by Trader Type:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Dropdown(id='trends-trader-type-dropdown', options=[{'label': 'All Traders', 'value': 'All'}] + [{'label': t, 'value': t} for t in trader_df['trader_id'].unique() if t], value='All'),
            ]),
            html.Div(style={'flex':1, 'min-width': '250px'}, children=[
                html.Label("Select Analysis Level:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.RadioItems(id='grouping-level-radio', options=[{'label': 'Group by County', 'value': 'county_id'}, {'label': 'Group by Region', 'value': 'region'}], value='county_id', inline=True),
            ]),
        ]),
        html.Div(style={'marginTop':'15px'}, children=[
            dcc.Dropdown(id='area-dropdown', value=[], multi=True, placeholder="Select one or more areas..."),
        ]),
        dcc.Graph(id='trends-chart', style={'marginTop': '20px'}),
    ]),

    html.Footer([
        html.P("The INCATA project is funded by the Gates Foundation.",
               style={'color': '#6c757d', 'fontSize': '0.9em'})
    ], style={'textAlign': 'center', 'padding': '20px 0', 'marginTop': '40px', 'borderTop': '1px solid #dee2e6'})
])


# --- 3. DASH CALLBACKS ---

@app.callback(
    Output('network-info-collapse', 'style'),
    Input('network-info-button', 'n_clicks'),
    State('network-info-collapse', 'style'),
    prevent_initial_call=True
)
def toggle_network_info(n, current_style):
    if n > 0:
        if current_style.get('display') == 'none':
            return {'display': 'block'}
        else:
            return {'display': 'none'}
    return {'display': 'none'}

@app.callback(
    [Output('network-map', 'figure'),
     Output('network-map-title', 'children')],
    [Input('master-market-type-filter', 'value'),
     Input('season-toggle', 'value'),
     Input('network-time-slider', 'value'),
     Input('opacity-slider', 'value'),
     Input('toggle-routes', 'value'),
     Input('layer-toggles', 'value')],
    [State('network-map', 'relayoutData')]
)
def update_network_map(selected_market_type, selected_season, time_value, opacity_percent, toggle_value, layer_toggles, relayout_data):
    time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
    selected_time = time_map[time_value]
    layer_toggles = layer_toggles or []

    df_flow = network_df[(network_df['season'] == selected_season) & (network_df['Time Period'] == selected_time)]
    if selected_market_type != 'All Markets':
        df_flow = df_flow[df_flow['mkt_type'] == selected_market_type]
    df_map = df_flow[df_flow['share'] > 0].copy()
    df_vol = market_volume_df[(market_volume_df['season'] == selected_season) & (market_volume_df['Time Period'] == selected_time)]

    map_title = f'Produce Flow Network - {selected_season} ({selected_time})'
    map_style = "carto-darkmatter" if 'show_nightlights' in layer_toggles else "carto-positron"
    mapbox_layers = []

    if 'show_nightlights' in layer_toggles and selected_time in nightlights_data:
        b64_img, coords = nightlights_data[selected_time]
        mapbox_layers.append({
            "source": b64_img, "sourcetype": "image", "coordinates": coords,
            "opacity": 0.8, "below": ""
        })

    if 'show_roads' in layer_toggles and selected_time in roads_data:
        road_color = 'rgba(211, 211, 211, 0.6)' if 'show_nightlights' in layer_toggles else 'rgba(100, 100, 100, 0.7)'
        mapbox_layers.append({
            'source': roads_data[selected_time],
            'type': 'line',
            'color': road_color,
            'line': {'width': 0.8},
            'below': 'traces'
        })

    zoom = 5.5
    center = {"lat": 0.5, "lon": 37.5}
    if relayout_data and 'mapbox.center' in relayout_data:
        zoom = relayout_data['mapbox.zoom']
        center = relayout_data['mapbox.center']

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
                line_lats, line_lons = [], []
                for _, row in s_bin['data'].iterrows():
                    line_lats.extend([row['origin_lat'], row['market_lat'], None])
                    line_lons.extend([row['origin_lon'], row['market_lon'], None])
                fig.add_trace(go.Scattermapbox(lat=line_lats, lon=line_lons, mode='lines', line=dict(width=s_bin['width'], color=s_bin['color']), name=s_bin['name'], hoverinfo='none', visible=routes_visible))

        if not df_map.empty:
            origins = df_map[['origin_name', 'origin_lat', 'origin_lon']].drop_duplicates()
            origin_counts = df_map.groupby('origin_name')['mkt_name'].nunique().reset_index(name='market_count')
            origins = pd.merge(origins, origin_counts, on='origin_name')
            origins['hover_text'] = origins['origin_name'] + '<br>Supplies ' + origins['market_count'].astype(str) + ' market(s)'
            fig.add_trace(go.Scattermapbox(lat=origins['origin_lat'], lon=origins['origin_lon'], mode='markers', marker=dict(size=(5 + origins['market_count']), color='#a50f15', opacity=0.9), name='Produce Origin', text=origins['hover_text'], hoverinfo='text'))

            markets = df_map[['mkt_id', 'mkt_name', 'market_lat', 'market_lon', 'mkt_type']].drop_duplicates()
            markets = pd.merge(markets, df_vol[['mkt_id', 'Total Volume']], on='mkt_id', how='left').fillna(0)
            df_map['origin_share_str'] = df_map['origin_name'].astype(str) + ': ' + df_map['share'].astype(int).astype(str) + '%'
            market_hover_info = df_map.groupby('mkt_name')['origin_share_str'].apply('<br>'.join).reset_index(name='details')
            markets = pd.merge(markets, market_hover_info, on='mkt_name')
            markets['hover_text'] = '<b>' + markets['mkt_name'] + '</b><br><i>' + markets['mkt_type'] + '</i><br>' + 'Trade Quantity: ' + markets['Total Volume'].round(0).astype(int).apply(lambda x: f'{x:,}') + ' units<br>' + '--- Origins ---<br>' + markets['details']
            markets['size'] = 4 + (markets['Total Volume'] ** 0.5) * 0.08
            fig.add_trace(go.Scattermapbox(lat=markets['market_lat'], lon=markets['market_lon'], mode='markers', marker=dict(size=markets['size'], color='blue', opacity=0.9), name='Market', text=markets['hover_text'], hoverinfo='text'))
        
        elif 'show_markers' in layer_toggles:
            fig.add_annotation(text="No trade flow data for this selection.", showarrow=False, font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"))

    fig.add_trace(go.Scattermapbox(
        lat=[0], lon=[37.5], mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='none'
    ))

    fig.update_layout(
        mapbox_style=map_style,
        mapbox_layers=mapbox_layers,
        mapbox_zoom=zoom,
        mapbox_center=center,
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.7)',
            traceorder='normal',
            itemsizing='constant'
        )
    )
    
    if 'show_roads' in layer_toggles:
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=[None], lat=[None],
            line=dict(color=road_color, width=2),
            name='Road Network'
        ))

    return fig, map_title

@app.callback(
    [Output('opacity-input', 'value'), Output('opacity-slider', 'value')],
    [Input('opacity-input', 'value'), Input('opacity-slider', 'value')],
    prevent_initial_call=True
)
def sync_opacity_controls(input_val, slider_val):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    return (input_val, input_val) if triggered_id == 'opacity-input' else (slider_val, slider_val)

@app.callback(
    [Output('trader-map', 'figure'),
     Output('trader-map-title', 'children')],
    [Input('master-market-type-filter', 'value'), Input('trader-type-dropdown', 'value'), Input('time-slider', 'value')],
    [State('trader-map', 'relayoutData')]
)
def update_trader_map(selected_market_type, selected_trader, time_value, relayout_data):
    df = trader_df.copy()
    if selected_market_type != 'All Markets': df = df[df['mkt_type'] == selected_market_type]
    
    time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
    selected_time_col = time_map[time_value]
    if selected_trader != 'All': df = df[df['trader_id'] == selected_trader]
    
    fig = go.Figure()
    map_title = f'Quantity of {selected_trader} - {selected_time_col}'
    
    if df.empty:
        fig.update_layout(annotations=[dict(text="No data.", showarrow=False)])
        return fig, map_title
    
    market_traders = df.groupby(['mkt_name', 'lat', 'lon'])[selected_time_col].sum().reset_index()
    market_traders = market_traders[market_traders[selected_time_col] > 0]
    
    if market_traders.empty:
        fig.update_layout(annotations=[dict(text="No traders.", showarrow=False)])
        return fig, map_title
    
    market_traders['hover_text'] = market_traders['mkt_name'] + '<br>' + market_traders[selected_time_col].astype(int).astype(str) + ' ' + selected_trader + '(s)'
    fig.add_trace(go.Scattermapbox(lat=market_traders['lat'], lon=market_traders['lon'], mode='markers', marker=go.scattermapbox.Marker(size=(4 + (market_traders[selected_time_col]**0.5) * 1.5), color=market_traders[selected_time_col], colorscale='Plasma', cmin=0, cmax=market_traders[selected_time_col].quantile(0.95), showscale=True, colorbar_title_text='No. of Traders'), text=market_traders['hover_text'], hoverinfo='text'))
    
    zoom = 5.5
    center = {"lat": 0.5, "lon": 37.5}
    if relayout_data and 'mapbox.center' in relayout_data:
        zoom = relayout_data['mapbox.zoom']
        center = relayout_data['mapbox.center']
        
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=zoom, mapbox_center=center, margin={"r":0, "t":0, "l":0, "b":0})
    return fig, map_title

@app.callback(
    [Output('volume-map', 'figure'),
     Output('volume-map-title', 'children')],
    [Input('master-market-type-filter', 'value'),
     Input('volume-season-toggle', 'value'),
     Input('volume-time-slider', 'value')],
    [State('volume-map', 'relayoutData')]
)
def update_volume_map(selected_market_type, selected_season, time_value, relayout_data):
    df = market_volume_df.copy()
    if selected_market_type != 'All Markets':
        df = df[df['mkt_type'] == selected_market_type]
    
    time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
    selected_time = time_map[time_value]
    df = df[(df['season'] == selected_season) & (df['Time Period'] == selected_time)]
    df = df[df['Total Volume'] > 0]
    
    fig = go.Figure()
    map_title = f'Tomato Trade Quantity - {selected_season} ({selected_time})'

    if df.empty:
        fig.update_layout(annotations=[dict(text="No volume data for this selection.", showarrow=False)])
        return fig, map_title

    df['hover_text'] = '<b>' + df['mkt_name'] + '</b><br>Quantity: ' + df['Total Volume'].round(0).astype(int).apply(lambda x: f'{x:,}') + ' units'
    
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'], lon=df['lon'], mode='markers',
        marker=go.scattermapbox.Marker(
            size=4 + (df['Total Volume'] ** 0.5) * 0.08,
            color=df['Total Volume'],
            colorscale='Viridis', cmin=0, cmax=df['Total Volume'].quantile(0.95),
            showscale=True,
            colorbar_title_text='Trade Quantity'
        ),
        text=df['hover_text'], hoverinfo='text'
    ))
    
    zoom = 5.5
    center = {"lat": 0.5, "lon": 37.5}
    if relayout_data and 'mapbox.center' in relayout_data:
        zoom = relayout_data['mapbox.zoom']
        center = relayout_data['mapbox.center']
        
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=zoom, mapbox_center=center, margin={"r":0, "t":0, "l":0, "b":0})
    return fig, map_title

@app.callback(
    [Output('area-dropdown', 'options'), Output('area-dropdown', 'placeholder')], 
    Input('grouping-level-radio', 'value')
)
def update_area_dropdown(grouping_level):
    if grouping_level == 'region':
        options = [{'label': r, 'value': r} for r in sorted(trends_base_df['region'].dropna().unique())]
        placeholder = "Select one or more regions..."
    else:
        options = [{'label': c, 'value': c} for c in sorted(trends_base_df['county_id'].unique())]
        placeholder = "Select one or more counties..."
    return options, placeholder

@app.callback(
    Output('trends-chart', 'figure'),
    [Input('master-market-type-filter', 'value'), Input('trends-trader-type-dropdown', 'value'),
     Input('grouping-level-radio', 'value'), Input('area-dropdown', 'value')]
)
def update_trends_chart(selected_market_type, selected_trader, grouping_level, selected_areas):
    df = trends_base_df.copy()
    if selected_market_type != 'All Markets': df = df[df['mkt_type'] == selected_market_type]
    if selected_trader != 'All': df = df[df['trader_id'] == selected_trader]

    df_agg = df.groupby([grouping_level, 'Time Period'])['Number of Traders'].sum().reset_index()
    if selected_areas: df_agg = df_agg[df_agg[grouping_level].isin(selected_areas)]
    
    if df_agg.empty or df_agg['Number of Traders'].sum() == 0: 
        return go.Figure().update_layout(title_text="No data.", annotations=[dict(text="No data available.", showarrow=False)])
    
    df_agg = df_agg.sort_values('Time Period')
    fig = px.bar(
        df_agg, x=grouping_level, y='Number of Traders', color='Time Period', barmode='group',
        title=f'Trader Population for {selected_trader} by {grouping_level.replace("_id","").title()}',
        labels={grouping_level: grouping_level.replace("_id","").title(), 'Number of Traders': 'Total Number of Traders'},
        color_discrete_map={'10 Yrs Ago': '#66a61e', '5 Yrs Ago': '#7570b3', 'Now': '#d95f02'}
    )
    fig.update_layout(plot_bgcolor='white', xaxis_title=None, legend_title_text='Time Period', xaxis={'categoryorder':'total descending'})
    return fig


# --- 4. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=False)
