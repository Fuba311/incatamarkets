# filename: app.py
# --- FINAL, CORRECTED VERSION FOR RENDER DEPLOYMENT - v5 ---
# This version restores the opacity slider and the correct behavior of the 'How to Read This Map' feature.

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
print("--- Loading Pre-Processed Data ---")
PROCESSED_DATA_FOLDER = Path(__file__).parent / "processed_data"

try:
    network_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'network_df.parquet')
    market_volume_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'market_volume_df.parquet')
    trader_df = pd.read_parquet(PROCESSED_DATA_FOLDER / 'trader_df.parquet')
    print("SUCCESS: Tabular data (.parquet) loaded.")

    roads_data, nightlights_data = {}, {}
    time_period_map = {"10 Yrs Ago": "10_yrs_ago", "5 Yrs Ago": "5_yrs_ago", "Now": "now"}

    for key, file_suffix in time_period_map.items():
        road_path = PROCESSED_DATA_FOLDER / f"roads_{file_suffix}_processed.geojson"
        if road_path.exists():
            roads_data[key] = gpd.read_file(road_path).__geo_interface__
    print("SUCCESS: Roads data (.geojson) loaded.")

    nightlight_path = PROCESSED_DATA_FOLDER / "nightlights_data.json"
    if nightlight_path.exists():
        with open(nightlight_path, 'r') as f: nightlights_data = json.load(f)
        print("SUCCESS: Nightlights data (.json) loaded.")

except FileNotFoundError as e:
    print(f"---! FATAL ERROR !---: {e}")
    app.layout = html.Div([ html.H1("Application Error"), html.P(f"Could not load necessary data file: {e}.") ])
else:
    print("--- All Data Loaded. Building Layout. ---")

    # --- 2. DASH APP LAYOUT ---
    section_style = {'background-color': '#f0f8ff', 'border': '1px solid #cce5ff', 'border-radius': '10px', 'padding': '25px', 'box-shadow': '2px 2px 10px lightgrey', 'margin-bottom': '40px'}
    title_style = {'textAlign': 'center', 'color': '#333333', 'marginBottom': '20px'}
    map_style_light = "carto-positron"
    map_style_dark = "carto-darkmatter"

    app.layout = html.Div(style={'fontFamily': "'Segoe UI', 'Roboto', Arial, sans-serif", 'padding': '2% 5%', 'background-color': '#f8f9fa'}, children=[
        
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
        
        html.Div(style=section_style, children=[
            html.H2("Produce Flow Network", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
            html.P("This map shows the origin and flow of tomatoes. Use the toggles to show/hide roads, nightlights, and the market network. The origins of tomatoes' (Red dots) position are approximations only. Please be patient when selecting any option that will dynamically update the map, as this website is hosted on a free instance and may be slow at times.", style={'marginBottom': '20px'}),
            
            # --- MAP CONTAINER ---
            html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                
                # --- FLOATING CONTROL PANEL ---
                html.Div(id={'type': 'floating-panel-wrapper', 'index': 'network'}, className='floating-controls', children=[
                    html.Div(id={'type': 'panel-header', 'index': 'network'}, className='control-panel-header', n_clicks=0, children=[
                        html.Span("⚙️", className='header-icon'), html.Span("Map Controls", className='header-text'), html.Button('−', className='toggle-btn')
                    ]),
                    html.Div(id={'type': 'panel-content', 'index': 'network'}, className='controls-content', children=[
                        html.Div(className='control-group', children=[html.Label([html.Span("📅"), " Time Period"]), dcc.Slider(id='network-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False)]),
                        html.Div(className='control-group', children=[html.Label([html.Span("🌱"), " Season"]), dcc.RadioItems(id='season-toggle', options=[{'label': ' High', 'value': 'High Season'}, {'label': ' Low', 'value': 'Low Season'}], value='High Season', labelClassName='radio-label', inputClassName='radio-input')]),
                        html.Div(className='control-group', children=[html.Label([html.Span("🗺"), " Map Layers"]), dcc.Checklist(id='layer-toggles', options=[{'label': ' Markets & Origins', 'value': 'show_markers'}, {'label': ' Roads', 'value': 'show_roads'}, {'label': ' Nightlights', 'value': 'show_nightlights'}], value=['show_markers'], labelClassName='check-label')]),
                        html.Div(className='control-group', children=[html.Label([html.Span("🔗"), " Trade Routes"]), dcc.Checklist(id='toggle-routes', options=[{'label': ' Show Trade Routes', 'value': 'show'}], value=['show'], labelClassName='check-label')]),
                        
                        # <<< ROUTE OPACITY SLIDER RESTORED HERE
                        html.Div(className='control-group', children=[
                            html.Label([html.Span("💧"), " Route Opacity"]),
                            dcc.Slider(id='opacity-slider', min=0, max=100, step=10, value=70, marks={i: f'{i}%' for i in range(0, 101, 20)}),
                        ]),
                        
                        # <<< 'HOW TO READ THIS MAP' BUTTON MOVED HERE
                        html.Div(style={'borderTop': '1px solid #e0e0e0', 'paddingTop': '10px', 'marginTop': '10px'}, children=[
                            html.Button('ℹ How to Read This Map', id='network-info-button', n_clicks=0, style={'width': '100%', 'cursor': 'pointer', 'border': '1px solid #004085', 'backgroundColor': '#e7f3ff', 'padding': '5px 10px', 'borderRadius': '5px', 'fontSize': '12px'}),
                        ]),
                    ]),
                ]),
                
                # --- THE MAP GRAPH ITSELF ---
                dcc.Graph(id='network-map', style={'height': '85vh', 'width': '100%'}, config={'scrollZoom': True}),

                # <<< INFO BOX OVERLAY RESTORED HERE
                html.Div(id='network-info-collapse', style={'display': 'none', 'position': 'absolute', 'bottom': '10px', 'right': '10px', 'backgroundColor': 'rgba(248, 249, 250, 0.95)', 'padding': '15px', 'border': '1px solid #cce5ff', 'borderRadius': '5px', 'z-index': '998', 'max-width': '400px', 'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'}, children=[
                    html.Button('✕', id='close-info-btn', n_clicks=0, style={'position': 'absolute', 'top': '5px', 'right': '10px', 'background': 'transparent', 'border': 'none', 'fontSize': '18px', 'cursor': 'pointer', 'lineHeight': '1'}),
                    dcc.Markdown('''
                        * **Red Dots (Produce Origins):** Size reflects the number of unique markets it supplies.
                        * **Blue Dots (Markets):** Size reflects the total trade volume.
                        * **Lines (Trade Routes):** Connects an origin to a market. Color and thickness show the share of that market's tomatoes that come from the connected origin:
                            * **Orange (Thick):** High Share (>75%)
                            * **Purple (Medium):** Medium Share (25-75%)
                            * **Green (Thin):** Low Share (<25%)
                        ''', style={'fontSize': '12px', 'margin': '0'})
                ])
            ]),
        ]),
        
        # --- Other sections remain the same ---
        html.Div(style=section_style, children=[
            html.H2("Market Concentration Analysis", style={'color': '#004085', 'border-bottom': '2px solid #b8daff', 'padding-bottom': '10px'}),
            html.P("Analyze tomato trade volume or trader concentration. Use the controls to switch data types and visualization styles.", style={'marginBottom': '20px'}),
            html.H3(id='combined-map-title', style=title_style),
            html.Div(style={'position': 'relative', 'width': '100%'}, children=[
                html.Div(id={'type': 'floating-panel-wrapper', 'index': 'combined'}, className='floating-controls', children=[
                    html.Div(id={'type': 'panel-header', 'index': 'combined'}, className='control-panel-header', n_clicks=0, children=[
                        html.Span("⚙️", className='header-icon'), html.Span("Analysis Controls", className='header-text'), html.Button('−', className='toggle-btn')
                    ]),
                    html.Div(id={'type': 'panel-content', 'index': 'combined'}, className='controls-content', children=[
                        html.Div(className='control-group', children=[html.Label([html.Span("📊"), " Analysis Type"]), dcc.RadioItems(id='data-type-toggle', options=[{'label': ' Tomato Volume', 'value': 'tomatoes'}, {'label': ' Traders', 'value': 'traders'}], value='tomatoes', labelClassName='radio-label', inputClassName='radio-input')]),
                        html.Div(className='control-group', children=[html.Label([html.Span("🎨"), " View Style"]), dcc.RadioItems(id='view-type-toggle', options=[{'label': ' Points', 'value': 'points'}, {'label': ' Heatmap', 'value': 'heatmap'}], value='points', labelClassName='radio-label', inputClassName='radio-input')]),
                        html.Div(id='season-control-div', className='control-group', children=[html.Label([html.Span("🌱"), " Season"]), dcc.RadioItems(id='combined-season-toggle', options=[{'label': ' High', 'value': 'High Season'}, {'label': ' Low', 'value': 'Low Season'}], value='High Season', labelClassName='radio-label', inputClassName='radio-input')]),
                        html.Div(id='trader-control-div', className='control-group', children=[html.Label([html.Span("👤"), " Trader Type"]), dcc.Dropdown(id='combined-trader-type-dropdown', options=[{'label': 'All Traders', 'value': 'All'}] + [{'label': t, 'value': t} for t in sorted(trader_df['trader_id'].unique()) if pd.notna(t)], value='All')]),
                        html.Div(className='control-group', children=[html.Label([html.Span("📅"), " Time Period"]), dcc.Slider(id='combined-time-slider', min=0, max=2, marks={0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}, value=2, step=None, included=False)]),
                    ]),
                ]),
                dcc.Graph(id='combined-map', style={'height': '85vh'}),
            ]),
        ]),
        
        html.Footer([
            html.P("The INCATA project is funded by the Gates Foundation.", style={'color': '#6c757d', 'fontSize': '0.9em'})
        ], style={'textAlign': 'center', 'padding': '20px 0', 'marginTop': '40px', 'borderTop': '1px solid #dee2e6'})
    ])

    # --- 3. DASH CALLBACKS ---

    @app.callback([Output({'type': 'floating-panel-wrapper', 'index': MATCH}, 'className'), Output({'type': 'panel-content', 'index': MATCH}, 'className')], Input({'type': 'panel-header', 'index': MATCH}, 'n_clicks'), State({'type': 'floating-panel-wrapper', 'index': MATCH}, 'className'), prevent_initial_call=True)
    def toggle_panel_animation(n, current_class):
        if n and n > 0: return ('floating-controls icon-only', 'controls-content hidden') if 'icon-only' not in current_class else ('floating-controls', 'controls-content')
        return no_update, no_update

    # <<< CALLBACK FOR INFO BOX RESTORED HERE
    @app.callback(Output('network-info-collapse', 'style'), [Input('network-info-button', 'n_clicks'), Input('close-info-btn', 'n_clicks')], State('network-info-collapse', 'style'), prevent_initial_call=True)
    def toggle_network_info(info_clicks, close_clicks, current_style):
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'network-info-button':
            return {'display': 'block'}
        if triggered_id == 'close-info-btn':
            return {'display': 'none'}
        return current_style

    # ----- NETWORK MAP CALLBACK (Corrected and Final)-----
    @app.callback(Output('network-map', 'figure'), [Input('master-market-type-filter', 'value'), Input('season-toggle', 'value'), Input('network-time-slider', 'value'), Input('toggle-routes', 'value'), Input('layer-toggles', 'value'), Input('opacity-slider', 'value')], [State('network-map', 'relayoutData')])
    def update_network_map(selected_market_type, selected_season, time_value, toggle_value, layer_toggles, opacity_percent, relayout_data):
        time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
        selected_time = time_map[time_value]
        layer_toggles = layer_toggles or []
        opacity = opacity_percent / 100.0

        df_flow = network_df[(network_df['season'] == selected_season) & (network_df['Time Period'] == selected_time)]
        if selected_market_type != 'All Markets':
            df_flow = df_flow[df_flow['mkt_type'] == selected_market_type]
        df_map = df_flow[df_flow['share'] > 0].copy()
        df_vol = market_volume_df[(market_volume_df['season'] == selected_season) & (market_volume_df['Time Period'] == selected_time)]
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
            zoom, center = relayout_data['mapbox.zoom'], relayout_data['mapbox.center']
        fig = go.Figure()
        if 'show_markers' in layer_toggles and not df_map.empty:
            routes_visible = 'show' in toggle_value
            share_bins = [
                {'name': 'High Share (>75%)', 'data': df_map[df_map['share'] >= 75], 'width': 4, 'color': f'rgba(217, 95, 2, {opacity})'},
                {'name': 'Medium Share (25-75%)', 'data': df_map[(df_map['share'] < 75) & (df_map['share'] >= 25)], 'width': 2, 'color': f'rgba(117, 112, 179, {opacity})'},
                {'name': 'Low Share (<25%)', 'data': df_map[df_map['share'] < 25], 'width': 1, 'color': f'rgba(102, 166, 30, {opacity})'}
            ]
            for s_bin in share_bins:
                if not s_bin['data'].empty:
                    lats, lons = [item for _, row in s_bin['data'].iterrows() for item in (row['origin_lat'], row['market_lat'], None)], [item for _, row in s_bin['data'].iterrows() for item in (row['origin_lon'], row['market_lon'], None)]
                    fig.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode='lines', line=dict(width=s_bin['width'], color=s_bin['color']), name=s_bin['name'], hoverinfo='none', visible=routes_visible))
            origins = df_map[['origin_name', 'origin_lat', 'origin_lon']].drop_duplicates().merge(df_map.groupby('origin_name', observed=True)['mkt_name'].nunique().reset_index(name='market_count'), on='origin_name')
            origins['hover_text'] = origins['origin_name'] + '<br>Supplies ' + origins['market_count'].astype(str) + ' market(s)'
            fig.add_trace(go.Scattermapbox(lat=origins['origin_lat'], lon=origins['origin_lon'], mode='markers', marker=dict(size=(5 + origins['market_count']), color='#a50f15', opacity=0.9), name='Produce Origin', text=origins['hover_text'], hoverinfo='text'))
            markets = df_map[['mkt_id', 'mkt_name', 'market_lat', 'market_lon', 'mkt_type']].drop_duplicates().merge(df_vol[['mkt_id', 'Total Volume']], on='mkt_id', how='left').fillna(0)
            market_hover_info = df_map.assign(origin_share_str=df_map['origin_name'].astype(str) + ': ' + df_map['share'].astype(int).astype(str) + '%').groupby('mkt_name', observed=True)['origin_share_str'].apply('<br>'.join).reset_index(name='details')
            markets = markets.merge(market_hover_info, on='mkt_name', how='left')
            markets['hover_text'] = '<b>' + markets['mkt_name'] + '</b><br><i>' + markets['mkt_type'] + '</i><br>' + 'Trade Volume: ' + markets['Total Volume'].round(0).astype(int).apply(lambda x: f'{x:,}') + ' units<br>' + '--- Origins ---<br>' + markets['details'].fillna('')
            markets['size'] = 4 + (markets['Total Volume'] ** 0.5) * 0.08
            fig.add_trace(go.Scattermapbox(lat=markets['market_lat'], lon=markets['market_lon'], mode='markers', marker=dict(size=markets['size'], color='blue', opacity=0.9), name='Market', text=markets['hover_text'], hoverinfo='text'))
        elif 'show_markers' in layer_toggles:
             fig.add_annotation(text="No trade flow data for this selection.", showarrow=False, font=dict(size=16, color="white" if "show_nightlights" in layer_toggles else "black"))
        fig.update_layout(mapbox_style=map_style, mapbox_layers=mapbox_layers, mapbox_zoom=zoom, mapbox_center=center, margin={"r":0, "t":0, "l":0, "b":0}, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, traceorder='normal', itemsizing='constant', font=dict(size=11)))
        return fig

    # ----- COMBINED MAP CALLBACKS -----
    @app.callback([Output('trader-control-div', 'style'), Output('season-control-div', 'style')], Input('data-type-toggle', 'value'))
    def update_combined_map_controls(data_type):
        if data_type == 'traders': return {'display': 'block'}, {'display': 'none'}
        else: return {'display': 'none'}, {'display': 'block'}

    @app.callback([Output('combined-map', 'figure'), Output('combined-map-title', 'children')], [Input('master-market-type-filter', 'value'), Input('data-type-toggle', 'value'), Input('view-type-toggle', 'value'), Input('combined-trader-type-dropdown', 'value'), Input('combined-season-toggle', 'value'), Input('combined-time-slider', 'value')], [State('combined-map', 'relayoutData')])
    def update_combined_map(market_type, data_type, view_type, selected_trader, selected_season, time_value, relayout_data):
        time_map = {0: '10 Yrs Ago', 1: '5 Yrs Ago', 2: 'Now'}
        selected_time = time_map[time_value]
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
            title_parts, z_value_col, colorscale, colorbar_title, heatmap_radius = [f'{selected_trader} Traders' if selected_trader != 'All' else 'All Traders', selected_time], selected_time, 'Plasma', 'No. of Traders', 30
        map_title, fig = ' - '.join(title_parts), go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available for this selection.", showarrow=False)
        else:
            df['hover_text'] = '<b>' + df['mkt_name'] + '</b><br>' + colorbar_title + ': ' + df[z_value_col].round(0).astype(int).apply(lambda x: f'{x:,}')
            if view_type == 'points':
                df['size'] = 5 + (df[z_value_col] ** 0.5) * (0.1 if data_type == 'tomatoes' else 0.8)
                fig.add_trace(go.Scattermapbox(lat=df['lat'], lon=df['lon'], mode='markers', marker=dict(size=df['size'], color=df[z_value_col], colorscale=colorscale, cmin=0, cmax=df[z_value_col].quantile(0.95), showscale=True, colorbar_title_text=colorbar_title), text=df['hover_text'], hoverinfo='text'))
            else:
                fig.add_trace(go.Densitymapbox(lat=df['lat'], lon=df['lon'], z=df[z_value_col], radius=heatmap_radius, colorscale=colorscale, colorbar_title_text=colorbar_title))
                fig.add_trace(go.Scattermapbox(lat=df['lat'], lon=df['lon'], mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'), text=df['hover_text'], hoverinfo='text', showlegend=False))
        zoom, center = (5.5, {"lat": 0.5, "lon": 37.5})
        if relayout_data and 'mapbox.center' in relayout_data:
            zoom, center = relayout_data['mapbox.zoom'], relayout_data['mapbox.center']
        fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0}, mapbox_style=map_style_light, mapbox_zoom=zoom, mapbox_center=center)
        return fig, map_title

# --- 4. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=False)
