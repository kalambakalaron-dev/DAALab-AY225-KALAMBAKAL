import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import networkx as nx
import plotly.graph_objects as go

route_data = [
    {"from": "IMUS", "to": "BACOOR", "dist": 10, "time": 15, "fuel": 1.2},
    {"from": "BACOOR", "to": "DASMA", "dist": 12, "time": 25, "fuel": 1.5},
    {"from": "DASMA", "to": "KAWIT", "dist": 12, "time": 25, "fuel": 1.5},
    {"from": "KAWIT", "to": "INDANG", "dist": 12, "time": 25, "fuel": 1.2},
    {"from": "INDANG", "to": "SILANG", "dist": 14, "time": 25, "fuel": 1.5},
    {"from": "SILANG", "to": "GENTRI", "dist": 10, "time": 25, "fuel": 1.3},
    {"from": "GENTRI", "to": "NOVELETA", "dist": 10, "time": 25, "fuel": 1.5},
    {"from": "NOVELETA", "to": "IMUS", "dist": 10, "time": 15, "fuel": 1.2},
    {"from": "BACOOR", "to": "SILANG", "dist": 10, "time": 25, "fuel": 1.3},
    {"from": "DASMA", "to": "SILANG", "dist": 12, "time": 25, "fuel": 1.5},
    {"from": "SILANG", "to": "BACOOR", "dist": 10, "time": 25, "fuel": 1.3},
    {"from": "NOVELETA", "to": "BACOOR", "dist": 10, "time": 15, "fuel": 1.2},
    {"from": "SILANG", "to": "KAWIT", "dist": 14, "time": 25, "fuel": 1.2},
    {"from": "IMUS", "to": "NOVELETA", "dist": 10, "time": 15, "fuel": 1.2},
    {"from": "BACOOR", "to": "IMUS", "dist": 10, "time": 15, "fuel": 1.2},
    {"from": "KAWIT", "to": "DASMA", "dist": 12, "time": 25, "fuel": 1.5},
    {"from": "DASMA", "to": "BACOOR", "dist": 12, "time": 25, "fuel": 1.5},
    {"from": "BACOOR", "to": "NOVELETA", "dist": 10, "time": 15, "fuel": 1.2}
]

G = nx.DiGraph()
for r in route_data:
    G.add_edge(r["from"], r["to"], weight_dist=r["dist"], weight_time=r["time"], weight_fuel=r["fuel"])

node_list = sorted(list(G.nodes()))
pos = nx.spring_layout(G, k=1.5, seed=42)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])

app.layout = html.Div(style={
    'backgroundColor': '#121212', 'minHeight': '100vh', 'padding': '20px', 
    'fontFamily': 'Segoe UI, sans-serif'
}, children=[
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("🛰️ CAVITE OPTIMIZER", style={'color': '#00e5ff', 'fontWeight': '900', 'textShadow': '0 0 10px #00e5ff'}),
            html.P("REAL-TIME ROUTE OPTIMIZATION", className="text-muted small")
        ], className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(lg=4, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H6("PARAMETERS", className="text-info mb-3"),
                    html.Label("START POINT", className="small text-muted"),
                    dcc.Dropdown(id='start-node', options=[{'label': n, 'value': n} for n in node_list], value='KAWIT', className="mb-3"),
                    html.Label("END POINT", className="small text-muted"),
                    dcc.Dropdown(id='end-node', options=[{'label': n, 'value': n} for n in node_list], value='NOVELETA', className="mb-3"),
                    html.Label("CRITERIA", className="small text-muted"),
                    dbc.Select(id='criteria', options=[
                        {'label': '🛣️ Shortest Distance', 'value': 'weight_dist'},
                        {'label': '⏱️ Fastest Time', 'value': 'weight_time'},
                        {'label': '⛽ Fuel Efficiency', 'value': 'weight_fuel'}
                    ], value='weight_dist', className="mb-4"),
                    html.Div(id='stats-panel')
                ])
            ], style={'background': '#1a1a1a', 'border': '1px solid #333', 'borderRadius': '15px'})
        ]),

        dbc.Col(lg=8, children=[
            html.Div([
                dcc.Graph(id='main-graph', style={'height': '75vh'}, config={'displayModeBar': False})
            ], style={'borderRadius': '15px', 'background': '#000', 'padding': '10px', 'border': '1px solid #333'})
        ])
    ])
])

@app.callback(
    [Output('main-graph', 'figure'), Output('stats-panel', 'children')],
    [Input('start-node', 'value'), Input('end-node', 'value'), Input('criteria', 'value')]
)
def update_system(start, end, criteria):
    path = []
    d_tot = t_tot = f_tot = 0
    try:
        path = nx.shortest_path(G, start, end, weight=criteria)
        for i in range(len(path)-1):
            e = G[path[i]][path[i+1]]
            d_tot += e['weight_dist']; t_tot += e['weight_time']; f_tot += e['weight_fuel']
        
        stats = html.Div([
            html.Div(f"ROUTE: {' > '.join(path)}", className="p-2 mb-3 rounded small", 
                     style={'background': '#00e5ff11', 'border': '1px dashed #00e5ff', 'color': '#00e5ff'}),
            dbc.Row([
                dbc.Col(html.Div([html.Small("KM"), html.H4(d_tot)]), width=4),
                dbc.Col(html.Div([html.Small("MIN"), html.H4(t_tot)]), width=4),
                dbc.Col(html.Div([html.Small("LTR"), html.H4(round(f_tot, 1))]), width=4),
            ], className="text-center text-info")
        ])
    except:
        stats = dbc.Alert("NO PATH FOUND", color="danger", className="small")

    fig_data = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        fig_data.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=3, color='#333'), opacity=0.5, hoverinfo='none'
        ))
        fig_data.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=1, color='#555'), opacity=0.5, hoverinfo='none'
        ))

    if path:
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            x0, y0 = pos[u]; x1, y1 = pos[v]
            fig_data.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines', line=dict(width=10, color='rgba(0, 229, 255, 0.2)'),
                hoverinfo='none'
            ))
            fig_data.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines+markers', line=dict(width=4, color='#00e5ff'),
                marker=dict(symbol="arrow-bar-up", angleref="previous", size=10),
                hoverinfo='none'
            ))

    label_x, label_y, label_text = [], [], []
    for u, v, data in G.edges(data=True):
        label_x.append((pos[u][0] + pos[v][0]) / 2)
        label_y.append((pos[u][1] + pos[v][1]) / 2)
        label_text.append(f"{data['weight_dist']}km | {data['weight_fuel']}L")

    fig_data.append(go.Scatter(
        x=label_x, y=label_y, text=label_text, mode='text',
        texttemplate="<b style='background-color:#121212; border:1px solid #444; color:#fff; padding:2px'>%{text}</b>",
        hoverinfo='none'
    ))

    fig_data.append(go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode='markers+text', text=[n for n in G.nodes()],
        textfont=dict(color="#00e5ff", size=10, family="Impact"),
        textposition="top center",
        marker=dict(size=18, color='#000', line=dict(color='#00e5ff', width=2), symbol='hexagon')
    ))

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        transition_duration=500
    )
    
    return fig, stats

if __name__ == '__main__':
    app.run(debug=True)
