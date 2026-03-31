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
    {"from": "IMUS", "to": "NOVELETA", "dist": 10, "time": 15, "fuel": 1.2}
]

G = nx.DiGraph()
for r in route_data:
    G.add_edge(r["from"], r["to"], distance=r["dist"], time=r["time"], fuel=r["fuel"])

pos = nx.spring_layout(G, seed=42)
node_list = sorted(list(G.nodes()))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container(fluid=True, style={'padding': '20px'}, children=[
    dbc.Row([
        dbc.Col(html.H1("Cavite Route Optimizer", className="text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(width=4, children=[
            dbc.Card(dbc.CardBody([
                html.H4("Route Settings", className="card-title"),
                html.Label("Origin"),
                dcc.Dropdown(id='start-node', options=[{'label': n, 'value': n} for n in node_list], value='IMUS', className="mb-3", style={'color': '#000'}),
                html.Label("Destination"),
                dcc.Dropdown(id='end-node', options=[{'label': n, 'value': n} for n in node_list], value='SILANG', className="mb-3", style={'color': '#000'}),
                html.Label("Optimize For"),
                dbc.Select(id='criteria', options=[
                    {'label': 'Shortest Distance', 'value': 'distance'},
                    {'label': 'Fastest Time', 'value': 'time'},
                    {'label': 'Least Fuel Consumption', 'value': 'fuel'}
                ], value='distance', className="mb-4"),
                html.Div(id='stats-output')
            ]), style={'borderRadius': '15px', 'border': '1px solid #00d4ff'})
        ]), 
        dbc.Col(width=8, children=[
            dbc.Card(dbc.CardBody([
                dcc.Graph(id='network-graph', style={'height': '75vh'})
            ]), style={'borderRadius': '15px'})
        ])
    ])
])

@app.callback(
    [Output('network-graph', 'figure'), Output('stats-output', 'children')],
    [Input('start-node', 'value'), Input('end-node', 'value'), Input('criteria', 'value')]
)
def update_graph(start, end, criteria):
    path = []
    try:
        path = nx.shortest_path(G, source=start, target=end, weight=criteria)
        d, t, f = 0, 0, 0
        for i in range(len(path)-1):
            e = G[path[i]][path[i+1]]
            d += e['distance']; t += e['time']; f += e['fuel']
        stats_content = html.Div([
            html.Hr(),
            html.H5("Trip Summary", className="text-info"),
            html.P(f"🛣️ Distance: {d} km"),
            html.P(f"⏱️ Time: {t} mins"),
            html.P(f"⛽ Fuel: {round(f, 2)} Liters"),
            html.Div(style={'backgroundColor': '#333', 'padding': '10px', 'borderRadius': '5px'}, children=[
                html.Small("PATH SEQUENCE:", style={'color': '#aaa'}),
                html.P(" → ".join(path), style={'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
    except:
        stats_content = html.P("No valid path found.", className="text-danger")

    edge_traces = []
    # Lists to hold the positions and text for edge labels
    edge_label_x = []
    edge_label_y = []
    edge_label_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        is_in_path = False
        if path:
            for i in range(len(path)-1):
                if (u, v) == (path[i], path[i+1]): 
                    is_in_path = True
        
        # Draw the line
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=4 if is_in_path else 1, color='#00d4ff' if is_in_path else '#444'),
            hoverinfo='none', mode='lines'
        ))

        # Calculate Midpoint for the label
        edge_label_x.append((x0 + x1) / 2)
        edge_label_y.append((y0 + y1) / 2)
        # Construct the label string: D (Dist), T (Time), F (Fuel)
        label = f"D:{data['distance']} T:{data['time']} F:{data['fuel']}"
        edge_label_text.append(label)

    # Create a trace for the edge labels
    edge_label_trace = go.Scatter(
        x=edge_label_x, y=edge_label_y,
        text=edge_label_text,
        mode='text',
        textposition='middle center',
        textfont=dict(size=9, color='#00d4ff'),
        hoverinfo='none'
    )

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode='markers+text', text=[n for n in G.nodes()],
        textfont=dict(color="white"), textposition="top center",
        marker=dict(size=18, color='#00d4ff', line=dict(color='white', width=1)),
        hoverinfo='text'
    )

    fig = go.Figure(data=edge_traces + [edge_label_trace, node_trace],
                    layout=go.Layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False, margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig, stats_content

if __name__ == '__main__':
    app.run(debug=True)
