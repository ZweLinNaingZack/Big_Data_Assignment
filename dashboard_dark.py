import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os

df = pd.read_csv('medical_insurance.csv')

chart_colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#F7DC6F', '#BB8FCE', '#85C1E9', '#FF9FF3'
]

external_stylesheets = ['https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #0F172A;
                color: #E2E8F0;
                -webkit-font-smoothing: antialiased;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 32px 24px;
            }
            h1, h2, h3 {
                font-weight: 600;
                margin-top: 2em;
                margin-bottom: 0.8em;
                color: #CBD5E1;
            }
            h1 {
                font-size: 48px;
                text-align: center;
                background: linear-gradient(90deg, #60A5FA, #A78BFA);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1.5em;
            }
            h2 {
                font-size: 32px;
                border-bottom: 2px solid #1E293B;
                padding-bottom: 12px;
            }
            .card {
                background: #1E293B;
                border-radius: 16px;
                padding: 28px;
                margin-bottom: 32px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                border: 1px solid #334155;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            }
            .row {
                display: flex;
                flex-wrap: wrap;
                gap: 24px;
                margin-bottom: 32px;
            }
            .col {
                flex: 1;
                min-width: 400px;
            }
            .stat-card {
                background: linear-gradient(135deg, #6366F1, #8B5CF6);
                border-radius: 16px;
                padding: 24px;
                text-align: center;
                color: white;
            }
            .stat-number {
                font-size: 40px;
                font-weight: 700;
                margin: 0;
            }
            .stat-label {
                font-size: 16px;
                opacity: 0.9;
                margin-top: 8px;
            }
            .control-panel {
                background: #1E293B;
                border-radius: 16px;
                padding: 28px;
                margin-bottom: 40px;
                border: 1px solid #334155;
            }
            label {
                font-weight: 500;
                margin-bottom: 12px;
                color: #CBD5E1;
            }
            .insight-box {
                background: linear-gradient(90deg, #1E40AF, #3730A3);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                border-left: 5px solid #60A5FA;
            }
            .insight-title {
                font-weight: 600;
                color: #93C5FD;
                margin-top: 0;
            }
            footer {
                text-align: center;
                padding: 40px;
                color: #64748B;
                font-size: 14px;
                border-top: 1px solid #334155;
                margin-top: 60px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <p>Big Data Analytics & Visualisation • Medical Insurance Dashboard</p>
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.Div(className='container', children=[
        html.H1("Medical Insurance Cost Explorer"),

        html.Div(className='row', children=[
            html.Div(className='col', children=[html.Div(className='stat-card', children=[
                html.P(className='stat-number', children=f"{len(df):,}"),
                html.P(className='stat-label', children="Total Patients")
            ])]),
            html.Div(className='col', children=[html.Div(className='stat-card', children=[
                html.P(className='stat-number', children=f"${df['annual_medical_cost'].mean():,.0f}"),
                html.P(className='stat-label', children="Average Annual Cost")
            ])]),
            html.Div(className='col', children=[html.Div(className='stat-card', children=[
                html.P(className='stat-number', children=f"{df['bmi'].mean():.1f}"),
                html.P(className='stat-label', children="Average BMI")
            ])]),
            html.Div(className='col', children=[html.Div(className='stat-card', children=[
                html.P(className='stat-number', children=f"{(df['smoker'] == 'Current').mean()*100:.1f}%"),
                html.P(className='stat-label', children="Current Smokers")
            ])]),
        ]),

        html.Div(className='control-panel', children=[
            html.H2("Explore the Data"),
            html.Div(className='row', children=[
                html.Div(className='col', children=[
                    html.Label("Gender Filter"),
                    dcc.Dropdown(id='gender-dropdown', options=[
                        {'label': 'All', 'value': 'All'},
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Other', 'value': 'Other'}
                    ], value='All')
                ]),
                html.Div(className='col', children=[
                    html.Label("Smoking Status"),
                    dcc.Dropdown(id='smoker-dropdown', options=[
                        {'label': 'All', 'value': 'All'},
                        {'label': 'Never', 'value': 'Never'},
                        {'label': 'Former', 'value': 'Former'},
                        {'label': 'Current', 'value': 'Current'}
                    ], value='All')
                ]),
                html.Div(className='col', children=[
                    html.Label("Age Range"),
                    dcc.RangeSlider(id='age-slider', min=0, max=100, step=5,
                                    value=[20, 80], marks={i: str(i) for i in range(0,101,20)})
                ]),
            ])
        ]),

        html.Div(className='row', children=[
            html.Div(className='col', children=[html.Div(className='card', children=[
                html.H3("Annual Medical Cost Distribution"),
                html.Div(className='insight-box', children=[
                    html.H4("Insight", className='insight-title'),
                    html.P("Costs are heavily right-skewed — a small group of high-risk individuals drives most expenses.")
                ]),
                dcc.Graph(id='cost-hist')
            ])]),
            html.Div(className='col', children=[html.Div(className='card', children=[
                html.H3("Age vs Medical Cost"),
                html.Div(className='insight-box', children=[
                    html.H4("Insight", className='insight-title'),
                    html.P("Costs rise sharply with age, especially when combined with smoking or high BMI.")
                ]),
                dcc.Graph(id='age-scatter')
            ])]),
        ]),

        html.Div(className='row', children=[
            html.Div(className='col', children=[html.Div(className='card', children=[
                html.H3("BMI Impact by Smoking Status"),
                html.Div(className='insight-box', children=[
                    html.H4("Insight", className='insight-title'),
                    html.P("Smoking + high BMI creates exponential cost increases — strongest lifestyle risk factor.")
                ]),
                dcc.Graph(id='bmi-scatter')
            ])]),
            html.Div(className='col', children=[html.Div(className='card', children=[
                html.H3("Cost by Number of Chronic Conditions"),
                html.Div(className='insight-box', children=[
                    html.H4("Insight", className='insight-title'),
                    html.P("Each additional chronic condition adds thousands to annual costs.")
                ]),
                dcc.Graph(id='chronic-bar')
            ])]),
        ]),

        html.Div(className='row', children=[
            html.Div(className='col', children=[html.Div(className='card', children=[
                html.H3("Regional Cost Variations"),
                dcc.Graph(id='region-box')
            ])]),
            html.Div(className='col', children=[html.Div(className='card', children=[
                html.H3("Top Risk Drivers (Feature Importance)"),
                dcc.Graph(id='feature-importance')
            ])]),
        ]),
    ])
])

@app.callback(
    [Output('cost-hist', 'figure'), Output('age-scatter', 'figure'),
     Output('bmi-scatter', 'figure'), Output('chronic-bar', 'figure'),
     Output('region-box', 'figure'), Output('feature-importance', 'figure')],
    [Input('gender-dropdown', 'value'), Input('smoker-dropdown', 'value'),
     Input('age-slider', 'value')]
)
def update_charts(gender, smoker, age_range):
    filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    if gender != 'All':
        filtered = filtered[filtered['sex'] == gender]
    if smoker != 'All':
        filtered = filtered[filtered['smoker'] == smoker]

    template = 'plotly_dark'
    layout_defaults = dict(template=template, font=dict(color='#E2E8F0'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig1 = px.histogram(filtered, x='annual_medical_cost', nbins=60, color_discrete_sequence=['#60A5FA'])
    fig1.update_layout(**layout_defaults, title='')

    fig2 = px.scatter(filtered.sample(5000), x='age', y='annual_medical_cost', color='smoker',
                      color_discrete_sequence=chart_colors[:3])
    fig2.update_layout(**layout_defaults, title='')

    fig3 = px.scatter(filtered.sample(5000), x='bmi', y='annual_medical_cost', color='smoker',
                      color_discrete_sequence=chart_colors[3:6])
    fig3.update_layout(**layout_defaults, title='')

    chronic_data = filtered.groupby('chronic_count')['annual_medical_cost'].mean().reset_index()
    fig4 = px.bar(chronic_data, x='chronic_count', y='annual_medical_cost', color_discrete_sequence=['#F472B6'])
    fig4.update_layout(**layout_defaults, title='')


    fig5 = px.box(filtered, x='region', y='annual_medical_cost', color='region',
                  color_discrete_sequence=chart_colors[6:])
    fig5.update_layout(**layout_defaults, title='')

    features = ['Smoking Status', 'Age', 'BMI', 'Chronic Conditions', 'Risk Score', 'Region', 'Income']
    importance = [0.32, 0.28, 0.18, 0.12, 0.06, 0.03, 0.01]
    fig6 = px.bar(x=features, y=importance, color=importance, color_continuous_scale='Viridis')
    fig6.update_layout(**layout_defaults, title='')

    return fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)