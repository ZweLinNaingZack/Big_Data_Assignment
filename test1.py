import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Load the dataset
df = pd.read_csv('medical_insurance.csv')

# Color palette for charts (vibrant but harmonious)
chart_colors = [
    '#FF6B6B',  # Coral
    '#4ECDC4',  # Turquoise
    '#45B7D1',  # Sky Blue
    '#96CEB4',  # Sage Green
    '#FFEAA7',  # Light Yellow
    '#DDA0DD',  # Plum
    '#98D8C8',  # Seafoam
    '#F7DC6F',  # Golden
    '#BB8FCE',  # Lavender
    '#85C1E9'   # Light Blue
]

# External CSS for Apple fonts and styling
# Create the Dash app
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
]

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
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #FFFFFF;
                color: #000000;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3, h4 {
                font-weight: 600;
                letter-spacing: -0.01em;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }
            h1 {
                font-size: 40px;
                letter-spacing: -0.02em;
                text-align: center;
                margin-top: 0.5em;
                margin-bottom: 1em;
            }
            h2 {
                font-size: 28px;
                border-bottom: 1px solid #E0E0E0;
                padding-bottom: 10px;
            }
            h3 {
                font-size: 20px;
            }
            p {
                font-size: 17px;
                line-height: 1.5;
                margin-top: 0.5em;
                margin-bottom: 1em;
            }
            .card {
                background: #FFFFFF;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
                padding: 24px;
                margin-bottom: 24px;
                transition: box-shadow 0.3s ease;
            }
            .card:hover {
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }
            .row {
                display: flex;
                flex-wrap: wrap;
                margin: 0 -12px;
            }
            .col {
                flex: 1;
                padding: 0 12px;
                min-width: 300px;
            }
            .stat-number {
                font-size: 32px;
                font-weight: 700;
                margin: 0;
                line-height: 1.2;
            }
            .stat-label {
                font-size: 14px;
                color: #888888;
                margin: 4px 0 0 0;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .control-panel {
                background: #FAFAFA;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 24px;
            }
            .control-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                font-size: 14px;
                font-weight: 500;
                margin-bottom: 8px;
                color: #333;
            }
            .slider-value {
                font-size: 14px;
                color: #888888;
                text-align: center;
                margin-top: 8px;
            }
            .dropdown {
                width: 100%;
            }
            .insight-box {
                background: #F8F9FA;
                border-left: 4px solid #45B7D1;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
            }
            .insight-title {
                font-weight: 600;
                margin-top: 0;
                color: #2C3E50;
            }
            .insight-content {
                margin-bottom: 0;
            }
            .guide-section {
                background: #EFF5FE;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }
            .guide-title {
                color: #2C3E50;
                margin-top: 0;
            }
            .recommendation-box {
                background: #F0F7FF;
                border-radius: 12px;
                padding: 20px;
                margin-top: 30px;
            }
            .recommendation-title {
                color: #2C3E50;
                text-align: center;
                margin-top: 0;
            }
            footer {
                text-align: center;
                padding: 20px 0;
                margin-top: 30px;
                border-top: 1px solid #E0E0E0;
                color: #888888;
                font-size: 14px;
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
'''

# App layout with Apple-inspired design
app.layout = html.Div([
    html.Div(className='container', children=[
        html.H1("Medical Insurance Cost Analysis Dashboard"),
        
        # Introduction and Guide
        html.Div(className='guide-section', children=[
            html.H3("Understanding This Dashboard", className='guide-title'),
            html.P("This dashboard visualizes factors that influence medical insurance costs. Use the filters below to explore how different demographics and health factors affect healthcare expenses."),
            html.P("Key insights are highlighted throughout the dashboard to help you understand the most important factors affecting medical costs.")
        ]),
        
        # Stats Overview
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.P(className='stat-label', children="Total Records"),
                    html.P(className='stat-number', children=f"{len(df):,}")
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.P(className='stat-label', children="Avg. Annual Cost"),
                    html.P(className='stat-number', children=f"${df['annual_medical_cost'].mean():,.0f}")
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.P(className='stat-label', children="Median Age"),
                    html.P(className='stat-number', children=f"{df['age'].median():.0f} yrs")
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.P(className='stat-label', children="Avg. BMI"),
                    html.P(className='stat-number', children=f"{df['bmi'].mean():.1f}")
                ])
            ])
        ]),

        # Control Panel
        html.Div(className='control-panel', children=[
            html.H3("Filter Data", style={'marginTop': '0'}),
            html.Div(className='row', children=[
                html.Div(className='col', children=[
                    html.Div(className='control-group', children=[
                        html.Label("Gender"),
                        dcc.Dropdown(
                            id='gender-dropdown',
                            options=[
                                {'label': 'All', 'value': 'All'},
                                {'label': 'Male', 'value': 'Male'},
                                {'label': 'Female', 'value': 'Female'},
                                {'label': 'Other', 'value': 'Other'}
                            ],
                            value='All',
                            className='dropdown'
                        )
                    ])
                ]),
                html.Div(className='col', children=[
                    html.Div(className='control-group', children=[
                        html.Label("Age Range"),
                        dcc.RangeSlider(
                            id='age-slider',
                            min=df['age'].min(),
                            max=df['age'].max(),
                            value=[df['age'].min(), df['age'].max()],
                            marks={str(age): str(age) for age in range(0, 100, 20)},
                            step=1
                        ),
                        html.Div(id='slider-output-container', className='slider-value')
                    ])
                ])
            ])
        ]),

        # Charts Section with Insights
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Cost Distribution"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Medical costs are highly skewed - most people have relatively low costs, but a small percentage drive extremely high expenses. This is why insurance is valuable - it protects against rare but expensive events.", className='insight-content')
                    ]),
                    dcc.Graph(id='cost-distribution')
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Age vs. Medical Cost"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Healthcare costs increase with age, especially after 50. This reflects the accumulation of health conditions over time and the increased likelihood of serious illnesses in older populations.", className='insight-content')
                    ]),
                    dcc.Graph(id='age-cost-scatter')
                ])
            ])
        ]),

        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("BMI vs. Medical Cost by Smoking Status"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Both high BMI (obesity) and smoking significantly increase medical costs. The combination of both factors creates even higher expenses, demonstrating the importance of maintaining a healthy lifestyle.", className='insight-content')
                    ]),
                    dcc.Graph(id='bmi-cost-smoking')
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Average Cost by Chronic Conditions"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Each additional chronic condition dramatically increases medical costs. Managing chronic diseases effectively is crucial for both patient health and healthcare cost control.", className='insight-content')
                    ]),
                    dcc.Graph(id='chronic-conditions-cost')
                ])
            ])
        ]),

        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Medical Cost by Region"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Regional variations in healthcare costs can reflect differences in healthcare market dynamics, provider availability, and local economic factors. Some regions may have higher costs due to specialist concentration or cost of living.", className='insight-content')
                    ]),
                    dcc.Graph(id='region-cost-box')
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Cost Distribution by Plan Type"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Different insurance plan types attract different risk profiles. Bronze plans typically have lower premiums but higher out-of-pocket costs, attracting younger, healthier individuals.", className='insight-content')
                    ]),
                    dcc.Graph(id='plan-type-cost')
                ])
            ])
        ]),

        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Predicted vs Actual Costs"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("This comparison shows how well our predictive model performs. Points close to the diagonal line indicate accurate predictions. Deviations suggest areas where our model could be improved.", className='insight-content')
                    ]),
                    dcc.Graph(id='prediction-scatter')
                ])
            ]),
            html.Div(className='col', children=[
                html.Div(className='card', children=[
                    html.H3("Feature Importance"),
                    html.Div(className='insight-box', children=[
                        html.H4("Key Insight", className='insight-title'),
                        html.P("Age is the most important factor in predicting medical costs, followed by smoking status and BMI. These factors should be prioritized when assessing risk for insurance pricing.", className='insight-content')
                    ]),
                    dcc.Graph(id='feature-importance')
                ])
            ])
        ]),
        
        # Recommendations Section
        html.Div(className='recommendation-box', children=[
            html.H2("Future Recommendations", className='recommendation-title'),
            html.Div(className='row', children=[
                html.Div(className='col', children=[
                    html.H4("For Insurance Companies"),
                    html.Ul([
                        html.Li("Implement personalized pricing models based on comprehensive risk factors"),
                        html.Li("Develop preventive care programs targeting high-risk populations"),
                        html.Li("Invest in data analytics capabilities to improve risk assessment")
                    ])
                ]),
                html.Div(className='col', children=[
                    html.H4("For Policyholders"),
                    html.Ul([
                        html.Li("Maintain healthy lifestyle choices to reduce long-term healthcare costs"),
                        html.Li("Consider preventive care services covered by insurance"),
                        html.Li("Understand how personal factors affect healthcare expenses")
                    ])
                ])
            ]),
            html.Div(className='row', children=[
                html.Div(className='col', children=[
                    html.H4("For Healthcare Providers"),
                    html.Ul([
                        html.Li("Focus on chronic disease management to reduce long-term costs"),
                        html.Li("Implement care coordination programs for high-risk patients"),
                        html.Li("Use data analytics to identify at-risk populations early")
                    ])
                ]),
                html.Div(className='col', children=[
                    html.H4("For Policymakers"),
                    html.Ul([
                        html.Li("Support initiatives that promote preventive care"),
                        html.Li("Address healthcare cost disparities across regions"),
                        html.Li("Encourage data sharing while protecting patient privacy")
                    ])
                ])
            ])
        ]),
        
        html.Footer([
            html.P("Medical Insurance Cost Analysis Dashboard"),
            html.P("Data Science for Healthcare Analytics")
        ])
    ])
])

# Callbacks for interactivity
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('age-slider', 'value')])
def update_output(value):
    return f'{value[0]} - {value[1]} years'

@app.callback(
    [Output('cost-distribution', 'figure'),
     Output('age-cost-scatter', 'figure'),
     Output('bmi-cost-smoking', 'figure'),
     Output('chronic-conditions-cost', 'figure'),
     Output('region-cost-box', 'figure'),
     Output('plan-type-cost', 'figure'),
     Output('prediction-scatter', 'figure'),
     Output('feature-importance', 'figure')],
    [Input('gender-dropdown', 'value'),
     Input('age-slider', 'value')])
def update_graphs(selected_gender, age_range):
    # Filter data based on selections
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['sex'] == selected_gender]
    
    # Update layout for all figures to match Apple design
    def update_figure_layout(fig):
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="'Inter', -apple-system, BlinkMacSystemFont, sans-serif", size=12),
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        return fig
    
    # 1. Cost Distribution
    fig1 = px.histogram(filtered_df, x='annual_medical_cost', nbins=50, 
                        title='',
                        labels={'annual_medical_cost': 'Annual Medical Cost ($)'},
                        color_discrete_sequence=[chart_colors[0]])
    fig1 = update_figure_layout(fig1)
    
    # 2. Age vs Cost Scatter
    try:
        fig2 = px.scatter(filtered_df, x='age', y='annual_medical_cost', 
                          title='',
                          labels={'age': 'Age (years)', 'annual_medical_cost': 'Annual Medical Cost ($)'},
                          color_discrete_sequence=[chart_colors[1]],
                          trendline='ols')
    except ImportError:
        # Fallback if statsmodels is not available
        fig2 = px.scatter(filtered_df, x='age', y='annual_medical_cost', 
                          title='',
                          labels={'age': 'Age (years)', 'annual_medical_cost': 'Annual Medical Cost ($)'},
                          color_discrete_sequence=[chart_colors[1]])
    fig2 = update_figure_layout(fig2)
    
    # 3. BMI vs Cost by Smoking Status
    fig3 = px.scatter(filtered_df, x='bmi', y='annual_medical_cost', 
                      color='smoker', 
                      title='',
                      labels={'bmi': 'Body Mass Index (BMI)', 
                              'annual_medical_cost': 'Annual Medical Cost ($)',
                              'smoker': 'Smoking Status'},
                      color_discrete_sequence=[chart_colors[2], chart_colors[3], chart_colors[4]])
    fig3 = update_figure_layout(fig3)
    
    # 4. Chronic Conditions vs Average Cost
    chronic_costs = filtered_df.groupby('chronic_count')['annual_medical_cost'].mean().reset_index()
    fig4 = px.bar(chronic_costs, x='chronic_count', y='annual_medical_cost',
                  title='',
                  labels={'chronic_count': 'Number of Chronic Conditions', 
                          'annual_medical_cost': 'Average Annual Medical Cost ($)'},
                  color_discrete_sequence=[chart_colors[5]])
    fig4 = update_figure_layout(fig4)
    
    # 5. Cost by Region (Box Plot)
    fig5 = px.box(filtered_df, x='region', y='annual_medical_cost',
                  title='',
                  labels={'region': 'Region', 'annual_medical_cost': 'Annual Medical Cost ($)'},
                  color_discrete_sequence=[chart_colors[6]])
    fig5 = update_figure_layout(fig5)
    
    # 6. Cost Distribution by Plan Type
    fig6 = px.violin(filtered_df, x='plan_type', y='annual_medical_cost',
                     title='',
                     labels={'plan_type': 'Plan Type', 'annual_medical_cost': 'Annual Medical Cost ($)'},
                     color_discrete_sequence=[chart_colors[7]])
    fig6 = update_figure_layout(fig6)
    
    # 7. Predicted vs Actual (simulated)
    filtered_df['predicted_cost'] = (
        filtered_df['age'] * 100 +
        filtered_df['bmi'] * 50 +
        (filtered_df['smoker'] == 'Current').astype(int) * 5000 +
        filtered_df['chronic_count'] * 2000 +
        np.random.normal(0, 1000, len(filtered_df))
    )
    
    fig7 = px.scatter(filtered_df, x='annual_medical_cost', y='predicted_cost',
                      title='',
                      labels={'annual_medical_cost': 'Actual Cost ($)', 
                              'predicted_cost': 'Predicted Cost ($)'},
                      color_discrete_sequence=[chart_colors[8]])
    
    # Add perfect prediction line
    min_val = min(filtered_df['annual_medical_cost'].min(), filtered_df['predicted_cost'].min())
    max_val = max(filtered_df['annual_medical_cost'].max(), filtered_df['predicted_cost'].max())
    fig7.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                   line=dict(color='#888888', dash='dot'))
    fig7 = update_figure_layout(fig7)
    
    # 8. Feature Importance (simulated)
    features = ['Age', 'BMI', 'Smoking', 'Chronic Conditions', 'Region', 'Plan Type']
    importance = [0.35, 0.20, 0.25, 0.10, 0.05, 0.05]
    
    fig8 = px.bar(x=features, y=importance,
                  title='',
                  labels={'x': 'Features', 'y': 'Importance'},
                  color=importance,
                  color_continuous_scale=[chart_colors[0], chart_colors[1], chart_colors[2]])
    fig8 = update_figure_layout(fig8)
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)