import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import no_update
from matplotlib import pyplot as plt
import sympy as sp
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.io as pio
from Class_OOP import DoublePendulum
# Sympy variables for parameters
M1, M2, m1, m2, l1, l2, g = sp.symbols("M1, M2, m1, m2, l1, l2, g", positive=True, real=True)

# Matrix Equations
with open('assets/simple_eqns.txt', 'r') as file:
    simple_matrix = file.read()

with open('assets/compound_eqns.txt', 'r') as file:
    compound_matrix = file.read()


app = dash.Dash(
    __name__,
    external_scripts=[
        # ... (other scripts if any)
        'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'
    ]
)

app.layout = html.Div([
    html.H1("Double Pendulum Simulation", style={'textAlign': 'center', 'color': 'black'}),
    html.Div([
        html.P("'Unity parameters' sets pendulum arms to 1m & masses to 1kg, with g = 9.81m/s."),
        html.P("Ensure all initial conditions and parameters are filled."),
        html.P(" "),
        html.P("Choose the model type from the dropdown menu. 'Compound' models massive rod."),
        html.P(" "),
        html.P("Use the 'Run Simulation' button to start the simulation.")
        # Add more bullet points as required
    ], style={'textAlign': 'left', 'margin': '10px'}),
    # Equation markdown with display block style
    dcc.Markdown(
        simple_matrix,
        mathjax=True
    ),
    html.Div(className='container', children=[
        # Column for model and buttons
        html.Div(className='column', children=[
            html.Div(className='input-group', children=[
                html.Label('Model Type:', className='label'),
                dcc.Dropdown(
                    id='model-type',
                    options=[
                        {'label': 'Simple', 'value': 'simple'},
                        {'label': 'Compound', 'value': 'compound'}
                    ],
                    value='simple'
                ),
            ]),
            html.Div(className='container-buttons', children=[
            html.Button('Unity Parameters', id='unity-parameters', n_clicks=0, className='button'),
            html.Button('Reset', id='reset-button', n_clicks=0, className='button'),
            html.Button('Run Simulation', id='submit-val', n_clicks=0, className='button'),
            ]),
        ]),
        # Column for initial conditions and time vector input
        html.Div(className='column', children=[
            html.Div(className='input-group', children=[
                html.Label('Initial Conditions (θ1, θ2, ω1, ω2): degrees', className='label'),
                dcc.Input(id='init_cond_theta1', type='number', placeholder='Angle 1', className='input'),
                dcc.Input(id='init_cond_theta2', type='number', placeholder='Angle 2', className='input'),
                dcc.Input(id='init_cond_omega1', type='number', placeholder='Angular velocity 1', className='input'),
                dcc.Input(id='init_cond_omega2', type='number', placeholder='Angular velocity 2', className='input'),
            ]),
            html.Div(className='input-group', children=[
                html.Label('Time Vector (start, stop): seconds', className='label'),
                dcc.Input(id='time_start', type='number', placeholder='Start Time', value=0, className='input'),
                dcc.Input(id='time_end', type='number', placeholder='End Time', value=20, className='input'),
            ]),
        ]),
        # Column for parameters inputs
        html.Div(className='column', children=[
            html.Div(className='input-group', children=[
                html.Label('Parameters (l1, l2, m1, m2, M1, M2, g): m, kg, m/s', className='label'),
                dcc.Input(id='param_l1', type='number', placeholder='l1 (length of rod 1)', className='input'),
                dcc.Input(id='param_l2', type='number', placeholder='l2 (length of rod 2)', className='input'),
                dcc.Input(id='param_m1', type='number', placeholder='m1 (mass of bob 1)', className='input'),
                dcc.Input(id='param_m2', type='number', placeholder='m2 (mass of bob 2)', className='input'),
                dcc.Input(id='param_M1', type='number', placeholder='M1 (mass of rod 1)', className='input'),
                dcc.Input(id='param_M2', type='number', placeholder='M2 (mass of rod 2)', className='input'),
                dcc.Input(id='param_g', type='number', placeholder='g (acceleration due to gravity)',
                          className='input'),
            ]),
        ]),
    ]),
    # Graphs
    html.Div(className='above-graph-container', children=[
        dcc.Graph(id='pendulum-animation', className='graph'),
        dcc.Graph(id='phase-graph', className='graph'),
    ]),
    html.Div(className='graph-container', children=[
        dcc.Graph(id='time-graph', className='graph', responsive=True),
    ]),

    # Error message
    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center'}),

    # Markdown container for the matrix equation
    html.Div(
        dcc.Markdown(simple_matrix, mathjax=True),
        className='markdown-latex-container'
    ),

])


# Callback for the unity parameters
@app.callback(
    [Output('param_l1', 'value'),
     Output('param_l2', 'value'),
     Output('param_m1', 'value'),
     Output('param_m2', 'value'),
     Output('param_M1', 'value'),
     Output('param_M2', 'value'),
     Output('param_g', 'value')],
    [Input('unity-parameters', 'n_clicks')],
)
def set_unity_parameters(n_clicks):
    if n_clicks > 0:
        # Return unity values for the parameters, except g which is set to 9.81
        return 1, 1, 1, 1, 1, 1, 9.81
    return dash.no_update  # Prevents updating before button click


# Callback for the reset button
@app.callback(
    [Output('init_cond_theta1', 'value'),
     Output('init_cond_theta2', 'value'),
     Output('init_cond_omega1', 'value'),
     Output('init_cond_omega2', 'value'),
     Output('time_start', 'value'),
     Output('time_end', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_values(n_clicks):
    if n_clicks > 0:
        # Reset all values to default or initial state
        return '', '', '', '', 0, 20
    return dash.no_update


# Callback to update the graphs
@app.callback(
    [Output('time-graph', 'figure'),
     Output('phase-graph', 'figure'),
     Output('pendulum-animation', 'figure'),
     Output('error-message', 'children')],
    [Input('submit-val', 'n_clicks')],
    [State('init_cond_theta1', 'value'),
     State('init_cond_theta2', 'value'),
     State('init_cond_omega1', 'value'),
     State('init_cond_omega2', 'value'),
     State('time_start', 'value'),
     State('time_end', 'value'),
     State('param_l1', 'value'),
     State('param_l2', 'value'),
     State('param_m1', 'value'),
     State('param_m2', 'value'),
     State('param_M1', 'value'),
     State('param_M2', 'value'),
     State('param_g', 'value'),
     State('model-type', 'value')]  # Add the state for the model type dropdown
)
def update_graphs(n_clicks, init_cond_theta1, init_cond_theta2, init_cond_omega1, init_cond_omega2,
                  time_start, time_end,
                  param_l1, param_l2, param_m1, param_m2, param_M1, param_M2, param_g,
                  model_type):
    if n_clicks > 0:
        # Check if any required field is empty
        if any(param is None for param in [init_cond_theta1, init_cond_theta2, init_cond_omega1, init_cond_omega2,
                                           time_start, time_end,
                                           param_l1, param_l2, param_m1, param_m2, param_M1, param_M2, param_g]):
            return no_update, no_update, no_update, "Please fill in all required fields.", ""

        initial_conditions = [init_cond_theta1, init_cond_theta2, init_cond_omega1, init_cond_omega2]
        time_steps = int((time_end - time_start) * 200)  # Calculate time_steps
        time_vector = [time_start, time_end, time_steps]
        parameters = {l1: param_l1, l2: param_l2,
                      m1: param_m1, m2: param_m2,
                      M1: param_M1, M2: param_M2,
                      g: param_g}

        # Create an instance of DoublePendulum
        pendulum = DoublePendulum(parameters, initial_conditions, time_vector, model=model_type)

        # Convert the Matplotlib graphs to Plotly graphs
        matplotlib_time_fig = pendulum.time_graph()
        # Set the layout to be responsive
        time_fig = tls.mpl_to_plotly(matplotlib_time_fig)
        time_fig.update_layout(
            autosize=True,
            margin=dict(l=20, r=20, t=20, b=20),
            width=1400,  # Set the ideal width
            height=700  # Set the ideal height to maintain a 1:1 aspect ratio
        )
        plt.close(matplotlib_time_fig)

        matplotlib_phase_fig = pendulum.phase_path()
        # Set the layout with a fixed aspect ratio for the phase-path graph
        phase_fig = tls.mpl_to_plotly(matplotlib_phase_fig)
        phase_fig.update_layout(
            autosize=True,
            margin=dict(l=20, r=20, t=20, b=20),
            width=700,  # Set the ideal width
            height=700  # Set the ideal height to maintain a 1:1 aspect ratio
        )
        plt.close(matplotlib_phase_fig)

        # Define the layout for your figures
        layout = go.Layout(
            title_font=dict(family='Courier New, monospace', size=16, color='black'),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                titlefont=dict(family='Courier New, monospace', size=14, color='black'),
                showgrid=True,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                titlefont=dict(family='Courier New, monospace', size=14, color='black'),
                showgrid=True,
                gridcolor='lightgrey'
            )
        )

        # Apply the layout to your graphs
        time_fig.update_layout(layout)
        phase_fig.update_layout(layout)

        # Generate the animation figure
        pendulum.precompute_positions()  # Make sure positions are precomputed
        animation_fig = pendulum.animate_pendulum(trace=True)

        return time_fig, phase_fig, animation_fig, ""
    else:
        return go.Figure(), go.Figure(), go.Figure(), ""


if __name__ == '__main__':
    app.run_server(debug=True)