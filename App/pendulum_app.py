import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from matplotlib import pyplot as plt
import sympy as sp
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.io as pio
from Class_OOP import DoublePendulum
M1, M2, m1, m2, l1, l2, g = sp.symbols("M1, M2, m1, m2, l1, l2, g", positive=True, real=True)


app = dash.Dash(__name__)

app.layout = html.Div([
    # Dropdown for model type
    html.Div([
        html.Label('Model Type:'),
        dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'Simple', 'value': 'simple'},
                {'label': 'Compound', 'value': 'compound'}
            ],
            value='simple'  # Default value
        ),
    ], style={'padding': '20px'}),

    # Button for Unity Parameters
    html.Div([
        html.Button('Unity Parameters', id='unity-parameters', n_clicks=0),
    ], style={'padding': '20px'}),

    # Parameters Inputs
    html.Div([
        html.Label('Parameters: (l1, l2, m1, m2, M1, M2, g)'),
        dcc.Input(id='param_l1', type='number', placeholder='l1 (length of rod 1)'),
        dcc.Input(id='param_l2', type='number', placeholder='l2 (length of rod 2)'),
        dcc.Input(id='param_m1', type='number', placeholder='m1 (mass of bob 1)'),
        dcc.Input(id='param_m2', type='number', placeholder='m2 (mass of bob 2)'),
        dcc.Input(id='param_M1', type='number', placeholder='M1 (mass of rod 1)'),
        dcc.Input(id='param_M2', type='number', placeholder='M2 (mass of rod 2)'),
        dcc.Input(id='param_g', type='number', placeholder='g (acceleration due to gravity)'),
    ], style={'padding': '20px'}),

    # Initial Conditions Inputs
    html.Div([
        html.Label('Initial Conditions: (theta1, theta2, omega1, omega2)'),
        dcc.Input(id='init_cond_theta1', type='number', placeholder='theta1'),
        dcc.Input(id='init_cond_theta2', type='number', placeholder='theta2'),
        dcc.Input(id='init_cond_omega1', type='number', placeholder='omega1'),
        dcc.Input(id='init_cond_omega2', type='number', placeholder='omega2'),
    ], style={'padding': '20px'}),

    # Time Vector Inputs
    html.Div([
        html.Label('Time Vector: (start, stop, step)'),
        dcc.Input(id='time_start', type='number', placeholder='Start Time'),
        dcc.Input(id='time_end', type='number', placeholder='End Time'),
        dcc.Input(id='time_steps', type='number', placeholder='Number of Steps'),
    ], style={'padding': '20px'}),

    # Submit Button
    html.Div([
        html.Button('Run Simulation', id='submit-val', n_clicks=0),
    ], style={'padding': '20px'}),

    # Dropdown for Appearance
    html.Div([
        html.Label('Appearance:'),
        dcc.Dropdown(
            id='appearance-dropdown',
            options=[
                {'label': 'Dark', 'value': 'dark'},
                {'label': 'Light', 'value': 'light'}
            ],
            value='dark'  # Default value
        ),
    ], style={'padding': '20px'}),

    # Graph Outputs
    html.Div([
        dcc.Graph(id='time-graph'),
        dcc.Graph(id='phase-graph'),
        dcc.Graph(id='pendulum-animation'),
    ], style={'padding': '20px'}),

])


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


@app.callback(
    [Output('time-graph', 'figure'),
     Output('phase-graph', 'figure'),
     Output('pendulum-animation', 'figure')],
    [Input('submit-val', 'n_clicks')],
    [State('init_cond_theta1', 'value'),
     State('init_cond_theta2', 'value'),
     State('init_cond_omega1', 'value'),
     State('init_cond_omega2', 'value'),
     State('time_start', 'value'),
     State('time_end', 'value'),
     State('time_steps', 'value'),
     State('param_l1', 'value'),
     State('param_l2', 'value'),
     State('param_m1', 'value'),
     State('param_m2', 'value'),
     State('param_M1', 'value'),
     State('param_M2', 'value'),
     State('param_g', 'value'),
     State('appearance-dropdown', 'value'),  # Add state for appearance dropdown
     State('model-type', 'value')]  # Add the state for the model type dropdown
)
def update_graphs(n_clicks, init_cond_theta1, init_cond_theta2, init_cond_omega1, init_cond_omega2,
                  time_start, time_end, time_steps,
                  param_l1, param_l2, param_m1, param_m2, param_M1, param_M2, param_g,
                  appearance_dropdown, model_type):
    if n_clicks > 0:
        initial_conditions = [init_cond_theta1, init_cond_theta2, init_cond_omega1, init_cond_omega2]
        time_vector = [time_start, time_end, time_steps]
        parameters = {l1: param_l1, l2: param_l2,
                      m1: param_m1, m2: param_m2,
                      M1: param_M1, M2: param_M2,
                      g: param_g}

        # Create an instance of DoublePendulum
        pendulum = DoublePendulum(parameters, initial_conditions, time_vector, model=model_type)

        # Convert the Matplotlib graphs to Plotly graphs
        matplotlib_time_fig = pendulum.time_graph()
        time_fig = tls.mpl_to_plotly(matplotlib_time_fig)
        plt.close(matplotlib_time_fig)

        matplotlib_phase_fig = pendulum.phase_path()
        phase_fig = tls.mpl_to_plotly(matplotlib_phase_fig)
        plt.close(matplotlib_phase_fig)

        # Generate the animation figure
        pendulum.precompute_positions()  # Make sure positions are precomputed
        animation_fig = pendulum.animate_pendulum(trace=True, appearance=appearance_dropdown)

        return time_fig, phase_fig, animation_fig
    else:
        return go.Figure(), go.Figure(), go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)