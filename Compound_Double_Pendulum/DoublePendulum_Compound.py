import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from Functions_Compound import *

# Declare variables & constants
t = sp.Symbol("t")
l1, l2, M1, M2, g = sp.symbols('l1 l2 M1 M2 g', real=True, positive=True)
R1, R2 = sp.symbols('R1 R2', real=True, positive=True)

# Declare functions
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
omega1 = sp.Function('omega1')(t)
omega2 = sp.Function('omega2')(t)

# TODO! implement a different method for 'cylindrical rod' - (In functions)

# Form Lagrangian
L_uniform = form_lagrangian(theta1, theta2, l1, l2, M1, M2, g, model='uniform')

# Form EL equations
eq1, eq2 = euler_lagrange_system(L_uniform, theta1, theta2, model='uniform')

# Simplify equations
eqn1, eqn2 = simplify_system_equations(eq1, eq2, model='uniform')

# Extract coefficients
alpha1 = extract_coefficient(eqn1, sp.diff(theta2, t, 2))
alpha2 = extract_coefficient(eqn2, sp.diff(theta1, t, 2))
function_1 = eqn1.rhs
function_2 = eqn2.rhs

# Form matrix equations
RHS_1, RHS_2 = create_matrix_equation(alpha1, alpha2, function_1, function_2)

# Define equations
MAT_EQ, eqn1, eqn2, eqn3, eqn4 = first_order_system(RHS_1, RHS_2)


class DoublePendulum:
    """
    Instantiate a DoublePendulum object.

    Parameters:
    - parameters (dict): A dictionary containing values for system parameters,
                         e.g., {l1: 1.0, l2: 1.0, m1: 1.0, m2: 1.0, g: 9.81}
    - initial_conditions (list): Initial conditions for the pendulum in degrees,
                               [theta1 angle, theta2 angle, omega1 velocity, omega2 velocity]
    - time_vector (numpy.ndarray): Time vector for numerical integration
                               [start, end, step]
    - integrator: SciPy integrator used, default is solve_ivp
    """
    def __init__(self, parameters, initial_conditions, time_vector, integrator=solve_ivp, **integrator_args):
        # Convert initial conditions from degrees to radians
        self.initial_conditions = np.deg2rad(initial_conditions)

        # Time vector
        self.time = np.linspace(time_vector[0], time_vector[1], time_vector[2])

        # Parameters
        self.parameters = parameters

        # Substitute parameters into the equations
        eq1_subst = eqn1.subs(parameters)
        eq2_subst = eqn2.subs(parameters)
        eq3_subst = eqn3.subs(parameters)
        eq4_subst = eqn4.subs(parameters)

        # Lambdify the equations after substitution
        self.eqn1_func = sp.lambdify((theta1, theta2, omega1, omega2, t), eq1_subst, 'numpy')
        self.eqn2_func = sp.lambdify((theta1, theta2, omega1, omega2, t), eq2_subst, 'numpy')
        self.eqn3_func = sp.lambdify((theta1, theta2, omega1, omega2, t), eq3_subst, 'numpy')
        self.eqn4_func = sp.lambdify((theta1, theta2, omega1, omega2, t), eq4_subst, 'numpy')

        self.sol = self._solve_ode(integrator, **integrator_args)

    def _system(self, y, t):
        th1, th2, w1, w2 = y
        system = [
            self.eqn1_func(th1, th2, w1, w2, t),
            self.eqn2_func(th1, th2, w1, w2, t),
            self.eqn3_func(th1, th2, w1, w2, t),
            self.eqn4_func(th1, th2, w1, w2, t)
        ]
        return system

    def _solve_ode(self, integrator, **integrator_args):
        """
        Solve the system of ODEs using the specified integrator.

        Parameters:
        - integrator: The integrator function to use. Default is scipy's odeint.
        - system: The system function defining the ODEs.
        - **integrator_args: Additional arguments specific to the chosen integrator.
        """
        if integrator == odeint:
            sol = odeint(self._system, self.initial_conditions, self.time, **integrator_args)
        elif integrator == solve_ivp:
            t_span = (self.time[0], self.time[-1])
            sol = solve_ivp(lambda t, y: self._system(y, t), t_span, self.initial_conditions,
                            t_eval=self.time, **integrator_args)
            sol = sol.y.T  # Transpose
        else:
            raise ValueError("Unsupported integrator")
        return sol

    def _calculate_positions(self):
        # Unpack solution for theta1 and theta2
        theta_1, theta_2 = self.sol[:, 0], self.sol[:, 1]

        # Evaluate lengths of the pendulum arms using the provided parameter values
        l_1 = float(self.parameters[l1])
        l_2 = float(self.parameters[l2])

        # Calculate the (x, y) positions of the first pendulum bob
        x_1 = l_1 * np.sin(theta_1)
        y_1 = -l_1 * np.cos(theta_1)

        # Calculate the (x, y) positions of the second pendulum bob
        x_2 = x_1 + l_2 * np.sin(theta_2)
        y_2 = y_1 - l_2 * np.cos(theta_2)

        return x_1, y_1, x_2, y_2

    def time_graph(self):
        theta1_deg = np.rad2deg(self.sol[:, 0])
        theta2_deg = np.rad2deg(self.sol[:, 1])

        plt.figure(figsize=(10, 6))
        plt.plot(self.time, theta1_deg, 'b', label="$θ_1$", linewidth=2)
        plt.plot(self.time, theta2_deg, 'g', label="$θ_2$", linewidth=2)
        plt.xlabel('Time / seconds')
        plt.ylabel('Angular displacement / degrees')
        plt.title('Double Pendulum Time Graph')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def phase_path(self):
        theta1_deg = np.rad2deg(self.sol[:, 0])
        theta2_deg = np.rad2deg(self.sol[:, 1])

        plt.figure(figsize=(10, 10))
        plt.plot(theta1_deg, theta2_deg, color='purple', label="Phase Path", linewidth=2)
        plt.xlabel('$θ_1$ / degrees')
        plt.ylabel('$θ_2$ / degrees')
        plt.title('Double Pendulum Phase Path')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def precompute_positions(self):
        """
        Precomputes and stores the positions of both pendulum bobs for each time step.

        This method calculates the (x, y) positions of the first and second pendulum bobs at each time step,
        using the provided initial conditions and system parameters. The positions are stored in a NumPy array
        as an instance attribute, which can be used for plotting and animation purposes, reducing the
        computational load at rendering time.
        """
        self.precomputed_positions = np.array(self._calculate_positions())

    def animate_pendulum(self, trace=False, appearance='dark'):
        """
        Generates an animation for the double pendulum using precomputed positions.

        Parameters:
            trace (bool): If True, show the trace of the pendulum.
            appearance (str): 'dark' for dark mode (default), 'light' for light mode.

        Raises:
            AttributeError: If `precompute_positions` has not been called before animation.

        Returns:
            A Plotly figure object containing the animation.
        """
        # Check if precomputed_positions has been calculated
        if not hasattr(self, 'precomputed_positions') or self.precomputed_positions is None:
            raise AttributeError("Precomputed positions must be calculated before animating. "
                                 "Please call 'precompute_positions' method first.")

        x_1, y_1, x_2, y_2 = self.precomputed_positions

        # Colors
        # Check appearance and set colors
        if appearance == 'dark':
            pendulum_color = 'cyan'  # Lighter color for the pendulum
            trace_color_theta1 = 'salmon'  # Softer red for theta1 trace
            trace_color_theta2 = 'lightgreen'  # Softer green for theta2 trace

        elif appearance == 'light':
            pendulum_color = 'navy'  # Darker color for the pendulum
            trace_color_theta1 = 'tomato'  # Brighter red for theta1 trace
            trace_color_theta2 = 'mediumseagreen'  # Brighter green for theta2 trace

        else:
            print("Invalid appearance setting. Please choose 'dark' or 'light'.")
            return None  # Exit the function if invalid appearance

        # Create figure with initial trace
        fig = go.Figure(
            data=[go.Scatter(
                x=[0, x_1[0], x_2[0]],
                y=[0, y_1[0], y_2[0]],
                mode='lines+markers',
                name='Pendulum',
                line=dict(width=2, color=pendulum_color),
                marker=dict(size=10, color=pendulum_color)
            )]
        )

        # If trace is True, add path traces
        if trace:
            path_1 = go.Scatter(
                x=x_1, y=y_1,
                mode='lines',
                name='Path of P1',
                line=dict(width=1, color=trace_color_theta1),
            )
            path_2 = go.Scatter(
                x=x_2, y=y_2,
                mode='lines',
                name='Path of P2',
                line=dict(width=1, color=trace_color_theta2),
            )
            fig.add_trace(path_1)
            fig.add_trace(path_2)

        # Calculate the max extent based on the precomputed positions
        max_extent = max(
            np.max(np.abs(x_1)),
            np.max(np.abs(y_1)),
            np.max(np.abs(x_2)),
            np.max(np.abs(y_2))
        )

        # Add padding to the max extent
        padding = 0.1 * max_extent  # 10% padding
        axis_range_with_padding = [-max_extent - padding, max_extent + padding]

        # Add frames to the animation
        step = 10
        frames = [go.Frame(data=[go.Scatter(x=[0, x_1[k], x_2[k]], y=[0, y_1[k], y_2[k]],
                                            mode='lines+markers',
                                            line=dict(width=2))])
                  for k in range(0, len(x_1), step)]  # Use a step to reduce the number of frames
        fig.frames = frames

        # Update figure layout to create a square plot with padded axis range
        fig.update_layout(
            xaxis=dict(
                range=axis_range_with_padding,
                autorange=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=axis_range_with_padding,
                autorange=False,
                zeroline=False,
                scaleanchor='x',
                scaleratio=1,
            ),
            autosize=False,
            width=700,  # Use the same size for width and height
            height=700,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True,
                                     "mode": "immediate"}]
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                       "transition": {"duration": 0}}]
                    )
                ],
                'direction': "left",
                'pad': {"r": 10, "t": 87},
                'showactive': False,
                'type': "buttons",
                'x': 0.1,
                'xanchor': "right",
                'y': 0,
                'yanchor': "top"
            }],
            margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins to fit layout
        )

        return fig