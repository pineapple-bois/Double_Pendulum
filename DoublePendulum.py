import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Declare variables & constants
t = sp.Symbol("t")
g = sp.Symbol("g", positive=True, real=True)
l1, l2, m1, m2 = sp.symbols("l1, l2, m1, m2", positive=True, real=True)

# Declare symbolic variables
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta1_dot = sp.Function('theta1_dot')(t)
theta2_dot = sp.Function('theta2_dot')(t)

# Coordinates of P1
x1 = l1*sp.sin(theta1)
y1 = -l1*sp.cos(theta1)

# Coordinates of P2
x2 = x1 + l2*sp.sin(theta2)
y2 = y1 - l2*sp.cos(theta2)

# Derivatives w.r.t time
xdot1 = sp.diff(x1, t)
ydot1 = sp.diff(y1, t)

xdot2 = sp.diff(x2, t)
ydot2 = sp.diff(y2, t)


# Kinetic energy function
def kinetic_energy(m, dx, dy, dz):
    T = sp.Rational(1,2)*m*(dx**2 + dy**2 + dz**2)
    return T


# Potential energy function
def potential_energy(m, g, h):
    V = m*g*h
    return V


# For P1
T1 = kinetic_energy(m1, xdot1, ydot1, 0)
V1 = potential_energy(m1, g, y1)

# For P2
T2 = kinetic_energy(m2, xdot2, ydot2, 0)
V2 = potential_energy(m2, g, y2)

# Total kinetic energy of the system
T = T1 + T2
T_simplified = sp.trigsimp(T)
T = T_simplified

# Total potential energy of the system
V = V1 + V2
V_simplified = sp.simplify(V)
V = V_simplified

# Lagrangian
L = T - V


def euler_lagrange_equation(L, q):
    """
    Computes the Euler-Lagrange equation for a given Lagrangian and variable.

    Parameters
    ----------
    L : sympy.Expr
        The Lagrangian of the system, a function that depends on generalized coordinates,
        their derivatives, and time.
    q : sympy.Function
        A function representing a generalized coordinate of the system.

    Returns
    -------
    sympy.Expr
        The Euler-Lagrange equation, a second order differential equation
        describing the dynamics of the system.

    """
    qdot = sp.diff(q, t)  # the derivative of the coordinate with respect to time
    partial_q = sp.trigsimp(sp.diff(L, q))  # partial derivative of L with respect to q
    partial_qdot = sp.trigsimp(sp.diff(L, qdot))  # partial derivative of L with respect to qdot

    eqn = sp.diff(partial_qdot, t) - partial_q  # Euler-Lagrange equation
    simplified_eqn = sp.simplify(eqn)
    return simplified_eqn


theta1_eqn = euler_lagrange_equation(L, theta1)
theta2_eqn = euler_lagrange_equation(L, theta2)

theta1_eq = theta1_eqn / l1
theta2_eq = theta2_eqn / (l2*m2)

eq1 = sp.Eq(theta1_eq, 0)
eq2 = sp.Eq(theta2_eq, 0)


# Isolate second derivative coefficients
def isolate_terms(equation):
    """
    Returns the second derivative terms isolated from the given equation.

    Parameters:
        equation (sympy.Eq): The equation from which to isolate the second derivative terms.

    Returns:
        list: A list containing the isolated second derivative terms.
    """
    # Define derivative terms
    theta1ddot = sp.diff(theta1, t, 2)
    theta2ddot = sp.diff(theta2, t, 2)

    terms = []

    try:
        th1 = sp.Eq(theta1ddot, sp.solve(equation, theta1ddot)[0])
        rhs_eq = th1.rhs
        alpha = sp.together(rhs_eq).as_numer_denom()[1]
        eq_new = sp.Eq(th1.lhs * alpha, th1.rhs * alpha)
        theta1_second = eq_new.lhs
        terms.append(theta1_second)
    except IndexError:
        pass

    try:
        th2 = sp.Eq(theta2ddot, sp.solve(equation, theta2ddot)[0])
        rhs_eq = th2.rhs
        alpha = sp.together(rhs_eq).as_numer_denom()[1]
        eq_new = sp.Eq(th2.lhs * alpha, th2.rhs * alpha)
        theta2_second = eq_new.lhs
        terms.append(theta2_second)
    except IndexError:
        pass

    return terms


# Equation 1
second_derivatives_1 = isolate_terms(eq1)
lhs_1 = second_derivatives_1[0] + second_derivatives_1[1]
rhs_1 = sp.simplify(eq1.lhs - lhs_1)
eqn1 = sp.Eq(lhs_1, rhs_1)

# Equation 2
second_derivatives_2 = isolate_terms(eq2)
lhs_2 = second_derivatives_2[0] + second_derivatives_2[1]
rhs_2 = sp.simplify(eq2.lhs - lhs_2)
eqn2 = sp.Eq(lhs_2, rhs_2)

# Simplify
eqn1_lhs = sp.simplify(eqn1.lhs / (l1*(m1+m2)))
eqn1_lhs = sp.simplify(eqn1_lhs)
eqn1_rhs = sp.simplify(eqn1.rhs / (l1*(m1+m2)))
eqn1_rhs = sp.simplify(eqn1_rhs)
eqn1 = sp.Eq(eqn1_lhs, eqn1_rhs)

eqn2_lhs = sp.simplify(eqn2.lhs / l2)
eqn2_lhs = sp.simplify(eqn2_lhs)
eqn2_rhs = sp.simplify(eqn2.rhs / l2)
eqn2_rhs = sp.simplify(eqn2_rhs)
eqn2 = sp.Eq(eqn2_lhs, eqn2_rhs)


def extract_coefficient(equation, derivative_term):
    """
    Extracts the coefficient, including the denominator, of the specified derivative term from LHS of equation.

    Parameters:
        equation (sympy.Expr): The equation to extract the coefficient from.
        derivative_term (sympy.Expr): The derivative term to extract the coefficient of.

    Returns:
        coeff (sympy.Expr): The coefficient, including the denominator, of the derivative term in the equation.
    """
    lhs_parts = sp.fraction(equation.lhs)
    lhs_coeff_term = lhs_parts[0]
    lhs_denominator_term = lhs_parts[1]
    coeff = lhs_coeff_term.coeff(derivative_term) / lhs_denominator_term

    return coeff


# Let alpha1 be 2nd derivative coefficient of theta2 from eqn1
alpha1 = extract_coefficient(eqn1, sp.diff(theta2, t, 2))
# Let alpha2 be 2nd derivative coefficient of theta1 from eqn2
alpha2 = extract_coefficient(eqn2, sp.diff(theta1, t, 2))
# Declare the RHSs as functions
function_1 = eqn1.rhs
function_2 = eqn2.rhs

# Define symbolic matrix equations
alpha1_func = sp.Function('alpha1')(theta1, theta2)
alpha2_func = sp.Function('alpha2')(theta1, theta2)
function1 = sp.Function('f1')(theta1, theta2, sp.diff(theta1, t), sp.diff(theta2, t))
function2 = sp.Function('f2')(theta1, theta2, sp.diff(theta1, t), sp.diff(theta2, t))

# Differential operators
LHS = sp.Matrix([[sp.diff(theta1, t, 2)], [sp.diff(theta2, t, 2)]])
# Coefficients
A = sp.Matrix([[1, alpha1_func], [alpha2_func, 1]])
RHS = sp.Matrix([[function1], [function2]])

# Invert coefficient matrix
inverse = A.inv()
# Form new RHS
NewRHS = sp.simplify(inverse*RHS)
EQS = sp.Eq(LHS, (-1)*NewRHS)

# Substitute values of the functions
EQS_subst = EQS.subs({alpha1_func: alpha1, alpha2_func: alpha2, function1: function_1, function2: function_2})
RHS_subst = EQS_subst.rhs

# Form equations from Matrix
theta1ddot_eqn = RHS_subst[0]
RHS_1 = sp.simplify(theta1ddot_eqn)

theta2ddot_eqn = RHS_subst[1]
RHS_2 = sp.simplify(theta2ddot_eqn)

# New variables to represent derivatives of theta_i, i=1,2
omega1 = sp.Function('omega1')(t)
omega2 = sp.Function('omega2')(t)

# Cast as first order system
eq1 = sp.Eq(omega1, sp.diff(theta1, t))
eq2 = sp.Eq(omega2, sp.diff(theta2, t))
eq3 = sp.Eq(sp.diff(omega1, t), RHS_1)
eq4 = sp.Eq(sp.diff(omega2, t), RHS_2)

# Substitute omega for derivative of theta
derivative_subs = {sp.Derivative(theta1, t): omega1, sp.Derivative(theta2, t): omega2}

# Isolate rhs of equations
eqn1 = eq1.rhs.subs(derivative_subs)
eqn2 = eq2.rhs.subs(derivative_subs)
eqn3 = eq3.rhs.subs(derivative_subs)
eqn4 = eq4.rhs.subs(derivative_subs)


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
