import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Declare variables & constants
t = sp.Symbol("t")
g = sp.Symbol("g", positive=True, real=True)
l1, l2, m1, m2 = sp.symbols("l1, l2, m1, m2", positive=True, real=True)

# declare symbolic variables
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta1_dot = sp.Function('theta1_dot')(t)
theta2_dot = sp.Function('theta2_dot')(t)

# coordinates of P1
x1 = l1*sp.sin(theta1)
y1 = -l1*sp.cos(theta1)

# coordinates of P2
x2 = x1 + l2*sp.sin(theta2)
y2 = y1 - l2*sp.cos(theta2)

# derivatives w.r.t time
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
    Extracts the coefficient, including the denominator, of the specified derivative term from the left-hand side of the equation.

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
    Initialize a DoublePendulum object.

    Parameters:
    - parameters (dict): A dictionary containing values for system parameters,
                         e.g., {l1: 1.0, l2: 1.0, m1: 1.0, m2: 1.0, g: 9.81}
    - initial_conditions (list): Initial conditions for the pendulum in degrees,
                               [theta1 angle, theta2 angle, omega1 velocity, omega2 velocity]
    - time_vector (numpy.ndarray): Time vector for numerical integration
                               [start, end, step]
    """
    def __init__(self, parameters, initial_conditions, time_vector, integrator=odeint, **integrator_args):
        self.eqn1 = eqn1.subs(parameters)
        self.eqn2 = eqn2.subs(parameters)
        self.eqn3 = eqn3.subs(parameters)
        self.eqn4 = eqn4.subs(parameters)

        # Convert initial conditions from degrees to radians
        self.initial_conditions = np.deg2rad(initial_conditions)

        # Time vector
        self.time = time_vector

        # Lambdify the equations individually
        eqn1_func = sp.lambdify((theta1, theta2, omega1, omega2, t), self.eqn1, 'numpy')
        eqn2_func = sp.lambdify((theta1, theta2, omega1, omega2, t), self.eqn2, 'numpy')
        eqn3_func = sp.lambdify((theta1, theta2, omega1, omega2, t), self.eqn3, 'numpy')
        eqn4_func = sp.lambdify((theta1, theta2, omega1, omega2, t), self.eqn4, 'numpy')

        # Define a system function for odeint
        def system(y, t):
            th1, th2, w1, w2 = y

            return [
                eqn1_func(th1, th2, w1, w2, t),
                eqn2_func(th1, th2, w1, w2, t),
                eqn3_func(th1, th2, w1, w2, t),
                eqn4_func(th1, th2, w1, w2, t)
            ]

        self.sol = self._solve_ode(integrator, system, **integrator_args)

    def _solve_ode(self, integrator, system, **integrator_args):
        """
        Solve the system of ODEs using the specified integrator.

        Parameters:
        - integrator: The integrator function to use. Default is scipy's odeint.
        - system: The system function defining the ODEs.
        - **integrator_args: Additional arguments specific to the chosen integrator.
        """
        if integrator == odeint:
            # Specific arguments can be passed through integrator_args
            # For example, could pass atol, rtol, etc.
            sol = integrator(system, self.initial_conditions, self.time, **integrator_args)
        else:
            # Add more branches for other integrators as needed
            raise ValueError("Unsupported integrator")

        return sol

    def time_graph(self):
        plt.figure(figsize=(10, 6))

        # We limit y-values between -1 and 1
        plt.plot(self.time, self.sol[:, 0], 'b', label="$θ_1$")
        plt.plot(self.time, self.sol[:, 1], 'g', label="$θ_2$")

        plt.xlabel('Time / seconds')
        plt.ylabel('Angular displacement / degrees')
        plt.title('Double Pendulum Time Graph')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def phase_path(self):
        # Plot the phase space
        plt.figure(figsize=(10, 6))
        plt.plot(self.sol[:, 0], self.sol[:, 1], 'b', label="Phase space")
        plt.xlabel('$θ_1$')
        plt.ylabel('$θ_2$')
        plt.title('Double Pendulum Phase Path')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()