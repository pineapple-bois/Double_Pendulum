from sympy import symbols, Function, Derivative, sympify, Eq, lambdify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define sympy variables and functions
l1, l2, m1, m2, g, t = symbols("l1, l2, m1, m2, g, t", positive=True, real=True)
theta1 = Function('theta1')(t)
theta2 = Function('theta2')(t)
omega1 = Function('omega1')(t)
omega2 = Function('omega2')(t)

# Define the first order derivatives
theta1dot = Derivative(theta1, t)
theta2dot = Derivative(theta2, t)

# Define the second order derivatives
alpha1 = Derivative(omega1, t)
alpha2 = Derivative(omega2, t)

# Read equations of motion
with open("equations_rhs.txt", "r") as f:
    lines = f.readlines()

# Extract RHS of equation between delimiters
delimiter = "#" * 30
eqn_rhs = [sympify(line.strip()) for line in lines if line.strip() != delimiter]

# Define equations
eqn1 = Eq(eqn_rhs[0], theta1dot)
eqn2 = Eq(eqn_rhs[1], theta2dot)
eqn3 = Eq(alpha1, eqn_rhs[2])
eqn4 = Eq(alpha2, eqn_rhs[3])

eqn_system = [eqn1, eqn2, eqn3, eqn4]
for i, eqn in enumerate(eqn_system):
    print(f"eqn{i+1}: {eqn}\n{type(eqn)}\n")


# Declare DoublePendulum Class
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
        self.eqn1 = eqn1.rhs.evalf(subs=parameters)
        self.eqn2 = eqn2.rhs.evalf(subs=parameters)
        self.eqn3 = eqn3.rhs.evalf(subs=parameters)
        self.eqn4 = eqn4.rhs.evalf(subs=parameters)

        print(self.eqn3)
        print(self.eqn4)

        # Convert initial conditions from degrees to radians
        self.initial_conditions = np.deg2rad(initial_conditions)

        # Time vector
        self.time = time_vector

        # Lambdify the equations individually
        eqn1_func = lambdify((theta1, theta2, omega1, omega2, t),
                             self.eqn1.subs({theta1dot: omega1, theta2dot: omega2}), 'numpy')
        eqn2_func = lambdify((theta1, theta2, omega1, omega2, t),
                             self.eqn2.subs({theta1dot: omega1, theta2dot: omega2}), 'numpy')
        eqn3_func = lambdify((theta1, theta2, omega1, omega2, t), self.eqn3, 'numpy')
        eqn4_func = lambdify((theta1, theta2, omega1, omega2, t), self.eqn4, 'numpy')

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







