import sympy as sp


# Declare variables & constants
t = sp.Symbol("t")
g = sp.Symbol("g", positive=True, real=True)
l1, l2, m1, m2 = sp.symbols("l1, l2, m1, m2", positive=True, real=True)

# declare symbolic variables
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta1_dot = sp.Function('theta1_dot')(t)
theta2_dot = sp.Function('theta2_dot')(t)