# Substitute omega for derivative of theta
derivative_subs = {sp.Derivative(theta1, t): omega1, sp.Derivative(theta2, t): omega2}
eqn1 = eqn1.evalf(subs=derivative_subs)
eqn2 = eqn2.evalf(subs=derivative_subs)
eqn3 = eqn3.evalf(subs=derivative_subs)
eqn4 = eqn4.evalf(subs=derivative_subs)
