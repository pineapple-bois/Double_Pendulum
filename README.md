# Double Pendulum Equations of Motion

### Lagrangian Formulation

----

![img](Resources/Double_Pendulum.png)

#### The above figure shows simple pendulum suspended from another simple pendulum by a frictionless hinge. 
- Both pendulums move in the same plane. 
- In this system, the rods $OP_1$ and $P_1P_2$ are rigid, massless and inextensible.
- The system has two degrees of freedom and is uniquely determined by the values of $\theta_1$ and $\theta_2$

----

We solve the Euler-Lagrange equations for $\textbf{q} = [\theta_1, \theta_2]$ such that, 

$$
\frac{\text{d}}{\text{d}t}\left(\frac{\partial L}{\partial \dot{\textbf{q}}}\right)-\frac{\partial L}{\partial \textbf{q}}=0
$$

The result is a system of $|\textbf{q}|$ coupled, second-order differential equations

----

The equations are uncoupled by letting $\omega_i = \frac{\text{d}}{\text{d} t}\theta_i$

So $\omega_i$ for $i=1,2$ represents the angular velocity with $\frac{\text{d}}{\text{d} t}\omega_i \equiv \frac{\text{d}^2}{\text{d}^2 t}\theta_i$

[Derivation](https://github.com/pineapple-bois/Double_Pendulum/blob/master/Derivation.ipynb)

----

### [Simulated systems](https://github.com/pineapple-bois/Double_Pendulum/blob/master/Simulation.ipynb)

A few systems illustrating periodic/chaotic behaviour are explored.

The gif below shows chaotic motion with release from rest for a large initial angle $[\theta_1=-105 \degree, \theta_2=105 \degree]$

![img](Resources/Chaotic.gif)

----

#### Next steps

- Creating an interactive app ✅
  - Need to deploy now...
- Deriving equations of motion to include moment of inertia of rod (non-zero mass).
- Quantifying chaotic behaviour.

----