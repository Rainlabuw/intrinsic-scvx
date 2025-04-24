import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.models import ClassicPoweredDescentModel_FixedFinalAttitude, IntrinsicPoweredDescentModel_FixedFinalAttitude
import matplotlib.pyplot as plt
from src.parameters import PoweredDescentParameters
from src.algorithms import ClassicSCvx_FixedFinalAttitude, IntrinsicSCvx_FixedFinalAttitude

params = PoweredDescentParameters()

system1 = ClassicPoweredDescentModel_FixedFinalAttitude(params=params)
algo1 = ClassicSCvx_FixedFinalAttitude(params, system1)

x, u = system1.initialize_trajectory()
x_init = system1.x_init.copy()

system2 = IntrinsicPoweredDescentModel_FixedFinalAttitude(params, x_init=x_init)
algo2 = IntrinsicSCvx_FixedFinalAttitude(params, system2)
r = 1
J = algo2.trajectory_cost(x, u)
eta, xi, _, _ = algo2.subproblem(x, u, r)

tspan = np.logspace(-10,0,100)
yspan = []
for t in tspan:
    x_next = system2.retract_trajectory(x, t*eta)
    u_next = u + t*xi
    J_next = algo2.trajectory_cost(x_next, u_next)

    v = np.zeros((12, algo2.K + 1))
    s = np.zeros((algo2.constraints_dim, algo2.K))
    L = algo2.linearized_trajectory_cost(x, u, t*eta, t*xi, v, s)

    aux = np.abs(J_next - L)/t
    print(t, aux)
    yspan.append(aux)
yspan = np.array(yspan)

plt.figure()
plt.loglog(tspan, yspan)
plt.grid()
plt.show()