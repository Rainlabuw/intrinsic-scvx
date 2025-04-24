import numpy as np
np.random.seed(100)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import IntrinsicPoweredDescentModel
import matplotlib.pyplot as plt
from src.parameters import PoweredDescentParameters
from src.algorithms import IntrinsicSCvx

params = PoweredDescentParameters()

system = IntrinsicPoweredDescentModel(params=params)

algo = IntrinsicSCvx(params, system)
x, u = system.initialize_trajectory()
x_opt, u_opt, traj_cost_hist, time, penalty_hist = algo.run(x, u, verbose=True)

#system.print_trajectory(x_opt, u_opt)
system.animate_trajectory(
    x_opt, u_opt, filename='media/intrinsic_scvx_rocket_descent.gif', show=False
)

print("Traj Cost: ", algo.trajectory_cost(x_opt, u_opt))
print("Penalized Traj Cost: ", algo.penalized_trajectory_cost(x_opt, u_opt))
print("Penalty: ", (algo.penalized_trajectory_cost(x_opt, u_opt) - algo.trajectory_cost(x_opt, u_opt))/params.penalty_coeff)
print("Time: ", time)
print("x_final: ", x_opt[:,-1])
print()


plt.figure()

plt.subplot(2,1,1)
plt.plot(traj_cost_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("traj cost")
plt.title("Intrinsic SCvx")


plt.subplot(2,1,2)
plt.semilogy(penalty_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("penalty")
plt.title("penalty")


plt.tight_layout()
plt.savefig('media/intrinsic_scvx_test.pdf')


plt.close()