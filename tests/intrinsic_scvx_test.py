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

system.print_trajectory(x_opt, u_opt)
system.animate_trajectory(
    x_opt, u_opt, filename='media/intrinsic_scvx_rocket_descent.gif', show=False
)

plt.figure()
plt.plot(traj_cost_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("traj cost")
plt.title("Intrinsic SCvx")
plt.savefig('media/intrinsic_scvx_traj_cost.pdf')


plt.figure()
plt.semilogy(penalty_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("penalized traj cost")
plt.title("Penalized traj cost")
plt.legend()
plt.savefig('media/intrinsic_scvx_penalized_traj_cost.pdf')

plt.close()