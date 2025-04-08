import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Model import ClassicPoweredDescentModel
import matplotlib.pyplot as plt
from src.parameters import SCvxParameters, PoweredDescentParameters
from src.SCvx import ClassicSCvx

algo_params = SCvxParameters()
system_params = PoweredDescentParameters()

system = ClassicPoweredDescentModel(params=system_params)
algo = ClassicSCvx(algo_params, system)

x, u = system.initialize_trajectory()

x_opt, u_opt, traj_cost_hist, time, penalty_hist = algo.run(x, u, verbose=True)

system.print_trajectory(x_opt, u_opt)
system.animate_trajectory(
    x_opt, u_opt, filename='media/classic_scvx_rocket_descent.gif', show=False
)

plt.figure()
plt.plot(traj_cost_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("traj cost")
plt.title("Classic SCvx")
plt.savefig('media/classic_scvx_traj_cost.pdf')


plt.figure()
plt.semilogy(penalty_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("penalized traj cost")
plt.title("Classic SCvx")
plt.legend()
plt.savefig('media/classic_scvx_penalized_traj_cost.pdf')

plt.close()