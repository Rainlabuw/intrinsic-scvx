import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import IntrinsicPoweredDescentModel_FixedFinalAttitude
import matplotlib.pyplot as plt
from src.parameters import PoweredDescentParameters
from src.algorithms import IntrinsicSCvx_FixedFinalAttitude

params = PoweredDescentParameters()

system = IntrinsicPoweredDescentModel_FixedFinalAttitude(params=params)
algo = IntrinsicSCvx_FixedFinalAttitude(params, system)

x, u = system.initialize_trajectory()

x_opt, u_opt, traj_cost_hist, total_time, penalty_hist = algo.run(x, u, verbose=True)

system.print_trajectory(x_opt, u_opt)
system.animate_trajectory(
    x_opt, u_opt, filename='media/intrinsic_scvx_fixed_final_attitude_rocket_descent.gif', show=False
)

plt.figure()
plt.plot(traj_cost_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("traj cost")
plt.title("Intrinsic SCvx")
plt.savefig('media/intrinsic_scvx_fixed_final_attitude_traj_cost.pdf')


plt.figure()
plt.semilogy(penalty_hist)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("penalized traj cost")
plt.title("Intrinsic SCvx")
plt.legend()
plt.savefig('media/intrinsic_scvx_fixed_final_attitude_penalized_traj_cost.pdf')

plt.close()