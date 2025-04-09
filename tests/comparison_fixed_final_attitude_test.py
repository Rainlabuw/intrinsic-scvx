import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ClassicPoweredDescentModel_FixedFinalAttitude, IntrinsicPoweredDescentModel_FixedFinalAttitude
import matplotlib.pyplot as plt
from src.parameters import PoweredDescentParameters
from src.algorithms import ClassicSCvx_FixedFinalAttitude, IntrinsicSCvx_FixedFinalAttitude

params = PoweredDescentParameters()

system1 = ClassicPoweredDescentModel_FixedFinalAttitude(params=params)
algo1 = ClassicSCvx_FixedFinalAttitude(params, system1)

x, u = system1.initialize_trajectory()
x_init = system1.x_init.copy()

x1_opt, u1_opt, traj_cost_hist_1, time1, penalty_hist_1 = algo1.run(
    x.copy(), u.copy(), verbose=True
)

system1.animate_trajectory(
    x1_opt, u1_opt, filename="media/classic_scvx_fixed_final_attitude_rocket_descent.gif", show=False
)

system2 = IntrinsicPoweredDescentModel_FixedFinalAttitude(params, x_init=x_init)
algo2 = IntrinsicSCvx_FixedFinalAttitude(params, system2)

x2_opt, u2_opt, traj_cost_hist_2, time2, penalty_hist_2 = algo2.run(
    x.copy(), u.copy(), verbose=True
)

system2.animate_trajectory(
    x2_opt, u2_opt, filename="media/intrinsic_scvx_fixed_final_attitude_rocket_descent.gif", show=False
)

print("System 1")
system1.print_trajectory(x1_opt, u1_opt)

print("System 2")
system2.print_trajectory(x2_opt, u2_opt)

print("System 1")
print("Eucl traj cost: ", algo1.trajectory_cost(x1_opt, u1_opt))
print("Geo traj cost: ", algo2.trajectory_cost(x1_opt, u1_opt))
print("traj dynamic total: ", system1.get_trajectory_dynamic_error(x1_opt, u1_opt))
print("traj constraints total: ", system1.get_trajectory_constraints_error(x1_opt, u1_opt))
print("time: ", time1)
print("final state: ", x1_opt[:,-1])
print()
print("System 2")
print("Eucl traj cost: ", algo1.trajectory_cost(x2_opt, u2_opt))
print("Geo traj cost: ", algo2.trajectory_cost(x2_opt, u2_opt))
print("traj dynamic total: ", system2.get_trajectory_dynamic_error(x2_opt, u2_opt))
print("traj constraints total: ", system2.get_trajectory_constraints_error(x2_opt, u2_opt))
print("time: ", time2)
print("final state: ", x2_opt[:,-1])


plt.figure()
plt.plot(traj_cost_hist_1, label='classic')
plt.plot(traj_cost_hist_2, label='intrinsic')
plt.grid()
plt.xlabel("iteration")
plt.ylabel("traj cost")
plt.title("Traj cost Comparison")
plt.legend()
plt.savefig('media/fixed_final_attitude_traj_cost_comparison.pdf')

plt.figure()
plt.semilogy(penalty_hist_1, label='classic')
plt.semilogy(penalty_hist_2, label='intrinsic')
plt.grid()
plt.xlabel("iteration")
plt.ylabel("penalized traj cost")
plt.title("Penalized Traj cost comparison")
plt.legend()
plt.savefig('media/fixed_final_attitude_penalized_traj_cost_comparison.pdf')

plt.close()