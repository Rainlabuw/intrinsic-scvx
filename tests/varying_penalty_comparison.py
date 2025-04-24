import numpy as np
np.random.seed(100)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import IntrinsicPoweredDescentModel, ClassicPoweredDescentModel
import matplotlib.pyplot as plt
from src.parameters import PoweredDescentParameters
from src.algorithms import IntrinsicSCvx, ClassicSCvx

params = PoweredDescentParameters()
system = IntrinsicPoweredDescentModel(params=params)
algo = IntrinsicSCvx(params, system)
x, u = system.initialize_trajectory()
x_init = x[:,0]

C_hist = []
J_hist = []
penalty_hist = []
time_hist = []
steps_hist = []

cls_C_hist = []
cls_J_hist = []
cls_penalty_hist = []
cls_time_hist = []
cls_steps_hist = []

penalty_coeff_hist = [1e2,1e3,1e4,1e5]


for penalty_coeff in penalty_coeff_hist:
    params.penalty_coeff = penalty_coeff
    
    
    system = ClassicPoweredDescentModel(params=params, x_init=x_init.copy())
    algo = ClassicSCvx(params, system)
    x_opt, u_opt, traj_cost_hist, time, _ = algo.run(x.copy(), u.copy(), verbose=False)

    C = algo.trajectory_cost(x_opt, u_opt)
    J = algo.penalized_trajectory_cost(x_opt, u_opt)
    penalty = (J - C)/params.penalty_coeff
    print("CLASSIC")
    print(f"Penalty Coeff: {params.penalty_coeff}")
    print(f"Traj Cost: {C}")
    print(f"Penalized Traj Cost: {J}")
    print(f"Penalty: {penalty}")
    print(f"Time: {time}")
    print(f"num steps: {len(traj_cost_hist)}")
    print(f"r_final: {x_opt[:3,-1]*1000}")
    print(f"v_final: {x_opt[3:6,-1]*1000}")
    print()

    cls_C_hist.append(C)
    cls_J_hist.append(J)
    cls_penalty_hist.append(penalty)
    cls_time_hist.append(time)
    cls_steps_hist.append(len(traj_cost_hist))
    
    system = IntrinsicPoweredDescentModel(params=params, x_init=x_init.copy())
    algo = IntrinsicSCvx(params, system)
    x_opt, u_opt, traj_cost_hist, time, _ = algo.run(x.copy(), u.copy(), verbose=False)

    C = algo.trajectory_cost(x_opt, u_opt)
    J = algo.penalized_trajectory_cost(x_opt, u_opt)
    penalty = (J - C)/params.penalty_coeff
    print("INTRINSIC")
    print(f"Penalty Coeff: {params.penalty_coeff}")
    print(f"Traj Cost: {C}")
    print(f"Penalized Traj Cost: {J}")
    print(f"Penalty: {penalty}")
    print(f"Time: {time}")
    print(f"num steps: {len(traj_cost_hist)}")
    print(f"r_final: {x_opt[:3,-1]*1000}")
    print(f"v_final: {x_opt[3:6,-1]*1000}")
    print()

    C_hist.append(C)
    J_hist.append(J)
    penalty_hist.append(penalty)
    time_hist.append(time)
    steps_hist.append(len(traj_cost_hist))


    

plt.figure()

plt.subplot(2,2,1)
plt.title("Traj Cost Comparison")
plt.ylabel("Traj Cost")
plt.xlabel("penalty coeff")
plt.semilogx(penalty_coeff_hist, C_hist, label='iSCvx')
plt.semilogx(penalty_coeff_hist, cls_C_hist, label='SCvx')
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.title("Clocktime Comparison")
plt.ylabel("Clocktime (sec)")
plt.xlabel("penalty coeff")
plt.semilogx(penalty_coeff_hist, time_hist, label='iSCvx')
plt.semilogx(penalty_coeff_hist, cls_time_hist, label='SCvx')
plt.grid()
#plt.legend()

plt.subplot(2,2,3)
plt.title("Number of steps Comparison")
plt.ylabel("Number of steps")
plt.xlabel("penalty coeff")
plt.semilogx(penalty_coeff_hist, steps_hist, label='iSCvx')
plt.semilogx(penalty_coeff_hist, cls_steps_hist, label='SCvx')
plt.grid()
#plt.legend()

plt.subplot(2,2,4)
plt.title("Penalty Comparison")
plt.ylabel("penalty")
plt.xlabel("penalty coeff")
plt.loglog(penalty_coeff_hist, penalty_hist, label='iSCvx')
plt.loglog(penalty_coeff_hist, cls_penalty_hist, label='SCvx')
plt.grid()
#plt.legend()

plt.tight_layout()
plt.savefig(f'media/big_comparison_{params.K}.pdf')


plt.close()