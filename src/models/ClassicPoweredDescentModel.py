import numpy as np
from typing import List, Optional
from ..parameters import PoweredDescentParameters
from ..models import PoweredDescentModel
import cvxpy as cvx

class ClassicPoweredDescentModel(PoweredDescentModel):
    
    def __init__(self, params: PoweredDescentParameters, cache_file: str = 'ClassicModel_data.pkl', x_init: Optional[np.ndarray] = None):
        """Initialize classic powered descent model.

        Inputs:
            params (PoweredDescentParameters): Model parameters.
            cache_file (str): Path to cache symbolic expressions, default 'ClassicModel_data.pkl'.
            x_init (Optional[np.ndarray]): Initial state, shape (13,), defaults to None (randomized).

        Outputs:
            None
        """
        super().__init__(params=params, cache_file=cache_file, x_init=x_init)

    def A_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state Jacobian.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).

        Outputs:
            np.ndarray: State Jacobian, shape (13, 13).
        """
        return self.lambdified_A_matrix(x, u)

    def B_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute control Jacobian.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).

        Outputs:
            np.ndarray: Control Jacobian, shape (13, 6).
        """
        return self.lambdified_B_matrix(x, u)

    def S_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute constraint Jacobian.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).

        Outputs:
            np.ndarray: Constraint Jacobian, shape (constraint_dim, 13).
        """
        return self.lambdified_S_matrix(x, u)

    def retract_trajectory(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Adjust trajectory with quaternion normalization.

        Inputs:
            x (np.ndarray): State trajectory, shape (13, K+1).
            eta (np.ndarray): Perturbation, shape (13, K+1).

        Outputs:
            np.ndarray: Adjusted trajectory, shape (13, K+1).
        """
        x_retract = x + eta
        x_retract[6:10, :] /= np.linalg.norm(x_retract[6:10, :], axis=0)
        return x_retract

    def get_subproblem_constraints(self, x: np.ndarray, u: np.ndarray, eta: cvx.Variable, xi: cvx.Variable, v: cvx.Variable, s: cvx.Variable, r: float) -> List:
        """Define constraints for optimization subproblem.

        Inputs:
            x (np.ndarray): State trajectory, shape (13, K+1).
            u (np.ndarray): Control trajectory, shape (6, K).
            eta (cvx.Variable): State perturbation variable, shape (13, K+1).
            xi (cvx.Variable): Control perturbation variable, shape (6, K).
            v (cvx.Variable): Dynamics slack variable, shape (13, K).
            s (cvx.Variable): Constraint slack variable, shape (constraint_dim, K).
            r (float): Trust region radius.

        Outputs:
            List: List of cvxpy constraint expressions.
        """
        constraints = [eta[:, 0] == np.zeros(self.n_x)]
        for k in range(self.K):
            xk = x[:, k]
            uk = u[:, k]
            etak = eta[:, k]
            xik = xi[:, k]
            xkp1 = x[:, k + 1]
            etakp1 = eta[:, k + 1]
            vk = v[:, k]
            sk = s[:, k]
            Ak = self.A_matrix(xk, uk)
            Bk = self.B_matrix(xk, uk)
            Sk = self.S_matrix(xk, uk)
            zk = self.dynamics(xk, uk)
            sigma_k = self.constraints(xk, uk)
            constraints += [
                etakp1 + xkp1 == zk + Ak @ etak + Bk @ xik + vk,
                cvx.norm(xik) <= r,
                cvx.norm(etak) <= r,
                sigma_k + Sk @ etak - sk <= 0,
                sk >= 0,
            ]
        constraints += [
            x[:3, -1] + eta[:3, -1] == self.r_I_final,
            x[3:6, -1] + eta[3:6, -1] == self.v_I_final,
            x[10:, -1] + eta[10:, -1] == self.w_B_final,
        ]
        return constraints
    
class ClassicPoweredDescentModel_FixedFinalAttitude(ClassicPoweredDescentModel):
    def __init__(self, params: PoweredDescentParameters, x_init: Optional[np.ndarray] = None):
        """Initialize classic model with fixed final attitude.

        Inputs:
            params (PoweredDescentParameters): Model parameters with q_BI_final.
            x_init (Optional[np.ndarray]): Initial state, shape (13,), defaults to None (randomized).

        Outputs:
            None
        """
        super().__init__(params=params, x_init=x_init)
        self.q_BI_final = params.q_BI_final  # Fixed final quaternion, shape (4,)

    def get_subproblem_constraints(self, x: np.ndarray, u: np.ndarray, eta: cvx.Variable, xi: cvx.Variable, v: cvx.Variable, s: cvx.Variable, r: float) -> List:
        """Define constraints with fixed final attitude.

        Inputs:
            x (np.ndarray): State trajectory, shape (13, K+1).
            u (np.ndarray): Control trajectory, shape (6, K).
            eta (cvx.Variable): State perturbation variable, shape (13, K+1).
            xi (cvx.Variable): Control perturbation variable, shape (6, K).
            v (cvx.Variable): Dynamics slack variable, shape (13, K).
            s (cvx.Variable): Constraint slack variable, shape (constraint_dim, K).
            r (float): Trust region radius.

        Outputs:
            List: List of cvxpy constraint expressions.
        """
        constraints = [eta[:, 0] == np.zeros(self.n_x)]
        for k in range(self.K):
            xk = x[:, k]
            uk = u[:, k]
            etak = eta[:, k]
            xik = xi[:, k]
            xkp1 = x[:, k + 1]
            etakp1 = eta[:, k + 1]
            vk = v[:, k]
            sk = s[:, k]
            Ak = self.A_matrix(xk, uk)
            Bk = self.B_matrix(xk, uk)
            Sk = self.S_matrix(xk, uk)
            zk = self.dynamics(xk, uk)
            sigma_k = self.constraints(xk, uk)
            constraints += [
                etakp1 + xkp1 == zk + Ak @ etak + Bk @ xik + vk,
                cvx.norm(xik) <= r,
                cvx.norm(etak) <= r,
                sigma_k + Sk @ etak - sk <= 0,
                sk >= 0,
            ]
        constraints += [
            x[:3, -1] + eta[:3, -1] == self.r_I_final,
            x[3:6, -1] + eta[3:6, -1] == self.v_I_final,
            x[10:, -1] + eta[10:, -1] == self.w_B_final,
            x[6:10, -1] + eta[6:10, -1] == self.q_BI_final,
        ]
        return constraints