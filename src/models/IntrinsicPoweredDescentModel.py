import numpy as np
from typing import List, Optional
from ..parameters import PoweredDescentParameters
from ..models import PoweredDescentModel
import cvxpy as cvx
from src.utils import frame, retract, inv_retract, d_inv_retract


class IntrinsicPoweredDescentModel(PoweredDescentModel):
    def __init__(self, params: PoweredDescentParameters, cache_file: str = 'IntrinsicModel_data.pkl', x_init: Optional[np.ndarray] = None):
        """Initialize intrinsic powered descent model with geometric retraction.

        Inputs:
            params (PoweredDescentParameters): Model parameters.
            cache_file (str): Path to cache symbolic expressions, default 'IntrinsicModel_data.pkl'.
            x_init (Optional[np.ndarray]): Initial state, shape (13,), defaults to None (randomized).

        Outputs:
            None
        """
        super().__init__(params=params, cache_file=cache_file, x_init=x_init)
        self.retract = lambda x, dx: retract(x, dx)  # Geometric retraction function
        self.inv_retract = lambda x, z: inv_retract(x, z)  # Inverse retraction function

    def S_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute constraint Jacobian in intrinsic coordinates.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).

        Outputs:
            np.ndarray: Constraint Jacobian, shape (constraint_dim, 12).
        """
        S = self.lambdified_S_matrix(x, u)
        x_frame = frame(x)  # Assumed to return basis for tangent space, shape (13, 12)
        return S @ x_frame

    def D_matrix(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute differential of inverse retraction.

        Inputs:
            x (np.ndarray): Reference state, shape (13,).
            z (np.ndarray): Target state, shape (13,).

        Outputs:
            np.ndarray: Differential matrix, shape (13, 12).
        """
        qx = x[6:10]
        qz = z[6:10]
        if np.linalg.norm(qx - qz) < self.TOL:  # TOL assumed as small float tolerance
            return np.eye(self.state_dim)
        x_frame = frame(x)
        z_frame = frame(z)
        D = np.zeros((13, 12))
        for j in range(self.state_dim):
            gj = z_frame[:, j]
            D[:, j] = d_inv_retract(x, z, gj)  # Assumed helper function
        return x_frame.T @ D

    def A_matrix(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Compute state Jacobian in intrinsic coordinates.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).
            x_next (np.ndarray): Next state, shape (13,).

        Outputs:
            np.ndarray: State Jacobian, shape (12, 12).
        """
        x_frame = frame(x)
        z = self.dynamics(x, u)
        z_frame = frame(z)
        A = self.lambdified_A_matrix(x, u)
        A_coords = z_frame.T @ A @ x_frame
        D = self.D_matrix(x_next, z)
        return D @ A_coords

    def B_matrix(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Compute control Jacobian in intrinsic coordinates.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).
            x_next (np.ndarray): Next state, shape (13,).

        Outputs:
            np.ndarray: Control Jacobian, shape (12, 6).
        """
        B = self.lambdified_B_matrix(x, u)
        x_frame = frame(x)
        B_coords = x_frame.T @ B
        z = self.dynamics(x, u)
        D = self.D_matrix(x_next, z)
        return D @ B_coords

    def retract_trajectory(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Adjust trajectory using geometric retraction.

        Inputs:
            x (np.ndarray): State trajectory, shape (13, K+1).
            eta (np.ndarray): Perturbation in intrinsic coords, shape (12, K+1).

        Outputs:
            np.ndarray: Adjusted trajectory, shape (13, K+1).
        """
        out = np.zeros(x.shape)
        for k in range(self.K + 1):
            xk = x[:, k]
            etak_coords = eta[:, k]
            xk_frame = frame(xk)
            etak = xk_frame @ etak_coords
            out[:, k] = self.retract(xk, etak)
        return out

    def get_subproblem_constraints(self, x: np.ndarray, u: np.ndarray, eta: cvx.Variable, xi: cvx.Variable, v: cvx.Variable, s: cvx.Variable, r: float) -> List:
        """Define constraints for optimization subproblem in intrinsic coords.

        Inputs:
            x (np.ndarray): State trajectory, shape (13, K+1).
            u (np.ndarray): Control trajectory, shape (6, K).
            eta (cvx.Variable): State perturbation variable, shape (12, K+1).
            xi (cvx.Variable): Control perturbation variable, shape (6, K).
            v (cvx.Variable): Dynamics slack variable, shape (12, K).
            s (cvx.Variable): Constraint slack variable, shape (constraint_dim, K).
            r (float): Trust region radius.

        Outputs:
            List: List of cvxpy constraint expressions.
        """

        constraints = [eta[:, 0] == np.zeros(self.state_dim)]
        for k in range(self.K):
            xk = x[:, k]
            uk = u[:, k]
            etak = eta[:, k]
            xik = xi[:, k]
            xkp1 = x[:, k + 1]
            etakp1 = eta[:, k + 1]
            vk = v[:, k]
            sk = s[:, k]
            Ak = self.A_matrix(xk, uk, xkp1)
            Bk = self.B_matrix(xk, uk, xkp1)
            Sk = self.S_matrix(xk, uk)
            zk = self.inv_retract(xkp1, self.dynamics(xk, uk))
            xkp1_frame = frame(xkp1)
            zk_coords = np.linalg.pinv(xkp1_frame) @ zk
            sigma_k = self.constraints(xk, uk)
            constraints += [
                etakp1 == zk_coords + Ak @ etak + Bk @ xik + vk,
                cvx.norm(xik) <= r,
                cvx.norm(etak) <= r,
                sigma_k + Sk @ etak - sk <= 0,
                sk >= 0,
            ]
        constraints += [
            x[:3, -1] + eta[:3, -1] == self.r_I_final,
            x[3:6, -1] + eta[3:6, -1] == self.v_I_final,
            x[10:, -1] + eta[9:, -1] == self.w_B_final,
        ]
        return constraints



class IntrinsicPoweredDescentModel_FixedFinalAttitude(IntrinsicPoweredDescentModel):
    def __init__(self, params: PoweredDescentParameters, x_init: Optional[np.ndarray] = None):
        """Initialize intrinsic model with fixed final attitude.

        Inputs:
            params (PoweredDescentParameters): Model parameters with q_BI_final.
            x_init (Optional[np.ndarray]): Initial state, shape (13,), defaults to None (randomized).

        Outputs:
            None
        """
        super().__init__(params=params, x_init=x_init)
        self.q_BI_final = params.q_BI_final  # Fixed final quaternion, shape (4,)
    
