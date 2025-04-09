import numpy as np
import sympy as sp
from typing import Tuple, List
from ..parameters import SatelliteParameters
from models import Model
from ..utils import exp, quat_mult_matrix, quat_rotate, dexp, right_quat_mult_matrix, sq_dist, log, conj
import cvxpy as cvx

class SatelliteModel(Model):
    def __init__(self, params: SatelliteParameters, x_init: np.ndarray = None):
        super().__init__(params=params, x_init=x_init)
        self.params = params
        self.x_init = x_init
        self._init_params()

    def _set_random_initial_state(self):
        while True:
            xi = np.random.randn(3)
            xi = xi/np.linalg.norm(xi)
            xi = np.random.rand()*np.pi/2*xi
            x_final = exp(xi)
            y_o = quat_rotate(x_final, self.params.y_b)
            ang = np.arccos(y_o.T@self.params.t_o)
            if ang > self.params.theta_max:
                break
        
        while True:
            xi = -xi + .01*np.random.randn(3)
            xi = xi/np.linalg.norm(xi)
            x_init = exp(xi)
            y_o = quat_rotate(x_init, self.params.y_b)
            ang = np.arccos(y_o.T@self.params.t_o)
            if ang > self.params.theta_max and np.sqrt(sq_dist(x_init, x_final)) < np.pi/2:
                break

        self.x_init = x_init
        self.x_final = x_final

    def _init_params(self):
        self.dt = self.params.dt
        self.K = self.params.K
        self.n_x = self.params.n_x
        self.n_u = self.params.n_u
        self.state_dim = self.params.state_dim
        self.input_dim = self.params.input_dim
        self.t_o = self.params.t_o
        self.y_b = self.params.y_b
        self.theta_max = self.params.theta_max
        self.cos_theta_max = self.params.cos_theta_max
        self.tf = self.params.tf
        self.n_s = 1

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_mult = quat_mult_matrix(x)
        exp_dt_u = exp(self.dt*u)
        return x_mult @ exp_dt_u

    def constraints(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        y_o = quat_rotate(x, self.y_b)
        return np.inner(self.t_o, y_o) - self.cos_theta_max

    def get_trajectory_dynamic_error(self, x: np.ndarray, u: np.ndarray) -> float:
        """Calculate total dynamics error over trajectory.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Total dynamics error.
        """
        K = u.shape[1]
        cost = 0
        for k in range(K):
            xk = x[:, k]
            uk = u[:, k]
            xkp1 = x[:, k + 1]
            cost += np.linalg.norm(xkp1 - self.dynamics(xk, uk))
        return cost

    def get_trajectory_constraints_error(self, x: np.ndarray, u: np.ndarray) -> float:
        """Sum constraint violations over trajectory.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Total constraint violation sum.
        """
        K = u.shape[1]
        cost = 0
        for k in range(K):
            xk = x[:, k]
            uk = u[:, k]
            cost += np.sum(np.maximum(self.constraints(xk, uk), 0))
        return cost

    def A_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p = exp(self.dt*u)
        p_mult = quat_mult_matrix(p)
        return p_mult
    
    def B_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dt_u = self.dt*u
        I = np.eye(3)
        D = np.zeros((3,3))
        for i in range(3):
            D[:,i] = dexp(dt_u, I[:,i])
        x_mult = quat_mult_matrix(x)
        return self.dt*x_mult@D

    def S_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pure_y_b = np.zeros(4)
        pure_y_b[1:] = self.y_b
        pure_t_o = np.zeros(4)
        pure_t_o[1:] = self.t_o
        t_o_mult = quat_mult_matrix(pure_t_o)
        y_b_cross = right_quat_mult_matrix(pure_y_b)
        M_H = np.block([
            [np.zeros((4,4)), y_b_cross.T],
            [y_b_cross, np.zeros((4,4))]
        ])
        Q = np.block([
            [np.eye(4)],
            [1/2*t_o_mult]
        ])
        out = 2*Q.T@M_H@Q@x
        return out

    def Q_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.zeros((1,3))

    def retract_trajectory(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        x_retract = x + eta
        x_retract /= np.linalg.norm(x_retract, axis=0)
        return x_retract

    def initialize_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial state and control trajectories.

        Inputs:
            None

        Outputs:
            Tuple[np.ndarray, np.ndarray]: State trajectory (state_dim, K+1), control trajectory (input_dim, K).
        """
        x = np.zeros((self.n_x, self.K + 1))
        u = np.zeros((self.n_u, self.K))
        x[:,0] = self.x_init
        for k in range(self.K):
            xk = x[:,k]
            if sq_dist(self.q_final, xk) > (np.pi/2)**2:
                raise ValueError("Trajectory is outside geodesic convex space")
            inv_xk = conj(xk)
            uk = .1*log(inv_xk, self.x_final)
            x[:,k + 1] = self.dynamics(xk, uk)
            u[:,k] = uk
        return x, u

    def simulate_trajectory(self, u: np.ndarray) -> np.ndarray:
        """Simulate state trajectory from control inputs.

        Inputs:
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            np.ndarray: Simulated state trajectory, shape (state_dim, K+1).
        """
        K = u.shape[1]
        n_x = len(self.x_init)
        x = np.zeros((n_x, K + 1))
        x[:, 0] = self.x_init
        for k in range(K):
            uk = u[:, k]
            x[:, k + 1] = self.dynamics(x[:, k], uk)
        return x

    def get_subproblem_constraints(self, x: np.ndarray, u: np.ndarray, eta: np.ndarray, xi: np.ndarray, v: np.ndarray, s: np.ndarray, r: int) -> List:
        """Define constraints for optimization subproblem.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).
            eta (np.ndarray): State perturbation, shape (state_dim, K+1).
            xi (np.ndarray): Control perturbation, shape (input_dim, K).
            v (np.ndarray): Dynamics slack, shape (state_dim, K).
            s (np.ndarray): Constraint slack, shape (constraint_dim, K).
            r (int): Trust region radius.

        Outputs:
            List: List of constraint expressions for optimization.
        """
        constraints = [eta[:,0] == np.zeros(self.n_x)]
        for k in range(self.K):
            xk = x[:,k]
            xk_p1 = x[:, k + 1]
            uk = u[:,k]
            etak = eta[:,k]
            etak_p1 = eta[:,k + 1]
            xik = xi[:,k]
            Ak = self.A_matrix(xk, uk)
            Bk = self.B_matrix(xk, uk)
            Sk = self.S_matrix(xk, uk)
            vk = v[:,k]
            sk = s[:,k]
            constraints.extend([
                etak_p1 + xk_p1 == self.dynamics(xk, uk) + Ak@etak + Bk@xik + vk,
                cvx.norm(xik, 2) <= r,
                s(xk, uk) + Sk@etak - sk <= 0,
                sk >= 0
            ])