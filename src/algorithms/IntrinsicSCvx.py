import cvxpy as cvx
import numpy as np
from ..algorithms import SCvx
from ..models import IntrinsicPoweredDescentModel, IntrinsicPoweredDescentModel_FixedFinalAttitude
from ..parameters import PoweredDescentParameters
from ..utils import conj, quat_mult_matrix, log, quat_frame, frame


class IntrinsicSCvx(SCvx):
    def __init__(
        self,
        params: PoweredDescentParameters,
        system: IntrinsicPoweredDescentModel,
    ) -> None:
        """Initialize IntrinsicSCvx with specific system model.

        Inputs:
            params (PoweredDescentParameters): Configuration parameters for the SCvx algorithm.
            system (IntrinsicPoweredDescentModel): Intrinsic powered descent model.

        Outputs:
            None (sets instance attributes via parent class).
        """
        super().__init__(params, system)
        self.state_dim = system.state_dim
        self.input_dim = system.input_dim

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the cost for a single time step based on state and control deviations.

        Inputs:
            x (np.ndarray): State vector at a single time step, shape (state_dim,).
            u (np.ndarray): Control input at a single time step, shape (input_dim,).

        Outputs:
            float: Stage cost value.
        """
        r, v, w = x[:3], x[3:6], x[10:]
        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        return (r - self.system.r_I_final).T @ Q @ (r - self.system.r_I_final) + \
               (v - self.system.v_I_final).T @ Q @ (v - self.system.v_I_final) + \
               (w - self.system.w_B_final).T @ Q @ (w - self.system.w_B_final) + \
               u.T @ R @ u

    def final_stage_cost(self, x: np.ndarray) -> float:
        """Compute the cost for the final time step based on state deviations.

        Inputs:
            x (np.ndarray): Final state vector, shape (state_dim,).

        Outputs:
            float: Final stage cost value.
        """
        r, v, w = x[:3], x[3:6], x[10:]
        Q = np.eye(3) * self.state_coeff
        return (r - self.system.r_I_final).T @ Q @ (r - self.system.r_I_final) + \
               (v - self.system.v_I_final).T @ Q @ (v - self.system.v_I_final) + \
               (w - self.system.w_B_final).T @ Q @ (w - self.system.w_B_final)

    def trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the total trajectory cost without penalties.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Total trajectory cost.
        """
        cost = 0
        for k in range(self.K):
            cost += self.stage_cost(x[:, k], u[:, k])
        cost += self.final_stage_cost(x[:, self.K])
        return cost

    def cvx_stage_cost(self, x: np.ndarray, u: np.ndarray) -> cvx.Expression:
        """Compute the convex stage cost for use in optimization.

        Inputs:
            x (np.ndarray): State vector at a single time step, shape (state_dim,).
            u (np.ndarray): Control input at a single time step, shape (input_dim,).

        Outputs:
            cvx.Expression: Convex stage cost expression.
        """
        r, v, w = x[:3], x[3:6], x[10:]
        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        return cvx.quad_form(r - self.system.r_I_final, Q) + \
               cvx.quad_form(v - self.system.v_I_final, Q) + \
               cvx.quad_form(w - self.system.w_B_final, Q) + \
               cvx.quad_form(u, R)

    def cvx_final_stage_cost(self, x: np.ndarray) -> cvx.Expression:
        """Compute the convex final stage cost for use in optimization.

        Inputs:
            x (np.ndarray): Final state vector, shape (state_dim,).

        Outputs:
            cvx.Expression: Convex final stage cost expression.
        """
        r, v, w = x[:3], x[3:6], x[10:]
        Q = np.eye(3) * self.state_coeff
        return cvx.quad_form(r - self.system.r_I_final, Q) + \
               cvx.quad_form(v - self.system.v_I_final, Q) + \
               cvx.quad_form(w - self.system.w_B_final, Q)

    def cvx_trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> cvx.Expression:
        """Compute the convex total trajectory cost for use in optimization.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            cvx.Expression: Convex total trajectory cost expression.
        """
        cost = 0
        for k in range(self.K):
            cost += self.cvx_stage_cost(x[:, k], u[:, k])
        cost += self.cvx_final_stage_cost(x[:, self.K])
        return cost

    def cvx_linearized_trajectory_cost(self, x: np.ndarray, u: np.ndarray, eta: cvx.Variable, xi: cvx.Variable, v: cvx.Variable, s: cvx.Variable) -> cvx.Expression:
        """Compute the convex linearized trajectory cost including penalties, adjusted for intrinsic coordinates.

        Inputs:
            x (np.ndarray): Current state trajectory, shape (state_dim, K+1).
            u (np.ndarray): Current control trajectory, shape (input_dim, K).
            eta (cvx.Variable): State perturbation variable, shape (state_dim, K+1).
            xi (cvx.Variable): Control perturbation variable, shape (input_dim, K).
            v (cvx.Variable): Dynamics slack variable, shape (state_dim, K).
            s (cvx.Variable): Constraint slack variable, shape (constraints_dim, K).

        Outputs:
            cvx.Expression: Convex linearized trajectory cost expression.
        """
        eta_adjusted = cvx.Variable((13, self.K + 1))
        eta_adjusted[:6, :] == eta[:6, :]
        eta_adjusted[10:, :] == eta[9:, :]
        eta_adjusted[6:10, :] == 0  # Zero quaternion perturbation
        cost = self.cvx_trajectory_cost(x + eta_adjusted, u + xi)
        for k in range(self.K):
            cost += self.penalty_coeff * cvx.norm(v[:, k], 1)
            cost += self.penalty_coeff * cvx.max(s[:, k])
        return cost

    def linearized_trajectory_cost(self, x: np.ndarray, u: np.ndarray, eta: np.ndarray, xi: np.ndarray, v: np.ndarray, s: np.ndarray) -> float:
        """Compute the linearized trajectory cost including penalties, adjusted for intrinsic coordinates.

        Inputs:
            x (np.ndarray): Current state trajectory, shape (state_dim, K+1).
            u (np.ndarray): Current control trajectory, shape (input_dim, K).
            eta (np.ndarray): State perturbation, shape (state_dim, K+1).
            xi (np.ndarray): Control perturbation, shape (input_dim, K).
            v (np.ndarray): Dynamics slack variable, shape (state_dim, K).
            s (np.ndarray): Constraint slack variable, shape (constraints_dim, K).

        Outputs:
            float: Linearized trajectory cost value.
        """
        eta_adjusted = np.zeros((13, self.K + 1))
        eta_adjusted[:6, :] = eta[:6, :]
        eta_adjusted[10:, :] = eta[9:, :]  # Map intrinsic perturbation to ambient space
        cost = self.trajectory_cost(x + eta_adjusted, u + xi)
        for k in range(self.K):
            cost += self.penalty_coeff * np.linalg.norm(v[:, k], 1)
            cost += self.penalty_coeff * np.max(s[:, k])
        return cost

    def penalized_trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the total trajectory cost with penalties for dynamics and constraints in intrinsic space.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Penalized trajectory cost value.
        """
        cost = self.trajectory_cost(x, u)
        for k in range(self.K):
            xk, uk = x[:, k], u[:, k]
            xkp1 = x[:, k + 1]
            zk = self.system.dynamics(xk, uk)
            sigmak = self.system.constraints(xk, uk)
            e = self.system.inv_retract(xkp1, zk)  # Dynamics error in ambient space
            xkp1_frame = frame(xkp1)
            e_coords = np.linalg.pinv(xkp1_frame) @ e  # Project to intrinsic coordinates
            cost += self.penalty_coeff * np.linalg.norm(e_coords, 1)
            cost += self.penalty_coeff * np.maximum(np.max(sigmak), 0)
        return cost
    


class IntrinsicSCvx_FixedFinalAttitude(IntrinsicSCvx):
    
    def __init__(
        self,
        params: PoweredDescentParameters,
        system: IntrinsicPoweredDescentModel_FixedFinalAttitude,
    ) -> None:
        super().__init__(params, system)
    
    def h(self, q):
        inv_q = conj(q)
        inv_q_mult = quat_mult_matrix(inv_q)
        p = self.system.q_BI_final
        inv_q_p = inv_q_mult@p
        error = log(inv_q_p)
        q_frame = quat_frame(q)
        error_coords = np.linalg.pinv(q_frame)@error
        Q_q = np.eye(3) * self.state_coeff
        return error_coords.T@Q_q@error_coords
    
    def final_h(self, q):
        inv_q = conj(q)
        inv_q_mult = quat_mult_matrix(inv_q)
        p = self.system.q_BI_final
        inv_q_p = inv_q_mult@p
        error = log(inv_q_p)
        q_frame = quat_frame(q)
        error_coords = np.linalg.pinv(q_frame)@error
        Q_q = np.eye(3) * self.final_state_coeff
        return error_coords.T@Q_q@error_coords
    
    def Hess_h_coords(self, q):
        inv_q = conj(q)
        inv_q_mult = quat_mult_matrix(inv_q)
        p = self.system.q_BI_final
        inv_q_p = inv_q_mult@p
        error = log(inv_q_p)
        q_mult = quat_mult_matrix(q)
        q_error = q_mult @ error
        theta = np.linalg.norm(q_error)
        if theta < self.system.TOL:
            return np.eye(3)
        f_theta = theta/np.sin(theta)
        u = q_error/theta
        uu = np.outer(u,u)
        H = uu + f_theta*np.cos(theta)*(np.eye(4) - np.outer(q, q) - uu)
        return self.state_coeff*(q_mult.T@H@q_mult)[1:,1:]
    
    def Hess_final_h_coords(self, q):
        inv_q = conj(q)
        inv_q_mult = quat_mult_matrix(inv_q)
        p = self.system.q_BI_final
        inv_q_p = inv_q_mult@p
        error = log(inv_q_p)
        q_mult = quat_mult_matrix(q)
        q_error = q_mult @ error
        theta = np.linalg.norm(q_error)
        if theta < self.system.TOL:
            return np.eye(3)
        f_theta = theta/np.sin(theta)
        u = q_error/theta
        uu = np.outer(u,u)
        H = uu + f_theta*np.cos(theta)*(np.eye(4) - np.outer(q, q) - uu)
        return self.final_state_coeff*(q_mult.T@H@q_mult)[1:,1:]
        
    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        r, v, q, w = self.system.extract_state(x)
        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        return (r - self.system.r_I_final).T @ Q @ (r - self.system.r_I_final) + \
            (v - self.system.v_I_final).T @ Q @ (v - self.system.v_I_final) + \
            (w - self.system.w_B_final).T @ Q @ (w - self.system.w_B_final) + \
            u.T @ R @ u + \
            self.h(q)

    def final_stage_cost(self, x: np.ndarray) -> float:
        r, v, q, w = self.system.extract_state(x)
        Q = np.eye(3) * self.final_state_coeff
        return (r - self.system.r_I_final).T @ Q @ (r - self.system.r_I_final) + \
            (v - self.system.v_I_final).T @ Q @ (v - self.system.v_I_final) + \
            (w - self.system.w_B_final).T @ Q @ (w - self.system.w_B_final) + \
            self.final_h(q)

    def trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        cost = 0
        for k in range(self.K):
            cost += self.stage_cost(x[:, k], u[:, k])
        cost += self.final_stage_cost(x[:, self.K])
        return cost

    def linearized_trajectory_cost(
            self, 
            x: np.ndarray, u: np.ndarray, 
            eta: np.ndarray, xi: np.ndarray, 
            v: np.ndarray, s: np.ndarray
    ) -> float:
        cost = self.local_cost(x, u, eta, xi)
        for k in range(self.K):
            cost += self.penalty_coeff * np.linalg.norm(v[:, k], 1)
            cost += self.penalty_coeff * np.max(s[:, k])
        return cost
    
    def local_cost(self, x, u, eta, xi):
        cost = 0
        for k in range(self.K):
            xk, uk, etak, xik = x[:,k], u[:,k], eta[:,k], xi[:,k]
            cost += self.local_stage_cost(xk, uk, etak, xik)
        xk, etak = x[:,self.K], eta[:,self.K]
        cost += self.local_final_stage_cost(xk, etak)
        return cost
    
    def local_stage_cost(self, x, u, eta_coords, xi_coords):
        r, v, q, w = self.system.extract_state(x)

        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        cost = (r + eta_coords[:3] - self.system.r_I_final).T @ Q @ (r + eta_coords[:3] - self.system.r_I_final) + \
            (v + eta_coords[3:6] - self.system.v_I_final).T @ Q @ (v + eta_coords[3:6] - self.system.v_I_final) + \
            (w + eta_coords[9:] - self.system.w_B_final).T @ Q @ (w + eta_coords[9:] - self.system.w_B_final) + \
            (u + xi_coords).T @ R @ (u + xi_coords) + \
            self.h(q)

        p = self.system.q_BI_final.copy()
        p_inv = conj(p)
        p_inv_mult = quat_mult_matrix(p_inv)
        quat_error = p_inv_mult @ q
        error = log(quat_error)
        q_mult = quat_mult_matrix(q)
        v = q_mult @ error
        Q_q = np.eye(3) * self.state_coeff
        q_frame = quat_frame(q)
        v_coords = np.linalg.pinv(q_frame)@v
        dphi_dq = 2*Q_q@v_coords
        H = self.Hess_h_coords(q)
        hess_q_phi = eta_coords[6:9].T@H@eta_coords[6:9]
        cost += eta_coords[6:9].T@dphi_dq + hess_q_phi
        return cost

    def local_final_stage_cost(self, x, eta_coords):
        r, v, q, w = self.system.extract_state(x)
        Q = np.eye(3) * self.final_state_coeff
        cost = (r + eta_coords[:3] - self.system.r_I_final).T @ Q @ (r + eta_coords[:3] - self.system.r_I_final) + \
            (v + eta_coords[3:6] - self.system.v_I_final).T @ Q @ (v + eta_coords[3:6] - self.system.v_I_final) + \
            (w + eta_coords[9:] - self.system.w_B_final).T @ Q @ (w + eta_coords[9:] - self.system.w_B_final) + \
            self.final_h(q)
        p = self.system.q_BI_final.copy()
        p_inv = conj(p)
        p_inv_mult = quat_mult_matrix(p_inv)
        quat_error = p_inv_mult @ q
        error = log(quat_error)
        q_mult = quat_mult_matrix(q)
        v = q_mult @ error
        Q_q = np.eye(3) * self.final_state_coeff
        q_frame = quat_frame(q)
        v_coords = np.linalg.pinv(q_frame)@v
        dphi_dq = 2*Q_q@v_coords
        H = self.Hess_final_h_coords(q)
        hess_q_phi = eta_coords[6:9].T@H@eta_coords[6:9]
        cost += eta_coords[6:9].T@dphi_dq + hess_q_phi
        return cost
    
    def cvx_linearized_trajectory_cost(self, x: np.ndarray, u: np.ndarray, eta, xi, v, s) -> float:
        cost = self.cvx_local_cost(x, u, eta, xi)
        for k in range(self.K):
            cost += self.penalty_coeff * cvx.norm(v[:, k], 1)
            cost += self.penalty_coeff * cvx.max(cvx.maximum(0, s[:, k]))
        return cost
    
    def cvx_local_cost(self, x, u, eta, xi):
        cost = 0
        for k in range(self.K):
            xk, uk, etak, xik = x[:,k], u[:,k], eta[:,k], xi[:,k]
            cost += self.cvx_local_stage_cost(xk, uk, etak, xik)
        xk, etak = x[:,self.K], eta[:,self.K]
        cost += self.cvx_local_final_stage_cost(xk, etak)
        return cost
    
    def cvx_local_stage_cost(self, x, u, eta_coords, xi_coords):
        r, v, q, w = self.system.extract_state(x)

        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        cost = cvx.quad_form(r + eta_coords[:3] - self.system.r_I_final, Q) + \
            cvx.quad_form(v + eta_coords[3:6] - self.system.v_I_final, Q) + \
            cvx.quad_form(w + eta_coords[9:] - self.system.w_B_final, Q) + \
            cvx.quad_form(u + xi_coords, R) + \
            self.h(q)

        p = self.system.q_BI_final.copy()
        p_inv = conj(p)
        p_inv_mult = quat_mult_matrix(p_inv)
        quat_error = p_inv_mult @ q
        error = log(quat_error)
        q_mult = quat_mult_matrix(q)
        v = q_mult @ error
        Q_q = np.eye(3) * self.state_coeff
        q_frame = quat_frame(q)
        v_coords = np.linalg.pinv(q_frame)@v
        dphi_dq = 2*Q_q@v_coords
        cost += eta_coords[6:9].T@dphi_dq
        H = self.Hess_h_coords(q)
        cost += cvx.quad_form(eta_coords[6:9], H)
        return cost

    def cvx_local_final_stage_cost(self, x, eta_coords):
        r, v, q, w = self.system.extract_state(x)
        Q = np.eye(3) * self.final_state_coeff
        cost = cvx.quad_form(r + eta_coords[:3] - self.system.r_I_final, Q) + \
            cvx.quad_form(v + eta_coords[3:6] - self.system.v_I_final, Q) + \
            cvx.quad_form(w + eta_coords[9:] - self.system.w_B_final, Q) + \
            self.final_h(q)
        p = self.system.q_BI_final.copy()
        p_inv = conj(p)
        p_inv_mult = quat_mult_matrix(p_inv)
        quat_error = p_inv_mult @ q
        error = log(quat_error)
        q_mult = quat_mult_matrix(q)
        v = q_mult @ error
        Q_q = np.eye(3) * self.final_state_coeff
        q_frame = quat_frame(q)
        v_coords = np.linalg.pinv(q_frame)@v
        dphi_dq = 2*Q_q@v_coords
        cost += eta_coords[6:9].T@dphi_dq
        H = self.Hess_final_h_coords(q)
        cost += cvx.quad_form(eta_coords[6:9], H)
        return cost
    
    def penalized_trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the total trajectory cost with penalties for dynamics and constraints in intrinsic space.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Penalized trajectory cost value.
        """
        cost = self.trajectory_cost(x, u)
        for k in range(self.K):
            xk, uk = x[:, k], u[:, k]
            xkp1 = x[:, k + 1]
            zk = self.system.dynamics(xk, uk)
            sigmak = self.system.constraints(xk, uk)
            e = self.system.inv_retract(xkp1, zk)  # Dynamics error in ambient space
            xkp1_frame = frame(xkp1)
            e_coords = np.linalg.pinv(xkp1_frame) @ e  # Project to intrinsic coordinates
            cost += self.penalty_coeff * np.linalg.norm(e_coords, 1)
            cost += self.penalty_coeff * np.max(np.maximum(sigmak, 0))
        return cost