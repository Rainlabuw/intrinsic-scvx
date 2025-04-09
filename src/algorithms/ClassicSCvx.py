import cvxpy as cvx
import numpy as np
from ..algorithms import SCvx
from ..models import ClassicPoweredDescentModel, ClassicPoweredDescentModel_FixedFinalAttitude
from ..parameters import PoweredDescentParameters

class ClassicSCvx(SCvx):
    def __init__(
        self,
        params: PoweredDescentParameters,
        system: ClassicPoweredDescentModel,
    ) -> None:
        """Initialize ClassicSCvx with specific system model.

        Inputs:
            params (PoweredDescentParameters): Configuration parameters for the SCvx algorithm.
            system (ClassicPoweredDescentModel): Classic powered descent model.

        Outputs:
            None (sets instance attributes via parent class).
        """
        super().__init__(params, system)

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
        """Compute the convex linearized trajectory cost including penalties.

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
        cost = self.cvx_trajectory_cost(x + eta, u + xi)
        for k in range(self.K):
            cost += self.penalty_coeff * cvx.norm(v[:, k], 1)
            cost += self.penalty_coeff * cvx.max(s[:, k])
        return cost

    def linearized_trajectory_cost(self, x: np.ndarray, u: np.ndarray, eta: np.ndarray, xi: np.ndarray, v: np.ndarray, s: np.ndarray) -> float:
        """Compute the linearized trajectory cost including penalties.

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
        cost = self.trajectory_cost(x + eta, u + xi)
        for k in range(self.K):
            cost += self.penalty_coeff * np.linalg.norm(v[:, k], 1)
            cost += self.penalty_coeff * np.max(s[:, k])
        return cost

    def penalized_trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the total trajectory cost with penalties for dynamics and constraints.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Penalized trajectory cost value.
        """
        cost = self.trajectory_cost(x, u)
        for k in range(self.K):
            xk, uk = x[:, k], u[:, k]
            x_kp1 = x[:, k + 1]
            zk = self.system.dynamics(xk, uk)
            sigmak = self.system.constraints(xk, uk)
            cost += self.penalty_coeff * np.linalg.norm(x_kp1 - zk, 1)
            cost += self.penalty_coeff * np.maximum(np.max(sigmak), 0)
        return cost
    

class ClassicSCvx_FixedFinalAttitude(ClassicSCvx):
    def __init__(
        self,
        params: PoweredDescentParameters,
        system: ClassicPoweredDescentModel_FixedFinalAttitude,
    ) -> None:
        super().__init__(params, system)

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        r, v, q, w = self.system.extract_state(x)
        Q_q = np.eye(4) * self.state_coeff
        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        return (r - self.system.r_I_final).T @ Q @ (r - self.system.r_I_final) + \
               (v - self.system.v_I_final).T @ Q @ (v - self.system.v_I_final) + \
               (q - self.system.q_BI_final).T @ Q_q @ (q - self.system.q_BI_final) + \
               (w - self.system.w_B_final).T @ Q @ (w - self.system.w_B_final) + \
               u.T @ R @ u

    def final_stage_cost(self, x: np.ndarray) -> float:
        """Compute the cost for the final time step based on state deviations.

        Inputs:
            x (np.ndarray): Final state vector, shape (state_dim,).

        Outputs:
            float: Final stage cost value.
        """
        r, v, q, w = self.system.extract_state(x)
        Q_q = np.eye(4) * self.state_coeff
        Q = np.eye(3) * self.state_coeff
        return (r - self.system.r_I_final).T @ Q @ (r - self.system.r_I_final) + \
               (v - self.system.v_I_final).T @ Q @ (v - self.system.v_I_final) + \
               (q - self.system.q_BI_final).T @ Q_q @ (q - self.system.q_BI_final) + \
               (w - self.system.w_B_final).T @ Q @ (w - self.system.w_B_final)

    def cvx_stage_cost(self, x: np.ndarray, u: np.ndarray) -> cvx.Expression:
        """Compute the convex stage cost for use in optimization.

        Inputs:
            x (np.ndarray): State vector at a single time step, shape (state_dim,).
            u (np.ndarray): Control input at a single time step, shape (input_dim,).

        Outputs:
            cvx.Expression: Convex stage cost expression.
        """
        r, v, q, w = self.system.extract_state(x)
        Q_q = np.eye(4) * self.state_coeff
        Q = np.eye(3) * self.state_coeff
        R = np.eye(self.input_dim) * self.control_coeff
        return cvx.quad_form(r - self.system.r_I_final, Q) + \
               cvx.quad_form(v - self.system.v_I_final, Q) + \
               cvx.quad_form(q - self.system.q_BI_final, Q_q) + \
               cvx.quad_form(w - self.system.w_B_final, Q) + \
               cvx.quad_form(u, R)

    def cvx_final_stage_cost(self, x: np.ndarray) -> cvx.Expression:
        """Compute the convex final stage cost for use in optimization.

        Inputs:
            x (np.ndarray): Final state vector, shape (state_dim,).

        Outputs:
            cvx.Expression: Convex final stage cost expression.
        """
        r, v, q, w = self.system.extract_state(x)
        Q_q = np.eye(4) * self.state_coeff
        Q = np.eye(3) * self.state_coeff
        return cvx.quad_form(r - self.system.r_I_final, Q) + \
               cvx.quad_form(v - self.system.v_I_final, Q) + \
               cvx.quad_form(q - self.system.q_BI_final, Q_q) + \
               cvx.quad_form(w - self.system.w_B_final, Q)