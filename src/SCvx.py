import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cvxpy as cvx
import numpy as np
import time
from typing import Tuple, List
from src.Model import Model, ClassicPoweredDescentModel, ClassicPoweredDescentModel_FixedFinalAttitude, IntrinsicPoweredDescentModel, IntrinsicPoweredDescentModel_FixedFinalAttitude
from src.parameters import SCvxParameters
from src.utils import conj, quat_mult_matrix, log, quat_frame, frame

# Base class for successive convexification (SCvx) optimization
class SCvx:
    def __init__(
        self,
        params: SCvxParameters,
        system: Model,
    ) -> None:
        """Initialize SCvx solver with parameters and dynamic system model.

        Inputs:
            params (SCvxParameters): Configuration parameters for the SCvx algorithm.
            system (Model): Dynamic system model providing dynamics and constraints.

        Outputs:
            None (sets instance attributes).
        """
        self.iterations = params.iterations  # Maximum iterations
        self.solver = params.solver  # Convex solver type (e.g., 'ECOS', 'SCS')
        self.verbose_solver = params.verbose_solver  # Solver verbosity flag
        self.r0 = params.r0  # Initial trust region radius
        self.rl = params.rl  # Minimum trust region radius
        self.alpha = params.alpha  # Trust region shrinkage factor
        self.beta = params.beta  # Trust region expansion factor
        self.eps_tol = params.eps_tol  # Convergence tolerance for cost change
        self.rho0 = params.rho0  # Lower threshold for trust region adjustment
        self.rho1 = params.rho1  # Middle threshold for trust region adjustment
        self.rho2 = params.rho2  # Upper threshold for trust region adjustment
        self.state_coeff = params.state_coeff  # State cost weighting
        self.final_state_coeff = params.final_state_coeff  # Final state cost weighting
        self.control_coeff = params.control_coeff  # Control cost weighting
        self.penalty_coeff = params.penalty_coeff  # Penalty weight for constraints/dynamics

        self.system = system  # Dynamic system model
        self.K = system.K  # Number of time steps
        self.state_dim = system.state_dim  # State dimension
        self.input_dim = system.input_dim  # Control input dimension
        self.constraints_dim = system.constraints_dim  # Number of constraints

    def subproblem(self, x: np.ndarray, u: np.ndarray, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the convex subproblem for the current iteration.

        Inputs:
            x (np.ndarray): Current state trajectory, shape (state_dim, K+1).
            u (np.ndarray): Current control trajectory, shape (input_dim, K).
            r (float): Trust region radius.

        Outputs:
            Tuple containing:
                eta (np.ndarray): State perturbation, shape (state_dim, K+1).
                xi (np.ndarray): Control perturbation, shape (input_dim, K).
                v (np.ndarray): Dynamics slack variable, shape (state_dim, K).
                s (np.ndarray): Constraint slack variable, shape (constraints_dim, K).
        """
        eta = cvx.Variable((self.system.state_dim, self.K + 1))  # State perturbation
        xi = cvx.Variable((self.system.input_dim, self.K))  # Control perturbation
        v = cvx.Variable((self.system.state_dim, self.K))  # Dynamics slack variable
        s = cvx.Variable((self.constraints_dim, self.K))  # Constraint slack variable
        constraints = self.system.get_subproblem_constraints(
            x, u, eta, xi, v, s, r
        )  # Get system-specific constraints

        J = self.cvx_linearized_trajectory_cost(x, u, eta, xi, v, s)  # Objective function
        problem = cvx.Problem(cvx.Minimize(J), constraints)  # Define convex problem
        opt_value = problem.solve(solver=self.solver, verbose=self.verbose_solver)  # Solve
        if problem.status == 'optimal_inaccurate':
            print(f"Status: {problem.status}")
            print(f"Objective value: {problem.value}")
            print(f"Solver stats: {problem.solver_stats}")
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Status: {problem.status}")
            print(f"Objective value: {problem.value}")
            print(f"Solver stats: {problem.solver_stats}")
            raise ValueError(f"Problem status not optimal: {problem.status}")

        return eta.value, xi.value, v.value, s.value  # Return optimized variables

    def run(self, x: np.ndarray, u: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float], float, List[float]]:
        """Execute the SCvx algorithm to optimize the trajectory.

        Inputs:
            x (np.ndarray): Initial state trajectory, shape (state_dim, K+1).
            u (np.ndarray): Initial control trajectory, shape (input_dim, K).
            verbose (bool): Flag to enable detailed iteration logging (default: False).

        Outputs:
            Tuple containing:
                x (np.ndarray): Optimized state trajectory, shape (state_dim, K+1).
                u (np.ndarray): Optimized control trajectory, shape (input_dim, K).
                C_hist (List[float]): History of trajectory costs per iteration.
                runtime (float): Total execution time in seconds.
                penalty_hist (List[float]): History of penalty terms per iteration.
        """
        k = 0  # Iteration counter
        r = self.r0  # Current trust region radius
        C_hist = []  # History of trajectory costs
        penalty_hist = []  # History of penalty terms
        start_time = time.time()
        while True:
            if k >= self.iterations:
                print(f"Reached max iterations: {self.iterations}")
                break

            eta, xi, v, s = self.subproblem(x, u, r)  # Solve subproblem

            J = self.penalized_trajectory_cost(x, u)  # Current penalized cost
            x_retract = self.system.retract_trajectory(x, eta)  # Updated state trajectory
            u_retract = u + xi  # Updated control trajectory
            J_trans = self.penalized_trajectory_cost(x_retract, u_retract)  # New penalized cost
            L = self.linearized_trajectory_cost(x, u, eta, xi, v, s)  # Linearized cost

            C = self.trajectory_cost(x, u)  # Current trajectory cost
            penalty = (J - C) / self.penalty_coeff  # Penalty contribution

            Delta_J = J - J_trans  # Actual cost reduction
            if np.abs(Delta_J) < self.eps_tol:  # Check convergence
                break
            Delta_L = J - L  # Predicted cost reduction
            rho_k = Delta_J / Delta_L  # Trust region ratio

            if verbose:  # Print iteration details
                print(
                    f"Step {k}:\n Delta J = {Delta_J:.3f}, " +
                    f"Delta L = {Delta_L:.3f}, rho = {rho_k:.3f}, " +
                    f"log(r) = {np.log10(r):.3e}"
                )
                print(
                    f"J: {J:.3f}, L: = {L:.3f}, " +
                    f"trans J = {J_trans:.3f}, " +
                    f"penalty = {penalty:.7f}, "
                )
                print(
                    f"C = {C:.3f}, " +
                    f"trans C = {self.trajectory_cost(x_retract, u_retract):.3f}"
                )
                print(
                    f"log(v_max) = {np.log10(np.max(np.abs(v))):.3e}, " +
                    f"log(s_max) = {np.max(s):.3e}"
                )
                print(
                    f"log(eta_max) = {np.log10(np.max(np.abs(eta))):.3e}, " +
                    f"log(xi_max) = {np.log10(np.max(xi)):.3e}"
                )
                print()

            if rho_k < self.rho0:  # Poor prediction: shrink trust region
                r = r / self.alpha
            else:  # Accept step
                u = u_retract
                x = x_retract
                if rho_k < self.rho1:  # Moderate prediction: shrink trust region
                    r = r / self.alpha
                elif rho_k >= self.rho2:  # Good prediction: expand trust region
                    r = r * self.beta
                r = max(r, self.rl)  # Enforce minimum radius
                k += 1
                C_hist.append(C)
                penalty_hist.append(J)
            if r < 1e-20:  # Trust region too small: terminate
                print(f"Trust region too small")
                break
        end_time = time.time()
        return x, u, C_hist, end_time - start_time, penalty_hist  # Return optimized trajectory and metrics

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the cost for a single time step.

        Inputs:
            x (np.ndarray): State vector at a single time step, shape (state_dim,).
            u (np.ndarray): Control input at a single time step, shape (input_dim,).

        Outputs:
            float: Stage cost value.
        """
        raise NotImplementedError("Subclasses must implement stage_cost")

    def final_stage_cost(self, x: np.ndarray) -> float:
        """Compute the cost for the final time step.

        Inputs:
            x (np.ndarray): Final state vector, shape (state_dim,).

        Outputs:
            float: Final stage cost value.
        """
        raise NotImplementedError("Subclasses must implement final_stage_cost")

    def trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the total trajectory cost without penalties.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Total trajectory cost.
        """
        raise NotImplementedError("Subclasses must implement trajectory_cost")

    def cvx_stage_cost(self, x: np.ndarray, u: np.ndarray) -> cvx.Expression:
        """Compute the convex stage cost for use in optimization.

        Inputs:
            x (np.ndarray): State vector at a single time step, shape (state_dim,).
            u (np.ndarray): Control input at a single time step, shape (input_dim,).

        Outputs:
            cvx.Expression: Convex stage cost expression.
        """
        raise NotImplementedError("Subclasses must implement cvx_stage_cost")

    def cvx_final_stage_cost(self, x: np.ndarray) -> cvx.Expression:
        """Compute the convex final stage cost for use in optimization.

        Inputs:
            x (np.ndarray): Final state vector, shape (state_dim,).

        Outputs:
            cvx.Expression: Convex final stage cost expression.
        """
        raise NotImplementedError("Subclasses must implement cvx_final_stage_cost")

    def cvx_trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> cvx.Expression:
        """Compute the convex total trajectory cost for use in optimization.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            cvx.Expression: Convex total trajectory cost expression.
        """
        raise NotImplementedError("Subclasses must implement cvx_trajectory_cost")

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
        raise NotImplementedError("Subclasses must implement cvx_linearized_trajectory_cost")

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
        raise NotImplementedError("Subclasses must implement linearized_trajectory_cost")

    def penalized_trajectory_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute the total trajectory cost with penalties for dynamics and constraints.

        Inputs:
            x (np.ndarray): State trajectory, shape (state_dim, K+1).
            u (np.ndarray): Control trajectory, shape (input_dim, K).

        Outputs:
            float: Penalized trajectory cost value.
        """
        raise NotImplementedError("Subclasses must implement penalized_trajectory_cost")

# Classic SCvx implementation for powered descent
class ClassicSCvx(SCvx):
    def __init__(
        self,
        params: SCvxParameters,
        system: ClassicPoweredDescentModel,
    ) -> None:
        """Initialize ClassicSCvx with specific system model.

        Inputs:
            params (SCvxParameters): Configuration parameters for the SCvx algorithm.
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
        params: SCvxParameters,
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

# Intrinsic SCvx implementation with manifold-based optimization
class IntrinsicSCvx(SCvx):
    def __init__(
        self,
        params: SCvxParameters,
        system: IntrinsicPoweredDescentModel,
    ) -> None:
        """Initialize IntrinsicSCvx with specific system model.

        Inputs:
            params (SCvxParameters): Configuration parameters for the SCvx algorithm.
            system (IntrinsicPoweredDescentModel): Intrinsic powered descent model.

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
        params: SCvxParameters,
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