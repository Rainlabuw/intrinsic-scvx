# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cvxpy as cvx
import numpy as np
import time
from typing import Tuple, List
from ..models import Model 
from ..parameters import PoweredDescentParameters


# Base class for successive convexification (SCvx) optimization
class SCvx:
    def __init__(
        self,
        params: PoweredDescentParameters,
        system: Model,
    ) -> None:
        """Initialize SCvx solver with parameters and dynamic system model.

        Inputs:
            params (PoweredDescentParameters): Configuration parameters for the SCvx algorithm.
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
        self.state_dim = system.n_x  # State dimension
        self.input_dim = system.n_u  # Control input dimension
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
        eta = cvx.Variable((self.state_dim, self.K + 1))  # State perturbation
        xi = cvx.Variable((self.input_dim, self.K))  # Control perturbation
        v = cvx.Variable((self.state_dim, self.K))  # Dynamics slack variable
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

