from dataclasses import dataclass
import numpy as np

@dataclass
class SatelliteParameters:
    tf: float = 60.0
    dt: float = 1
    TOL: float = 1e-8 
    K = int(tf / dt)  # Number of time steps
    n_x = 4
    n_u = 3
    state_dim = 3
    t_o = np.array([1,0,0])
    y_b = np.array([1,0,0])
    theta_max = np.deg2rad(30)
    cos_theta_max = np.cos(theta_max)

    # algorithm parameters
    penalty_lambda = 1e5
    state_lambda = 1
    control_lambda = .1
    final_state_lambda = 10

    r0 = 1
    rl = 0
    alpha = 2
    beta = 3.2
    eps_tol = 1e-5
    rho0 = 0
    rho1 = .25
    rho2 = .7

@dataclass
class PoweredDescentParameters:
    # Physical parameters in SI units (meters, seconds, etc.)
    tf: float = 60.0  # Final time for trajectory
    dt: float = 1  # Time step size
    TOL: float = 1e-8  # Numerical tolerance
    K = int(tf / dt)  # Number of time steps

    # Model scaling and dimensions
    r_scale: float = 1000.0  # Distance scaling factor (1 km)
    n_x: int = 13  # State dimension (position, velocity, quaternion, angular velocity)
    n_u: int = 6  # Control dimension (force and torque)

    state_dim: int = 12
    input_dim: int = 6
    
    # Inertia and gravity
    J_B: np.ndarray = np.diag([553.3, 779.8, 371.4])  # Moment of inertia in body frame
    g: float = -3.7114  # Gravitational acceleration (Mars-like)
    mass: float = 770.07  # Vehicle mass

    # Vector constants
    g_I: np.ndarray = np.array([0, 0, g]).astype('float64')  # Gravity in inertial frame
    y_B: np.ndarray = np.array([0, np.sqrt(2)/2, -np.sqrt(2)/2]).astype('float64')  # Body frame boresight
    z_I: np.ndarray = np.array([0, 0, 1]).astype('float64')  # Inertial frame vertical axis

    # Target final states
    r_I_final: np.ndarray = np.array([0, 0, 25.0])  # Final position
    v_I_final: np.ndarray = np.zeros(3).astype('float64')  # Final velocity
    q_BI_final: np.ndarray = np.array([1, 0, 0, 0]).astype('float64')
    w_B_final: np.ndarray = np.zeros(3).astype('float64')  # Final angular velocity

    # Physical limits
    F_B_max: float = 5000.0  # Max thrust in body frame
    r_I_max: float = 1000.0  # Max position magnitude
    M_B_max: float = 800.0  # Max torque in body frame
    v_I_max: float = 150.0  # Max velocity magnitude
    w_B_max: float = 0.04  # Max angular velocity magnitude

    # Angular constraints
    theta_max: float = np.deg2rad(30)  # Max line-of-sight angle
    phi_max: float = np.deg2rad(65)  # Max glide slope angle
    gamma_max: float = np.deg2rad(30)  # Max thrust pointing angle

    # SCvx algorithm settings
    iterations: int = 100  # Max iterations
    solver: str = ['ECOS', 'MOSEK', 'CLARABEL'][2]  # Convex solver choice
    verbose_solver: bool = False  # Solver verbosity flag
    r0: float = 1.0  # Initial trust region radius
    rl: float = 1e-5  # Minimum trust region radius
    alpha: float = 2.0  # Trust region shrinkage factor
    beta: float = 3.0  # Trust region expansion factor
    eps_tol: float = 1e-3  # Convergence tolerance
    rho0: float = 0.0  # Lower trust region adjustment threshold
    rho1: float = 0.25  # Middle trust region adjustment threshold
    rho2: float = 0.7  # Upper trust region adjustment threshold
    state_coeff: float = 0.1  # State cost weight
    final_state_coeff: float = 10.0  # Final state cost weight
    control_coeff: float = 1  # Control cost weight
    penalty_coeff: float = 1e5  # Penalty weight for constraints

    def __post_init__(self):
        """Post-initialization to scale parameters and compute derived values.

        Inputs:
            None (operates on instance attributes).

        Outputs:
            None (modifies instance attributes).
        """
        
        self.cos_phi_max = np.cos(self.phi_max)  # Cosine of glide slope angle
        self.cos_theta_max = np.cos(self.theta_max)  # Cosine of line-of-sight angle
        self.cos_gamma_max = np.cos(self.gamma_max)  # Cosine of thrust angle
        # Scale physical parameters by r_scale
        self.J_B /= self.r_scale**2
        self.g /= self.r_scale
        self.g_I /= self.r_scale
        self.r_I_final /= self.r_scale
        self.v_I_final /= self.r_scale
        self.F_B_max /= self.r_scale
        self.r_I_max /= self.r_scale
        self.M_B_max /= self.r_scale**2
        self.v_I_max /= self.r_scale