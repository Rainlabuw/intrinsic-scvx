import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from typing import Tuple, List, Optional
from ..utils import sp_dir_cosine, sp_quat_mult_matrix, sp_exp, sp_skew, dir_cosine
from ..parameters import PoweredDescentParameters
from ..models import Model


class PoweredDescentModel(Model):
    def __init__(self, params: PoweredDescentParameters, cache_file: str, x_init: Optional[np.ndarray] = None):
        """Initialize powered descent model.

        Inputs:
            params (PoweredDescentParameters): Model parameters.
            cache_file (str): Path to cache symbolic expressions.
            x_init (Optional[np.ndarray]): Initial state, shape (state_dim,), defaults to None (randomized).

        Outputs:
            None
        """
        super().__init__(params=params, x_init=x_init)
        self.cache_file = "data/" + cache_file
        self._init_params()
        self._load_or_compute_dynamics()
        self._setup_lambdified_functions()

    def _set_random_initial_state(self):
        """Generate a random initial state respecting physical constraints.

        Inputs:
            None
        Outputs:
            None
        """
        r_I_init = np.zeros(3)
        v_I_init = np.zeros(3)
        q_BI_init = np.zeros(4)
        w_B_init = np.zeros(4)
        r_I_init[2] = 500 / self.params.r_scale  # Initial altitude
        r_I_init[:2] = np.random.randn(2)
        r_I_init[:2] = r_I_init[:2] / np.linalg.norm(r_I_init[:2]) * 500 / self.params.r_scale  # Random horizontal position
        v_I_init[2] = np.random.uniform(-100, -60) / self.params.r_scale  # Downward velocity
        v_I_init[0:2] = np.random.uniform(-0.5, -0.2, size=2) * r_I_init[0:2] / self.params.r_scale  # Lateral velocity
        w_B_init = np.deg2rad((np.random.uniform(-20, 20), np.random.uniform(-20, 20), 0))  # Random angular velocity
        q_BI_init = np.array([1, 0, 0, 0]).astype('float64')  # Identity quaternion
        self.x_init = np.concatenate([r_I_init, v_I_init, q_BI_init, w_B_init])

    def _load_or_compute_dynamics(self):
        """Load cached dynamics or compute and cache them.

        Inputs:
            None

        Outputs:
            None
        """
        if os.path.exists(self.cache_file):
            print(f"Loading cached expressions from {self.cache_file}")
            with open(self.cache_file, 'rb') as file:
                cached_data: Tuple = pickle.load(file)
                self.f: sp.Matrix = cached_data[0]  # Dynamics equations
                self.s: sp.Matrix = cached_data[1]  # Constraints
                self.A: sp.Matrix = cached_data[2]  # State Jacobian
                self.B: sp.Matrix = cached_data[3]  # Control Jacobian
                self.S: sp.Matrix = cached_data[4]  # Constraint Jacobian
        else:
            self.f, self.A, self.B = self._init_symbolic_dynamics()  # Compute dynamics
            self.s, self.S = self._init_symbolic_constraints()  # Compute constraints
            with open(self.cache_file, 'wb') as file:
                pickle.dump((self.f, self.s, self.A, self.B, self.S), file)
        self.constraints_dim = self.S.shape[0]  # Number of constraints

    def _setup_lambdified_functions(self):
        """Convert symbolic expressions to callable numpy functions.

        Inputs:
            None
        Outputs:
            None
        """
        x: sp.Matrix = sp.Matrix(sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13', real=True))  # State symbols
        u: sp.Matrix = sp.Matrix(sp.symbols('u1 u2 u3 u4 u5 u6', real=True))  # Control symbols
        self.lambdified_dynamics = sp.lambdify((x, u), self.f, 'numpy')  # Dynamics function: (13,), (6,) -> (13,)
        self.lambdified_constraints = sp.lambdify((x, u), self.s, 'numpy')  # Constraints function: (13,), (6,) -> (constraint_dim,)
        self.lambdified_A_matrix = sp.lambdify((x, u), self.A, 'numpy')  # State Jacobian: (13,), (6,) -> (13, 13)
        self.lambdified_B_matrix = sp.lambdify((x, u), self.B, 'numpy')  # Control Jacobian: (13,), (6,) -> (13, 6)
        self.lambdified_S_matrix = sp.lambdify((x, u), self.S, 'numpy')  # Constraint Jacobian: (13,), (6,) -> (constraint_dim, 13)

    def _init_symbolic_dynamics(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        """Define symbolic dynamics and their Jacobians.

        Inputs:
            None

        Outputs:
            Tuple[sp.Matrix, sp.Matrix, sp.Matrix]: Dynamics f (13, 1), state Jacobian A (13, 13), control Jacobian B (13, 6)
        """
        x = sp.Matrix(sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13', real=True))  # State symbols
        u = sp.Matrix(sp.symbols('u1 u2 u3 u4 u5 u6', real=True))  # Control symbols
        F_B = u[:3, 0]  # Body-frame force, shape (3,)
        M_B = u[3:, 0]  # Body-frame torque, shape (3,)
        sp_g_I = sp.Matrix(self.g_I)  # Gravity vector, shape (3,)
        sp_J_B = sp.Matrix(self.J_B)  # Inertia matrix, shape (3, 3)
        r_I = x[:3, 0]  # Position, shape (3,)
        v_I = x[3:6, 0]  # Velocity, shape (3,)
        q_BI = x[6:10, 0]  # Quaternion, shape (4,)
        w_B = x[10:, 0]  # Angular velocity, shape (3,)
        C_BI = sp_dir_cosine(q_BI)  # Body-to-inertial DCM, shape (3, 3)
        C_IB = C_BI.T  # Inertial-to-body DCM, shape (3, 3)
        F_I = C_IB * F_B  # Inertial-frame force, shape (3,)
        f = sp.zeros(self.n_x, 1)  # Dynamics vector, shape (13, 1)
        f[:3, 0] = r_I + self.dt * v_I  # Position update
        f[3:6, 0] = v_I + self.dt * (1 / self.mass * F_I + sp_g_I)  # Velocity update
        q_BI_mult = sp_quat_mult_matrix(q_BI)  # Quaternion multiplication matrix, shape (4, 4)
        f[6:10, 0] = q_BI_mult * sp_exp(w_B * self.dt)  # Quaternion update
        f[10:, 0] = w_B + self.dt * ((sp_J_B ** -1) * (M_B - sp_skew(w_B) * (sp_J_B * w_B)))  # Angular velocity update
        f = sp.simplify(f)  # Simplify dynamics
        A = sp.simplify(f.jacobian(x))  # State Jacobian, shape (13, 13)
        B = sp.simplify(f.jacobian(u))  # Control Jacobian, shape (13, 6)
        return f, A, B
    
    def _init_symbolic_constraints(self) -> Tuple[sp.Matrix, sp.Matrix]:
        """Define symbolic constraints and their Jacobian.

        Inputs:
            None

        Outputs:
            Tuple[sp.Matrix, sp.Matrix]: Constraint vector s (8, 1), state Jacobian S (8, 13)
        """
        x = sp.Matrix(sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13', real=True))  # State symbols
        u = sp.Matrix(sp.symbols('u1 u2 u3 u4 u5 u6', real=True))  # Control symbols
        F_B = u[:3, 0]  # Body-frame force, shape (3,)
        r_I = x[:3, 0]  # Position, shape (3,)
        v_I = x[3:6, 0]  # Velocity, shape (3,)
        q_BI = x[6:10, 0]  # Quaternion, shape (4,)
        w_B = x[10:, 0]  # Angular velocity, shape (3,)
        C_BI = sp_dir_cosine(q_BI)  # Body-to-inertial DCM, shape (3, 3)
        r_I_dot_z = r_I.dot(self.z_I)  # Dot product with vertical axis
        glide_slope_constraint = r_I.norm() * self.cos_phi_max - r_I_dot_z  # Glide slope constraint
        glide_slope_constraint = sp.simplify(glide_slope_constraint)
        trigger = 10 / self.r_scale - sp.Matrix(r_I[:2]).norm()  # Trigger for line-of-sight
        r_B = C_BI * r_I  # Position in body frame
        r_B_dot_y_B = r_B.dot(self.y_B)  # Dot product with boresight
        line_of_sight_constraint = r_I.norm() * self.cos_theta_max + r_B_dot_y_B  # Line-of-sight constraint
        line_of_sight_constraint = sp.simplify(line_of_sight_constraint)
        h = -sp.Min(trigger, 0) * line_of_sight_constraint  # Combined constraint
        h = sp.simplify(h)
        gamma_constraint = F_B.norm() * self.cos_gamma_max - F_B[2]  # Thrust angle constraint
        gamma_constraint = sp.simplify(gamma_constraint)
        positive_r_constraint = -r_I[2]  # Positive altitude constraint
        r_constraint = r_I.dot(r_I) - self.r_I_max ** 2  # Position magnitude constraint
        v_constraint = v_I.dot(v_I) - self.v_I_max ** 2  # Velocity magnitude constraint
        F_constraint = F_B.dot(F_B) - self.F_B_max ** 2  # Force magnitude constraint
        w_constraint = w_B.dot(w_B) - self.w_B_max ** 2  # Angular velocity magnitude constraint
        s = sp.Matrix([
            glide_slope_constraint,
            h,
            gamma_constraint,
            r_constraint,
            positive_r_constraint,
            v_constraint,
            w_constraint,
            F_constraint
        ])  # Constraint vector, shape (8, 1)
        s = sp.simplify(s)
        S = s.jacobian(x)  # Constraint Jacobian, shape (8, 13)
        return s, S

    def _init_params(self):
        """Extract and store parameters from params object.

        Inputs:
            None

        Outputs:
            None
        """
        self.n_x = self.params.n_x  # State dimension, int
        self.n_u = self.params.n_u  # Control dimension, int
        self.state_dim = self.params.state_dim  # Alias for state dimension, int
        self.input_dim = self.params.input_dim  # Alias for control dimension, int
        self.J_B = self.params.J_B  # Inertia matrix, shape (3, 3)
        self.mass = self.params.mass  # Mass, float
        self.g_I = self.params.g_I  # Gravity vector in inertial frame, shape (3,)
        self.y_B = self.params.y_B  # Boresight vector in body frame, shape (3,)
        self.z_I = self.params.z_I  # Vertical axis in inertial frame, shape (3,)
        self.r_I_final = self.params.r_I_final  # Final position, shape (3,)
        self.v_I_final = self.params.v_I_final  # Final velocity, shape (3,)
        self.w_B_final = self.params.w_B_final  # Final angular velocity, shape (3,)
        self.F_B_max = self.params.F_B_max  # Max thrust magnitude, float
        self.r_I_max = self.params.r_I_max  # Max position magnitude, float
        self.v_I_max = self.params.v_I_max  # Max velocity magnitude, float
        self.w_B_max = self.params.w_B_max  # Max angular velocity magnitude, float
        self.M_B_max = self.params.M_B_max  # Max torque magnitude, float
        self.theta_max = self.params.theta_max  # Max line-of-sight angle, float
        self.phi_max = self.params.phi_max  # Max glide slope angle, float
        self.cos_phi_max = self.params.cos_phi_max  # Cosine of phi_max, float
        self.cos_theta_max = self.params.cos_theta_max  # Cosine of theta_max, float
        self.gamma_max = self.params.gamma_max  # Max thrust angle, float
        self.cos_gamma_max = self.params.cos_gamma_max  # Cosine of gamma_max, float
        self.r_scale = self.params.r_scale  # Position scaling factor, float
        self.dt = self.params.dt  # Time step, float
        self.K = self.params.K  # Number of time steps, int
        self.tf = self.params.tf  # Final time, float
        self.TOL = self.params.TOL

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute next state with normalized quaternion.

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input (force and torque), shape (6,).

        Outputs:
            np.ndarray: Next state, shape (13,).
        """
        f = self.lambdified_dynamics(x, u)[:, 0]
        f[6:10] /= np.linalg.norm(f[6:10])  # Normalize quaternion
        return f

    def constraints(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate physical constraints (e.g., glide slope).

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).

        Outputs:
            np.ndarray: Constraint values, shape (constraint_dim,).
        """
        return self.lambdified_constraints(x, u)[:, 0]

    def Q_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return zero control constraint Jacobian (placeholder).

        Inputs:
            x (np.ndarray): Current state, shape (13,).
            u (np.ndarray): Control input, shape (6,).

        Outputs:
            np.ndarray: Zero matrix, shape (constraint_dim, 6).
        """
        return np.zeros((self.constraints_dim, self.input_dim))

    def extract_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split state into components.

        Inputs:
            x (np.ndarray): State vector, shape (13,).

        Outputs:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Position (3,), velocity (3,), quaternion (4,), angular velocity (3,).
        """
        return x[:3], x[3:6], x[6:10], x[10:]

    def initialize_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial linearly interpolated trajectory.

        Inputs:
            None

        Outputs:
            Tuple[np.ndarray, np.ndarray]: State trajectory (13, K+1), control trajectory (6, K).
        """
        x = np.zeros((self.n_x, self.K + 1))
        u = np.zeros((self.n_u, self.K))
        r_I_init, v_I_init, q_BI_init, w_B_init = self.extract_state(self.x_init)
        x[:, 0] = self.x_init
        for k in range(self.K):
            alpha1 = (self.K - k) / self.K
            alpha2 = k / self.K
            r_I_k = alpha1 * r_I_init + alpha2 * self.r_I_final
            v_I_k = alpha1 * v_I_init + alpha2 * self.v_I_final
            w_B_k = alpha1 * w_B_init + alpha2 * self.w_B_final
            x[:, k + 1] = np.concatenate([r_I_k, v_I_k, q_BI_init, w_B_k])
            u[:, k] = np.concatenate([self.F_B_max / 2 * np.array([0, 0, 1]), np.zeros(3)])
        x[:, self.K] = np.concatenate([self.r_I_final, self.v_I_final, q_BI_init, self.w_B_final])
        return x, u
    
    def print_trajectory(self, x: np.ndarray, u: np.ndarray) -> None:
        """Print trajectory details with dynamics error and constraints.

        Inputs:
            x (np.ndarray): State trajectory, shape (13, K+1)
            u (np.ndarray): Control trajectory, shape (6, K)

        Outputs:
            None
        """
        tspan = np.linspace(0, self.tf, self.K)  # Time steps
        for k in range(self.K):
            tk = tspan[k]  # Time at step k
            xk = x[:, k]  # State at step k
            uk = u[:, k]  # Control at step k
            rk, vk, qk, wk = xk[:3], xk[3:6], xk[6:10], xk[10:]  # Position, velocity, quaternion, angular velocity
            xkp1 = x[:, k + 1]  # Next state
            print(f"t: {tk:.3f}, \n" +
                f"r: {rk*self.r_scale}, \n" +
                f"v: {vk*self.r_scale}, \n" +
                f"q: {qk},\nw: {wk}\n" +
                f"F_B: {uk[:3]*self.r_scale}\n" +
                f"M_B: {uk[3:]*self.r_scale*self.r_scale}")
            zk = self.dynamics(xk, uk)  # Predicted next state
            sigmak = xkp1 - zk  # Dynamics error
            print(f"log(dyn_error): {np.log10(np.linalg.norm(sigmak)):.7f}, ")
            print(f"constraints: {self.constraints(xk, uk)}")
            print()

    def animate_trajectory(self, x_traj: np.ndarray, u_traj: np.ndarray, speed: float = 1.0, filename: str = 'rocket_descent.gif', show: bool = True):
        """Create and save 3D animation of descent.

        Inputs:
            x_traj (np.ndarray): State trajectory, shape (13, K+1).
            u_traj (np.ndarray): Control trajectory, shape (6, K).
            speed (float): Animation speed factor, default 1.0.
            filename (str): Output GIF filename, default 'rocket_descent.gif'.
            show (bool): Display animation if True, default True.

        Outputs:
            None
        """
        r_traj = x_traj[:3, :]
        positions = r_traj.T / 1000.0 * self.r_scale

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Rocket Powered Descent Trajectory (in km)')

        x_min = float(np.nanmin(positions[:, 0]))
        x_max = float(np.nanmax(positions[:, 0]))
        y_min = float(np.nanmin(positions[:, 1]))
        y_max = float(np.nanmax(positions[:, 1]))
        z_max = float(np.nanmax(positions[:, 2]))

        z_max_m = max(self.x_init[2], z_max * 1000)
        z_max_km = z_max_m / 1000.0

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        xy_range = max(x_max - x_min, y_max - y_min)
        r_max_km = z_max_km * np.tan(self.phi_max)
        half_range = max(xy_range / 2, r_max_km)
        margin = 0.1 * half_range
        half_range += margin

        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(0, z_max_km + margin)

        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
        ax.set_box_aspect([1, 1, z_range / x_range])

        theta = np.linspace(0, 2 * np.pi, 50)
        z_levels = np.linspace(0, z_max_km, 10)
        for z_level in z_levels:
            r = z_level * np.tan(self.phi_max)
            x_circle = r * np.cos(theta)
            y_circle = r * np.sin(theta)
            z_circle = np.full_like(theta, z_level)
            ax.plot(x_circle, y_circle, z_circle, color='gray', alpha=0.3)

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k--', alpha=0.5, label='Full Trajectory')

        dot, = ax.plot([], [], [], 'ro', markersize=10, label='Rocket Position')
        thrust_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='orange', label='Thrust Vector', linewidth=2)
        boresight_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='blue', label='Boresight Vector', linewidth=2)
        position_line, = ax.plot([], [], [], 'black', linestyle=':', alpha=1, label='Position Vector', linewidth=1.5)
        ax.legend()

        thrust_scale = 0.1
        boresight_length = 0.2
        def update(frame: int) -> List:
            nonlocal thrust_arrow, boresight_arrow, position_line
            x = x_traj[:, frame]
            u = u_traj[:, frame]
            pos = positions[frame]
            F_B = u[:3]
            q_BI = x[6:10]
            C_BI = dir_cosine(q_BI)
            C_IB = C_BI.T
            F_I = C_IB @ F_B
            F_B_mag = np.linalg.norm(F_B)
            arrow_length = F_B_mag * thrust_scale
            F_I_dir = F_I / F_B_mag if F_B_mag > 0 else np.zeros(3)
            y_I = C_IB @ self.y_B
            y_I_dir = y_I / np.linalg.norm(y_I)

            dot.set_data([pos[0]], [pos[1]])
            dot.set_3d_properties([pos[2]])

            thrust_arrow.remove()
            thrust_arrow = ax.quiver(
                pos[0], pos[1], pos[2],
                F_I_dir[0] * arrow_length, F_I_dir[1] * arrow_length,
                F_I_dir[2] * arrow_length,
                color='orange', linewidth=1
            )

            boresight_arrow.remove()
            boresight_arrow = ax.quiver(
                pos[0], pos[1], pos[2],
                y_I_dir[0] * boresight_length, y_I_dir[1] * boresight_length,
                y_I_dir[2] * boresight_length,
                color='blue', linewidth=1.5
            )

            position_line.set_data([0, pos[0]], [0, pos[1]])
            position_line.set_3d_properties([0, pos[2]])

            return [dot, thrust_arrow, boresight_arrow, position_line]

        if speed <= 0:
            raise ValueError("Speed must be positive")
        interval = (self.dt * self.r_scale) / speed
        fps = self.r_scale / interval

        anim = FuncAnimation(fig, update, frames=self.K, interval=interval, blit=False)
        anim.save(filename, writer='pillow', fps=fps, dpi=100)
        print(f"Animation saved as {filename}")
        if show:
            plt.show()
        plt.close()