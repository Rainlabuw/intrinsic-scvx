import sympy as sp
import numpy as np
from typing import Tuple

TOL: float = 1e-10  # Numerical tolerance

def euler_to_quat(a: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion.

    Inputs:
        a (np.ndarray): Euler angles in degrees (roll, pitch, yaw), shape (3,).

    Outputs:
        np.ndarray: Quaternion [w, x, y, z], shape (4,).
    """
    a = np.deg2rad(a)
    cy = np.cos(a[1] * 0.5)
    sy = np.sin(a[1] * 0.5)
    cr = np.cos(a[0] * 0.5)
    sr = np.sin(a[0] * 0.5)
    cp = np.cos(a[2] * 0.5)
    sp = np.sin(a[2] * 0.5)
    q = np.zeros(4)
    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[3] = cy * cr * sp + sy * sr * cp
    q[2] = sy * cr * cp - cy * sr * sp
    return q

def rand_quat() -> np.ndarray:
    """Generate a random unit quaternion.

    Inputs:
        None.

    Outputs:
        np.ndarray: Random unit quaternion [w, x, y, z], shape (4,).
    """
    q = np.random.randn(4)
    return q / np.linalg.norm(q)

def rand_state() -> np.ndarray:
    """Generate a random state vector with a unit quaternion.

    Inputs:
        None.

    Outputs:
        np.ndarray: State vector [r, v, q, w], shape (13,).
    """
    q = rand_quat()
    x = np.random.randn(13)
    x[6:10] = q
    return x

def rand_tangent_quat(q: np.ndarray) -> np.ndarray:
    """Generate a random tangent quaternion orthogonal to q.

    Inputs:
        q (np.ndarray): Base quaternion, shape (4,).

    Outputs:
        np.ndarray: Tangent quaternion, shape (4,).
    """
    while True:
        w = np.random.randn(4)
        w[0] = 0
        if np.linalg.norm(w) < np.pi:
            break
    q_mult = quat_mult_matrix(q)
    dq = q_mult @ w
    return dq

def rand_tangent_state(x: np.ndarray) -> np.ndarray:
    """Generate a random tangent state vector.

    Inputs:
        x (np.ndarray): Base state vector, shape (13,).

    Outputs:
        np.ndarray: Tangent state vector, shape (13,).
    """
    q = x[6:10]
    dq = rand_tangent_quat(q)
    dx = np.random.randn(13)
    dx[6:10] = dq
    return dx

def sp_skew(v: sp.Matrix) -> sp.Matrix:
    """Compute symbolic skew-symmetric matrix from a vector.

    Inputs:
        v (sp.Matrix): 3D vector, shape (3, 1).

    Outputs:
        sp.Matrix: Skew-symmetric matrix, shape (3, 3).
    """
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def sp_dir_cosine(q: sp.Matrix) -> sp.Matrix:
    """Compute symbolic direction cosine matrix from quaternion.

    Inputs:
        q (sp.Matrix): Quaternion [w, x, y, z], shape (4, 1).

    Outputs:
        sp.Matrix: Rotation matrix, shape (3, 3).
    """
    return sp.Matrix([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])

def dir_cosine(q: np.ndarray) -> np.ndarray:
    """Compute direction cosine matrix from quaternion.

    Inputs:
        q (np.ndarray): Quaternion [w, x, y, z], shape (4,).

    Outputs:
        np.ndarray: Rotation matrix, shape (3, 3).
    """
    return np.array([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ]).astype('float64')

def sp_omega(w: sp.Matrix) -> sp.Matrix:
    """Compute symbolic quaternion derivative matrix from angular velocity.

    Inputs:
        w (sp.Matrix): Angular velocity, shape (3, 1).

    Outputs:
        sp.Matrix: Quaternion derivative matrix, shape (4, 4).
    """
    return sp.Matrix([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0]
    ])

def sp_exp(w: sp.Matrix) -> sp.Matrix:
    """Compute symbolic quaternion exponential from angular velocity.

    Inputs:
        w (sp.Matrix): Angular velocity, shape (3, 1).

    Outputs:
        sp.Matrix: Quaternion [w, x, y, z], shape (4, 1).
    """
    w1, w2, w3 = w[0], w[1], w[2]
    theta = sp.sqrt(w1 ** 2 + w2 ** 2 + w3 ** 2)
    q0 = sp.cos(theta)
    scale = sp.sinc(theta)
    q1 = w1 * scale
    q2 = w2 * scale
    q3 = w3 * scale
    return sp.Matrix([q0, q1, q2, q3])

def exp(w: np.ndarray) -> np.ndarray:
    """Compute quaternion exponential from angular velocity.

    Inputs:
        w (np.ndarray): Angular velocity, shape (3,).

    Outputs:
        np.ndarray: Quaternion [w, x, y, z], shape (4,).
    """
    theta = np.linalg.norm(w)
    if theta < TOL:
        return np.array([1, 0, 0, 0]).astype('float64')
    exp_w = np.array([np.cos(theta), 0, 0, 0])
    exp_w[1:] = np.sin(theta) / theta * w
    return exp_w

def sp_conj(q: sp.Matrix) -> sp.Matrix:
    """Compute symbolic quaternion conjugate.

    Inputs:
        q (sp.Matrix): Quaternion [w, x, y, z], shape (4, 1).

    Outputs:
        sp.Matrix: Conjugate quaternion [w, -x, -y, -z], shape (4, 1).
    """
    q0, qx, qy, qz = q[0], q[1], q[2], q[3]
    return sp.Matrix([q0, -qx, -qy, -qz])

def conj(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate.

    Inputs:
        q (np.ndarray): Quaternion [w, x, y, z], shape (4,).

    Outputs:
        np.ndarray: Conjugate quaternion [w, -x, -y, -z], shape (4,).
    """
    q0, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([q0, -qx, -qy, -qz])

def sp_log(q: sp.Matrix) -> sp.Matrix:
    """Compute symbolic quaternion logarithm.

    Inputs:
        q (sp.Matrix): Quaternion [w, x, y, z], shape (4, 1).

    Outputs:
        sp.Matrix: Logarithm [0, wx, wy, wz], shape (4, 1).
    """
    q0, qx, qy, qz = q[0], q[1], q[2], q[3]
    theta = sp.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    if theta == 0:
        return sp.zeros(4, 1)
    scale = sp.acos(q0) / theta
    wx = qx * scale
    wy = qy * scale
    wz = qz * scale
    return sp.Matrix([0, wx, wy, wz]).applyfunc(sp.re)

def log(q: np.ndarray) -> np.ndarray:
    """Compute quaternion logarithm.

    Inputs:
        q (np.ndarray): Quaternion [w, x, y, z], shape (4,).

    Outputs:
        np.ndarray: Logarithm [0, wx, wy, wz], shape (4,).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    q0, qx, qy, qz = q[0], q[1], q[2], q[3]
    theta = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    if theta < TOL:
        return np.zeros(4)
    scale = np.arccos(q0) / theta
    wx = qx * scale
    wy = qy * scale
    wz = qz * scale
    return np.real(np.array([0, wx, wy, wz]))

def sp_quat_retract(q: sp.Matrix, dq: sp.Matrix) -> sp.Matrix:
    """Compute symbolic quaternion retraction.

    Inputs:
        q (sp.Matrix): Base quaternion, shape (4, 1).
        dq (sp.Matrix): Tangent perturbation, shape (4, 1).

    Outputs:
        sp.Matrix: Retracted quaternion, shape (4, 1).
    """
    inv_q = sp_conj(q)
    inv_q_mult = sp_quat_mult_matrix(inv_q)
    pure_dq = inv_q_mult @ dq
    pure_dq = pure_dq[1:]
    exp_pure_dq = sp_exp(pure_dq)
    q_mult = sp_quat_mult_matrix(q)
    return q_mult @ exp_pure_dq

def sp_inv_quat_retract(q: sp.Matrix, p: sp.Matrix) -> sp.Matrix:
    """Compute symbolic inverse quaternion retraction.

    Inputs:
        q (sp.Matrix): Base quaternion, shape (4, 1).
        p (sp.Matrix): Target quaternion, shape (4, 1).

    Outputs:
        sp.Matrix: Tangent vector, shape (4, 1).
    """
    inv_q = sp_conj(q)
    inv_q_mult = sp_quat_mult_matrix(inv_q)
    q_mult = sp_quat_mult_matrix(q)
    inv_q_p = inv_q_mult @ p
    log_inv_q_p = sp_log(inv_q_p)
    return q_mult @ log_inv_q_p

def quat_retract(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Compute quaternion retraction.

    Inputs:
        q (np.ndarray): Base quaternion, shape (4,).
        dq (np.ndarray): Tangent perturbation, shape (4,).

    Outputs:
        np.ndarray: Retracted quaternion, shape (4,).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    if abs(dq.T @ q) > TOL:
        raise ValueError(f"q and dq must be orthogonal, got {q.T@dq}")
    inv_q = conj(q)
    inv_q_mult = quat_mult_matrix(inv_q)
    pure_dq = inv_q_mult @ dq
    exp_pure_dq = exp(pure_dq[1:])
    q_mult = quat_mult_matrix(q)
    p = q_mult @ exp_pure_dq
    p = p / np.linalg.norm(p)
    return p

def inv_quat_retract(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute inverse quaternion retraction.

    Inputs:
        q (np.ndarray): Base quaternion, shape (4,).
        p (np.ndarray): Target quaternion, shape (4,).

    Outputs:
        np.ndarray: Tangent vector, shape (4,).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    if abs(p.T @ p - 1) > TOL:
        raise ValueError(f"p is not unital, got {p.T@p}")
    inv_q = conj(q)
    inv_q_mult = quat_mult_matrix(inv_q)
    q_mult = quat_mult_matrix(q)
    inv_q_p = inv_q_mult @ p
    inv_q_p = inv_q_p / np.linalg.norm(inv_q_p)
    log_inv_q_p = log(inv_q_p)
    return q_mult @ log_inv_q_p

def sp_retract(x: sp.Matrix, dx: sp.Matrix) -> sp.Matrix:
    """Compute symbolic state retraction.

    Inputs:
        x (sp.Matrix): Base state vector, shape (13, 1).
        dx (sp.Matrix): Tangent perturbation, shape (13, 1).

    Outputs:
        sp.Matrix: Retracted state vector, shape (13, 1).
    """
    x = sp.Matrix(x)
    dx = sp.Matrix(dx)
    x_trans = sp.zeros(13, 1)
    x_trans[:6, 0] = x[:6, 0] + dx[:6, 0]
    x_trans[10:, 0] = x[10:, 0] + dx[10:, 0]
    q = x[6:10, 0]
    dq = dx[6:10, 0]
    q_trans = sp_quat_retract(q, dq)
    x_trans[6:10, 0] = q_trans
    return x_trans

def sp_inv_retract(x: sp.Matrix, z: sp.Matrix) -> sp.Matrix:
    """Compute symbolic inverse state retraction.

    Inputs:
        x (sp.Matrix): Base state vector, shape (13, 1).
        z (sp.Matrix): Target state vector, shape (13, 1).

    Outputs:
        sp.Matrix: Tangent vector, shape (13, 1).
    """
    x = sp.Matrix(x)
    z = sp.Matrix(z)
    dx = sp.zeros(13, 1)
    dx[:6, 0] = z[:6, 0] - x[:6, 0]
    dx[10:, 0] = z[10:, 0] - x[10:, 0]
    q_x = x[6:10, 0]
    q_z = z[6:10, 0]
    dq = sp_inv_quat_retract(q_x, q_z)
    dx[6:10, 0] = dq
    return dx

def retract(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Compute state retraction.

    Inputs:
        x (np.ndarray): Base state vector, shape (13,).
        dx (np.ndarray): Tangent perturbation, shape (13,).

    Outputs:
        np.ndarray: Retracted state vector, shape (13,).
    """
    x_trans = np.zeros(13)
    x_trans[:6] = x[:6] + dx[:6]
    x_trans[10:] = x[10:] + dx[10:]
    q = x[6:10]
    dq = dx[6:10]
    q_trans = quat_retract(q, dq)
    x_trans[6:10] = q_trans
    return x_trans

def inv_retract(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute inverse state retraction.

    Inputs:
        x (np.ndarray): Base state vector, shape (13,).
        z (np.ndarray): Target state vector, shape (13,).

    Outputs:
        np.ndarray: Tangent vector, shape (13,).
    """
    dx = np.zeros(13)
    dx[:6] = z[:6] - x[:6]
    dx[10:] = z[10:] - x[10:]
    q_x = x[6:10]
    q_z = z[6:10]
    dq = inv_quat_retract(q_x, q_z)
    dx[6:10] = dq
    return dx

def sp_dlog(q: sp.Matrix, dq: sp.Matrix) -> sp.Matrix:
    """Compute symbolic differential of quaternion logarithm.

    Inputs:
        q (sp.Matrix): Quaternion [w, x, y, z], shape (4, 1).
        dq (sp.Matrix): Quaternion perturbation, shape (4, 1).

    Outputs:
        sp.Matrix: Differential vector, shape (3, 1).
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    qv = sp.Matrix([q1, q2, q3])
    norm_qv = sp.sqrt(q1 ** 2 + q2 ** 2 + q3 ** 2)
    M = sp.zeros(3, 4)
    M[:, 0] = -1 / norm_qv / sp.sqrt(1 - q0 ** 2) * qv
    I = sp.eye(3)
    qv_outer = qv * qv.T
    M[:, 1:] = sp.acos(q0) / norm_qv * (I - qv_outer / norm_qv ** 2)
    return M @ dq

def dlog(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Compute differential of quaternion logarithm.

    Inputs:
        q (np.ndarray): Unit quaternion [w, x, y, z], shape (4,).
        dq (np.ndarray): Quaternion perturbation, shape (4,).

    Outputs:
        np.ndarray: Differential vector, shape (3,).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    if abs(dq.T @ q) > TOL:
        raise ValueError(f"q and dq must be orthogonal, got {q.T@dq}")
    
    q0 = q[0]  
    qv = q[1:] 
    norm_qv = np.linalg.norm(qv) 
    if abs(1 - q0) < TOL:
        return dq[1:]
    theta = np.arccos(q0)
    M = np.zeros((3, 4))
    M[:,0] = -qv/norm_qv**2
    I = np.eye(3)
    qv_outer = np.outer(qv, qv)
    M[:,1:] = theta/norm_qv * (I - qv_outer / norm_qv**2)
    return M @ dq

def dlog_matrix(q: np.ndarray) -> np.ndarray:
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    q0 = q[0]  
    qv = q[1:] 
    norm_qv = np.linalg.norm(qv) 
    M = np.zeros((3, 4))
    if norm_qv < TOL:
        M[:,1:] = np.eye(3)
        return M
    theta = np.arccos(q0)
    M[:,0] = -qv/norm_qv**2
    I = np.eye(3)
    qv_outer = np.outer(qv, qv)
    M[:,1:] = theta/norm_qv * (I - qv_outer / norm_qv**2)
    return M

def sp_grad_sinc(w: sp.Matrix) -> sp.Matrix:
    """Compute symbolic gradient of sinc function.

    Inputs:
        w (sp.Matrix): Vector input, shape (3, 1).

    Outputs:
        sp.Matrix: Gradient vector, shape (3, 1).
    """
    w1, w2, w3 = w[0], w[1], w[2]
    norm_w = sp.sqrt(w1 ** 2 + w2 ** 2 + w3 ** 2)
    scale = sp.cos(norm_w) / norm_w ** 2 - sp.sin(norm_w) / norm_w ** 3
    return scale * w

def sp_sinc(w: sp.Matrix) -> sp.Expr:
    """Compute symbolic sinc function.

    Inputs:
        w (sp.Matrix): Vector input, shape (3, 1).

    Outputs:
        sp.Expr: Scalar sinc value.
    """
    w1, w2, w3 = w[0], w[1], w[2]
    norm_w = sp.sqrt(w1 ** 2 + w2 ** 2 + w3 ** 2)
    return sp.sinc(norm_w)

def grad_sinc(w: np.ndarray) -> np.ndarray:
    """Compute gradient of sinc function.

    Inputs:
        w (np.ndarray): Vector input, shape (3,).

    Outputs:
        np.ndarray: Gradient vector, shape (3,).
    """
    w1, w2, w3 = w[0], w[1], w[2]
    norm_w = np.sqrt(w1 ** 2 + w2 ** 2 + w3 ** 2)
    scale = np.cos(norm_w) / norm_w ** 2 - np.sin(norm_w) / norm_w ** 3
    return scale * w

def sinc(w: np.ndarray) -> float:
    """Compute sinc function.

    Inputs:
        w (np.ndarray): Vector input, shape (3,).

    Outputs:
        float: Scalar sinc value.
    """
    w1, w2, w3 = w[0], w[1], w[2]
    norm_w = np.sqrt(w1 ** 2 + w2 ** 2 + w3 ** 2)
    return np.sinc(norm_w / np.pi)

def sp_dexp(w: sp.Matrix, eta: sp.Matrix) -> sp.Matrix:
    """Compute symbolic differential of quaternion exponential.

    Inputs:
        w (sp.Matrix): Angular velocity, shape (3, 1).
        eta (sp.Matrix): Perturbation vector, shape (3, 1).

    Outputs:
        sp.Matrix: Differential quaternion, shape (4, 1).
    """
    M = sp.zeros(4, 3)
    M[0, :] = -sp_sinc(w) * w.T
    I = sp.eye(3)
    M[1:, :] = sp_sinc(w) * I + sp_grad_sinc(w) * w.T
    return M @ eta

def dexp(w: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Compute differential of quaternion exponential.

    Inputs:
        w (np.ndarray): Angular velocity, shape (3,).
        eta (np.ndarray): Perturbation vector, shape (3,).

    Outputs:
        np.ndarray: Differential quaternion, shape (4,).
    """
    M = np.zeros((4, 3))
    M[0, :] = -sinc(w) * w.T
    I = np.eye(3)
    M[1:, :] = sinc(w) * I + grad_sinc(w) * w.T
    return M @ eta

def sp_d_inv_quat_retract(q: sp.Matrix, p: sp.Matrix, dp: sp.Matrix) -> sp.Matrix:
    """Compute symbolic differential of quaternion retraction.

    Inputs:
        q (sp.Matrix): Base quaternion, shape (4, 1).
        p (sp.Matrix): Target quaternion, shape (4, 1).
        dp (sp.Matrix): Quaternion perturbation, shape (4, 1).

    Outputs:
        sp.Matrix: Differential vector, shape (4, 1).
    """
    inv_q = sp_conj(q)
    inv_q_mult = sp_quat_mult_matrix(inv_q)
    q_mult = sp_quat_mult_matrix(q)
    inv_q_p = inv_q_mult @ p
    v = sp.zeros(4, 1)
    v[1:, 0] = sp_dlog(inv_q_p, inv_q_mult @ dp)
    return q_mult @ v

def d_inv_quat_retract(q: np.ndarray, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Compute differential of quaternion retraction.

    Inputs:
        q (np.ndarray): Base quaternion, shape (4,).
        p (np.ndarray): Target quaternion, shape (4,).
        dp (np.ndarray): Quaternion perturbation, shape (4,).

    Outputs:
        np.ndarray: Differential vector, shape (4,).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    if abs(p.T @ p - 1) > TOL:
        raise ValueError(f"p is not unital, got {p.T@p}")
    if abs(dp.T @ p) > TOL:
        raise ValueError(f"p and dp must be orthogonal, got {p.T@dp}")
    inv_q = conj(q)
    inv_q_mult = quat_mult_matrix(inv_q)
    q_mult = quat_mult_matrix(q)
    inv_q_p = inv_q_mult @ p
    v = np.zeros(4)
    v[1:] = dlog(inv_q_p, inv_q_mult @ dp)
    return q_mult @ v

def sp_d_inv_retract(x: sp.Matrix, z: sp.Matrix, dz: sp.Matrix) -> sp.Matrix:
    """Compute symbolic differential of inverse state retraction.

    Inputs:
        x (sp.Matrix): Base state vector, shape (13, 1).
        z (sp.Matrix): Target state vector, shape (13, 1).
        dz (sp.Matrix): Perturbation vector, shape (13, 1).

    Outputs:
        sp.Matrix: Differential tangent vector, shape (13, 1).
    """
    q_dz = dz[6:10, 0]
    q_x = x[6:10, 0]
    q_z = z[6:10, 0]
    q_dx = sp_d_inv_quat_retract(q_x, q_z, q_dz)
    dx = sp.zeros(13, 1)
    dx[6:10, 0] = q_dx
    dx[:6, 0] = dz[:6, 0]
    dx[10:, 0] = dz[10:, 0]
    return dx

def d_inv_retract(x: np.ndarray, z: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Compute differential of inverse state retraction.

    Inputs:
        x (np.ndarray): Base state vector, shape (13,).
        z (np.ndarray): Target state vector, shape (13,).
        dz (np.ndarray): Perturbation vector, shape (13,).

    Outputs:
        np.ndarray: Differential tangent vector, shape (13,).
    """
    q_dz = dz[6:10]
    q_x = x[6:10]
    q_z = z[6:10]
    q_dx = d_inv_quat_retract(q_x, q_z, q_dz)
    dx = np.zeros(13)
    dx[6:10] = q_dx
    dx[:6] = dz[:6]
    dx[10:] = dz[10:]
    return dx

def d_inv_retract_matrix(x: np.ndarray, z: np.ndarray, A) -> np.ndarray:
    """Compute differential of inverse state retraction.

    Inputs:
        x (np.ndarray): Base state vector, shape (13,).
        z (np.ndarray): Target state vector, shape (13,).
        dz (np.ndarray): Perturbation vector, shape (13,).

    Outputs:
        np.ndarray: Differential tangent vector, shape (13,).
    """
    D = np.zeros((13,A.shape[1]))
    for i in range(A.shape[1]):
        dz = A[:,i]
        q_dz = dz[6:10]
        q_x = x[6:10]
        q_z = z[6:10]
        q_dx = d_inv_quat_retract(q_x, q_z, q_dz)
        dx = np.zeros(13)
        dx[6:10] = q_dx
        dx[:6] = dz[:6]
        dx[10:] = dz[10:]
        D[:,i] = dx
    return D


def sp_quat_mult_matrix(q: sp.Matrix) -> sp.Matrix:
    """Compute symbolic quaternion multiplication matrix.

    Inputs:
        q (sp.Matrix): Quaternion [w, x, y, z], shape (4, 1).

    Outputs:
        sp.Matrix: Multiplication matrix, shape (4, 4).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_mult = sp.Matrix([
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w]
    ])
    return q_mult

def quat_mult_matrix(q: np.ndarray) -> np.ndarray:
    """Compute quaternion multiplication matrix.

    Inputs:
        q (np.ndarray): Quaternion [w, x, y, z], shape (4,).

    Outputs:
        np.ndarray: Multiplication matrix, shape (4, 4).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_mult = np.array([
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w]
    ])
    return q_mult

def right_quat_mult_matrix(q):
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    return np.array([
        [q_w, -q_x, -q_y, -q_z],
        [q_x,  q_w,  q_z, -q_y],
        [q_y, -q_z,  q_w,  q_x],
        [q_z,  q_y, -q_x,  q_w]
    ])

def quat_frame(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal basis for quaternion tangent space.

    Inputs:
        q (np.ndarray): Quaternion [w, x, y, z], shape (4,).

    Outputs:
        Tuple containing:
            np.ndarray: Basis vector 1, shape (4,).
            np.ndarray: Basis vector 2, shape (4,).
            np.ndarray: Basis vector 3, shape (4,).
    """
    if abs(q.T @ q - 1) > TOL:
        raise ValueError(f"q is not unital, got {q.T@q}")
    e1 = np.array([0, 1, 0, 0])
    e2 = np.array([0, 0, 1, 0])
    e3 = np.array([0, 0, 0, 1])
    quat_mult_mat = quat_mult_matrix(q)
    return np.array([quat_mult_mat @ e1, quat_mult_mat @ e2, quat_mult_mat @ e3]).T

def frame(x: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for state tangent space.

    Inputs:
        x (np.ndarray): State vector [r, v, q, w], shape (13,).

    Outputs:
        np.ndarray: Basis matrix, shape (13, 12).
    """
    q = x[6:10]
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    zeros3 = np.zeros(3)
    zeros4 = np.zeros(4)
    E = quat_frame(q)
    E1, E2, E3 = E[:,0], E[:,1], E[:,2]
    f1 = np.concatenate([e1, zeros3, zeros4, zeros3])
    f2 = np.concatenate([e2, zeros3, zeros4, zeros3])
    f3 = np.concatenate([e3, zeros3, zeros4, zeros3])
    f4 = np.concatenate([zeros3, e1, zeros4, zeros3])
    f5 = np.concatenate([zeros3, e2, zeros4, zeros3])
    f6 = np.concatenate([zeros3, e3, zeros4, zeros3])
    f7 = np.concatenate([zeros3, zeros3, E1, zeros3])
    f8 = np.concatenate([zeros3, zeros3, E2, zeros3])
    f9 = np.concatenate([zeros3, zeros3, E3, zeros3])
    f10 = np.concatenate([zeros3, zeros3, zeros4, e1])
    f11 = np.concatenate([zeros3, zeros3, zeros4, e2])
    f12 = np.concatenate([zeros3, zeros3, zeros4, e3])
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]).T

def quat_rotate(q, v):
    q_mult = quat_mult_matrix(q)
    pure_v = np.zeros(4)
    pure_v[1:] = v
    inv_q = conj(q)
    q_v = q_mult @ v
    q_v_mult = quat_mult_matrix(q_v)
    q_v_inv_q = q_v_mult @ inv_q
    return q_v_inv_q[1:]

def sq_dist(q, p):
    inv_q = conj(q)
    inv_q_mult = quat_mult_matrix(inv_q)
    error = inv_q_mult @ p
    log_error = log(error)
    return np.linalg.norm(log_error)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xi = np.random.randn(3)
    xi = xi/np.linalg.norm(xi)*np.random.rand()*np.pi/2
    q = exp(xi)
    q_mult = quat_mult_matrix(q)
    v = np.random.randn(3)
    v = v/np.linalg.norm(v)
    gamma = lambda t: log(q_mult @ exp(t*v))
    dq = np.zeros(4)
    dq[1:] = v
    dq = q_mult@dq
    tspan = np.logspace(-10,0,100)
    yspan = []
    for t in tspan:
        aux = (gamma(t) - gamma(0))/t
        dJ = dlog(q, dq)
        y = np.linalg.norm(dJ - aux[1:])
        yspan.append(y)
    yspan = np.array(yspan)
    plt.figure()
    plt.loglog(tspan, yspan)
    plt.grid()
    plt.show()