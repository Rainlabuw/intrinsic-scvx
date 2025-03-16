import numpy as np
import cvxpy as cvx

TOL = 1e-7
I3 = np.eye(3)
R3_basis = [
    np.array([1,0,0]), 
    np.array([0,1,0]), 
    np.array([0,0,1])

]
def norm(x):
    return np.linalg.norm(x)

def pure(q):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    return q[1:]

def normalize(x):
    return x/norm(x)

def real(q):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    return q[0]

def cross_matrix(u: np.ndarray) -> np.ndarray:
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    return np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

def inv(q: np.ndarray) -> np.ndarray:
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    return np.array([q[0], -q[1], -q[2], -q[3]]) 

def pure2quat(u):
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    return np.array([0, u[0], u[1], u[2]])

def exp(u: np.ndarray) -> np.ndarray:
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    theta = norm(u)
    if theta < TOL:
        return np.array([1,0,0,0])
    q = np.zeros(4)
    q[0] = np.cos(theta)
    q[1:] = u*np.sin(theta)/theta
    return q

def log(q):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    q0 = real(q)
    qv = pure(q)
    theta_v = norm(qv)
    if theta_v < TOL:
        return np.zeros(3)
    return qv/theta_v*np.arccos(q0)

def rand_quat():
    q = np.random.randn(4)
    return normalize(q)

def quat2mat(q, is_left=True):
    if q.shape != (4,) and q.shape != (3,):
        raise ValueError("q must be a 3- or 4-vector")
    if q.shape == (3,):
        q = pure2quat(q)
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    if is_left:
        matrix = np.array([
            [q_w, -q_x, -q_y, -q_z],
            [q_x,  q_w, -q_z,  q_y],
            [q_y,  q_z,  q_w, -q_x],
            [q_z, -q_y,  q_x,  q_w]
        ])
    else:
        matrix = np.array([
            [q_w, -q_x, -q_y, -q_z],
            [q_x,  q_w,  q_z, -q_y],
            [q_y, -q_z,  q_w,  q_x],
            [q_z,  q_y, -q_x,  q_w]
        ])
    return matrix

def mult(q, p):
    if p.shape != (4,) and p.shape != (3,):
        raise ValueError("p must be a 3- or 4-vector")
    if q.shape != (4,) and q.shape != (3,):
        raise ValueError("q must be a 3- or 4-vector")
    if p.shape == (3,):
        p = pure2quat(p)
    return quat2mat(q)@p

def rotate(u, q):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    return pure(mult(mult(q, u), inv(q)))

def translate(q, xi):
    return mult(q, exp(xi))

def translate_traj(q, eta, N):
    q_plus = np.zeros((4, N + 1))
    for k in range(N + 1):
        q_plus[:,k] = translate(q[:,k], eta[:,k])
    return q_plus

def get_frame(q):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    e1 = mult(q, np.array([0,1,0,0]))
    e2 = mult(q, np.array([0,0,1,0]))
    e3 = mult(q, np.array([0,0,0,1]))
    return [e1, e2, e3]

def get_coords(q, v):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if v.shape != (4,):
        raise ValueError("v must be a 4-vector")
    if np.dot(q,v) > TOL:
        raise ValueError("v must be perpendicular to q")
    E = get_frame(q)
    v_coords = np.zeros(3)
    for i in range(3):
        v_coords[i] = np.dot(E[i], v)
    return v_coords

def log_of_difference(q_head, q_base):
    if q_head.shape != (4,):
        raise ValueError("q_head must be a 4-vector")
    if q_base.shape != (4,):
        raise ValueError("q_base must be a 4-vector")
    return log(mult(inv(q_base), q_head))

def f(q, u, tau):
    u = np.array(u, dtype=np.float64)
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    exp_quat = exp(tau*u)
    return mult(q, exp_quat)

def df_dq(q: np.ndarray, u: np.ndarray, tau) -> np.ndarray:
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    p = exp(tau*u)
    P = quat2mat(p, is_left=False)
    return P

def dexp_du(u: np.ndarray) -> np.ndarray:
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    theta = norm(u)
    if theta < TOL:
        D = np.zeros((4,3))
        D[1:,:] = np.eye(3)
        return D
    inv_theta = 1/theta
    Q = np.zeros((4,3))
    norm_u = u*inv_theta
    sin_theta = np.sin(theta)
    sinc_theta = sin_theta*inv_theta
    grad_sinc = np.cos(theta)*inv_theta - sinc_theta*inv_theta
    Q[0,:] = -sin_theta*norm_u
    Q[1:,:] = sinc_theta*np.eye(3) + \
        grad_sinc*norm_u*np.vstack([u,u,u]).T
    return Q

def dlog_dq(q):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    q0 = real(q)
    qv = pure(q)
    theta = norm(qv)
    if theta < TOL:
        D = np.zeros((3,4))
        D[:,1:] = np.eye(3)
        return D  
    inv_theta = 1/theta
    D = np.zeros((3,4))
    D[:,0] = -qv*inv_theta/np.sqrt(1 - q0**2)
    D[:,1:] = np.arccos(q0)*inv_theta*(I3 - np.outer(qv, qv)*inv_theta*inv_theta)
    return D

def df_du(q: np.ndarray, u: np.ndarray, tau) -> np.ndarray:
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    Q = quat2mat(q)
    return tau*Q@dexp_du(tau*u)

def diff_f_q(q, u, dq, tau):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if dq.shape != (4,):
        raise ValueError("dq must be a 4-vector")
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    return mult(dq, exp(tau*u))

def diff_exp(u, du):
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    if du.shape != (3,):
        raise ValueError("du must be a 3-vector")
    return dexp_du(u)@du

def diff_log(q, dq):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if dq.shape != (4,):
        raise ValueError("dq must be a 4-vector")
    return dlog_dq(q)@dq

def diff_f_u(q, u, du, tau):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if du.shape != (3,):
        raise ValueError("du must be a 3-vector")
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    return tau*mult(q, diff_exp(tau*u, du))

def A_tilde(q_next, q, u, tau):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if q_next.shape != (4,):
        raise ValueError("q_next must be a 4-vector")
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    q_plus = f(q, u, tau)
    p = mult(inv(q_next), q_plus)
    E = get_frame(q)
    A_mat = np.zeros((3,3))
    for j in range(3):
        Aj = diff_f_q(q, u, E[j], tau)
        Aj = mult(inv(q_next), Aj)
        Aj = diff_log(p, Aj)
        Aj = mult(q_next, Aj)
        A_mat[:,j] = get_coords(q_next, Aj)
    return A_mat

def B_tilde(q_next, q, u, tau):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if q_next.shape != (4,):
        raise ValueError("q_next must be a 4-vector")
    if u.shape != (3,):
        raise ValueError("u must be a 3-vector")
    q_plus = f(q, u, tau)
    p = mult(inv(q_next), q_plus)
    B_mat = np.zeros((3,3))
    for j in range(3):
        Bj = diff_f_u(q, u, R3_basis[j], tau)
        Bj = mult(inv(q_next), Bj)
        Bj = diff_log(p, Bj)
        Bj = mult(q_next, Bj)
        B_mat[:,j] = get_coords(q_next, Bj)
    return B_mat

def sq_dist(p, q):
    s = log_of_difference(p, q)
    return np.inner(s,s)

def init_traj(q0, q_des, N):
    q = np.zeros((4, N + 1))
    u = np.zeros((3, N))
    q[:,0] = q0
    for k in range(N):
        qk = q[:,k]
        if sq_dist(q_des, qk) > (np.pi/2)**2:
            print("trajectory is outside geodesic convex space")
        uk = .1*log(mult(inv(qk), q_des))
        q[:,k + 1] = mult(qk, exp(uk))
        u[:,k] = uk
    return q, u

def trajectory(q0, u, tau, N):
    q = np.zeros((4, N + 1))
    q[:,0] = q0
    for k in range(N):
        q[:,k + 1] = f(q[:,k], u[:,k], tau)
    return q

def cvx_penalty(v):
        return cvx.norm(v, 1)

def lie_bracket(xi, eta):
    xi = pure2quat(xi)
    eta = pure2quat(eta)
    return pure(mult(xi, eta) - mult(eta, xi))

def penalty(v):
    return np.linalg.norm(v, 1)

def s(q, t_o, y_b, cos_theta_max):
    y_o = rotate(y_b, q)
    return np.inner(t_o, y_o) - cos_theta_max

def diff_s(q, dq, t_o, y_b):
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector")
    if dq.shape != (4,):
        raise ValueError("dq must be a 4-vector")
    if np.dot(q, dq) > TOL:
        raise ValueError("dq and q must be orthogonal.")
    eta = pure(mult(inv(q), dq))
    return t_o.T@(rotate(lie_bracket(eta, y_b), q))

def S(q, t_o, y_b):
    S_mat = np.zeros(3)
    E = get_frame(q)
    for i in range(3):
        S_mat[i] = diff_s(q, E[i], t_o, y_b)
    return S_mat

def dS_dq(q, t_o, y_b):
    y_b_cross = quat2mat(y_b, is_left=False)
    M_H = np.block([
        [np.zeros((4,4)), y_b_cross.T],
        [y_b_cross, np.zeros((4,4))]
    ])
    Q = np.block([
        [np.eye(4)],
        [1/2*quat2mat(t_o)]
    ])
    out = 2*Q.T@M_H@Q@q
    return out

def Hess_dist_coords(q, q0):
    p = mult(inv(q), q0)
    log_q_q0 = mult(q, pure2quat(log(p)))
    theta = norm(log_q_q0)
    if theta < TOL:
        return np.eye(3)
    f_theta = theta/np.sin(theta)
    u = log_q_q0/theta
    uu = np.outer(u,u)
    H = uu + f_theta*np.cos(theta)*(np.eye(4) - np.outer(q, q) - uu)
    Q = quat2mat(q)
    return (Q.T@H@Q)[1:,1:]

def h(q, q_des):
    return sq_dist(q, q_des)/2

def diff_h(q, xi, q_des):
    return log_of_difference(q, q_des).T@xi

def grad_h(q, q_des):
    return mult(q, log_of_difference(q, q_des))

def diff_grad_h(q, dq, q_des):
    p = mult(inv(q_des), q)
    return mult(dq, log(p)) + mult(q, diff_log(p, mult(inv(q_des), dq)))

def Hess_h(q, dq, q_des):
    p = mult(inv(q_des), q)
    return mult(q, dlog_dq(p)@dq)

