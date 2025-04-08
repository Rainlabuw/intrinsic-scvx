from methods import *
import time

def penalized_geo_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qk_p1 = q[:, k + 1]
        J += state_lambda*h(qk, q_des) + \
            control_lambda*uk.T@uk + \
            penalty_lambda*penalty(log_of_difference(qk_p1, f(qk, uk, tau))) + \
            penalty_lambda*np.abs(s(qk, t_o, y_b, cos_theta_max))
    qN = q[:,N]
    J += final_state_lambda*h(qN, q_des)
    return J

def geo_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        qk = normalize(qk)
        uk = u[:,k]

        J += state_lambda*h(qk, q_des) + \
            control_lambda*uk.T@uk
    qN = q[:,N]
    J += final_state_lambda*h(qN, q_des)
    return J

def penalized_eucl_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qk_p1 = q[:, k + 1]
        J += state_lambda*(qk - q_des).T@(qk - q_des) + \
            control_lambda*uk.T@uk + \
            penalty_lambda*penalty(qk_p1 - f(qk, uk, tau)) + \
            penalty_lambda*np.abs(s(qk, t_o, y_b, cos_theta_max))
    qN = q[:,N]
    J += final_state_lambda*(qN - q_des).T@(qN - q_des)
    return J

def eucl_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        J += state_lambda*(qk - q_des).T@(qk - q_des) + \
            control_lambda*uk.T@uk
    qN = q[:,N]
    J += final_state_lambda*(qN - q_des).T@(qN - q_des)
    return J

def cvx_linearized_geo_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    eta, xi, v, s1,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        s1k = s1[:,k]
        Hk = Hess_dist_coords(qk, q_des)
        pk = log(mult(inv(q_des), qk))
        J += penalty_lambda*cvx_penalty(vk) + \
            penalty_lambda*cvx_penalty(s1k) + \
            control_lambda*cvx.sum_squares(uk + xik) + \
            state_lambda*(h(qk, q_des) + pk.T @ etak + 1/2*cvx.quad_form(etak, Hk))
    qN = q[:,N]
    pN = log(mult(inv(q_des), qN))
    etaN = eta[:,N]
    HN = Hess_dist_coords(qN, q_des)
    J += final_state_lambda*(h(qN, q_des) + pN.T @ etaN + 1/2*cvx.quad_form(etaN, HN))
    return J

def linearized_geo_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    eta, xi, v, s1,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        s1k = s1[:,k]
        Hk = Hess_dist_coords(qk, q_des)
        pk = log(mult(inv(q_des), qk))
        J += penalty_lambda*penalty(vk) + \
            penalty_lambda*penalty(s1k) + \
            control_lambda*np.linalg.norm(uk + xik)**2 + \
            state_lambda*(h(qk, q_des) + pk.T @ etak + 1/2*etak@Hk@etak)
    qN = q[:,N]
    pN = log(mult(inv(q_des), qN))
    etaN = eta[:,N]
    HN = Hess_dist_coords(qN, q_des)
    J += final_state_lambda*(h(qN, q_des) + pN.T @ etaN + 1/2*etaN@HN@etaN)
    return J

def cvx_linearized_eucl_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    eta, xi, v, s1,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        sk = s1[:,k]
        J += state_lambda*cvx.sum_squares(qk + etak - q_des) + \
            control_lambda*cvx.sum_squares(uk + xik) + \
            penalty_lambda*cvx_penalty(vk) + \
            penalty_lambda*cvx_penalty(sk)
    qN = q[:,N]
    etaN = eta[:,N]
    J += final_state_lambda*cvx.sum_squares(qN + etaN - q_des)
    return J

def linearized_eucl_traj_cost(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    eta, xi, v, s1,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N
):
        J = 0
        for k in range(N):
            qk = q[:,k]
            uk = u[:,k]
            etak = eta[:,k]
            xik = xi[:,k]
            vk = v[:,k]
            sk = s1[:,k]
            J += state_lambda*np.linalg.norm(qk + etak - q_des)**2 + \
                control_lambda*np.linalg.norm(uk + xik)**2 + \
                penalty_lambda*penalty(vk) + \
                penalty_lambda*penalty(sk)
        qN = q[:,N]
        etaN = eta[:,N]
        J += final_state_lambda*np.linalg.norm(qN + etaN - q_des)**2
        return J

def convex_optimal_control_subproblem(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N,
    r
):
    eta = cvx.Variable((3, N + 1))
    xi = cvx.Variable((3, N))
    v = cvx.Variable((3, N))
    s1 = cvx.Variable((1, N))
    constraints = [eta[:,0] == np.zeros(3)]
    for k in range(N):
        qk = q[:,k]
        qk_p1 = q[:,k + 1]
        uk = u[:,k]
        etak = eta[:,k]
        etak_p1 = eta[:,k + 1]
        xik = xi[:,k]
        Ak = A_tilde(qk_p1, qk, uk, tau)
        Bk = B_tilde(qk_p1, qk, uk, tau)
        Sk = S(qk, t_o, y_b)
        vk = v[:,k]
        s1k = s1[:,k]
        constraints.extend([
            etak_p1 == log_of_difference(f(qk, uk, tau), qk_p1) + Ak@etak + Bk@xik + vk,
            cvx.norm(xik, 2) <= r,
            s(qk, t_o, y_b, cos_theta_max) + Sk@etak - s1k <= 0,
            s1k >= 0,
        ])
    J = cvx_linearized_geo_traj_cost(
        q, u, tau, t_o, y_b, cos_theta_max, q_des,
        eta, xi, v, s1,
        penalty_lambda, 
        state_lambda, 
        control_lambda, 
        final_state_lambda,
        N
    )
    problem = cvx.Problem(cvx.Minimize(J), constraints)
    opt_value = problem.solve(solver=cvx.CLARABEL)
    return eta.value, xi.value, v.value, s1.value

def eucl_convex_optimal_control_subproblem(
    q, u, tau, t_o, y_b, cos_theta_max, q_des,
    penalty_lambda, 
    state_lambda, 
    control_lambda, 
    final_state_lambda,
    N,
    r
):
    eta = cvx.Variable((4, N + 1))
    xi = cvx.Variable((3, N))
    v = cvx.Variable((4, N))
    s1 = cvx.Variable((1, N))
    constraints = [eta[:,0] == np.zeros(4)]
    for k in range(N):
        qk = q[:,k]
        qk_p1 = q[:, k + 1]
        uk = u[:,k]
        etak = eta[:,k]
        etak_p1 = eta[:,k + 1]
        xik = xi[:,k]
        Ak = df_dq(qk, uk, tau)
        Bk = df_du(qk, uk, tau)
        Sk = dS_dq(qk, t_o, y_b)
        vk = v[:,k]
        sk = s1[:,k]
        constraints.extend([
            etak_p1 + qk_p1 == f(qk, uk, tau) + Ak@etak + Bk@xik + vk,
            cvx.norm(xik, 2) <= r,
            s(qk, t_o, y_b, cos_theta_max) + Sk@etak - sk <= 0,
            sk >= 0
        ])
    J = cvx_linearized_eucl_traj_cost(
        q, u, tau, t_o, y_b, cos_theta_max, q_des,
        eta, xi, v, s1,
        penalty_lambda, 
        state_lambda, 
        control_lambda, 
        final_state_lambda,
        N
    )
    problem = cvx.Problem(cvx.Minimize(J), constraints)
    opt_value = problem.solve(solver=cvx.CLARABEL)
    return eta.value, xi.value, v.value, s1.value

def experiment_init_traj(theta_max, t_o, y_b, N):
    while True:
        while True:
            xi = np.random.randn(3)
            xi = normalize(xi)
            xi = np.random.rand()*np.pi/2*xi
            q_des = exp(xi)
            y_o = rotate(y_b, q_des)
            ang = np.arccos(y_o.T@t_o)
            if ang > theta_max:
                break

        while True:
            xi = -xi + .01*np.random.randn(3)
            xi = normalize(xi)
            q0 = exp(xi)
            y_o = rotate(y_b, q0)
            ang = np.arccos(y_o.T@t_o)
            if ang > theta_max and np.sqrt(sq_dist(q0, q_des)) < np.pi/2:
                break

        traj_breaks_constraint = False
        last_pass = False
        q, u = init_traj(q0, q_des, N)
        for qk in q.T:
            y_o = rotate(y_b, qk)
            ang = np.arccos(y_o.T@t_o)
            if ang < theta_max:
                traj_breaks_constraint = True
        last_pass = ang > theta_max
        if traj_breaks_constraint and last_pass:
            break

    return q, u, q_des

def run_SCvx_experiment(
    q, u, q_des, cos_theta_max, tau, N,
    r, rl, alpha, beta, eps_tol, rho0, rho1, rho2,
    t_o, y_b,
    penalty_lambda,
    state_lambda,
    control_lambda,
    final_state_lambda,
):
    k = 0
    time_start = time.time()
    while True:

        # step 1
        eta, xi, v, s1 = eucl_convex_optimal_control_subproblem(
            q, u, tau, t_o, y_b, cos_theta_max, q_des,
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda,
            N,
            r
        )

        # step 2
        pen_cost = penalized_eucl_traj_cost(
            q, u, tau, t_o, y_b, cos_theta_max, q_des,
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda,
            N
        )
        trans_pen_cost = penalized_eucl_traj_cost(
            q + eta, u + xi, tau, t_o, y_b, cos_theta_max, q_des,
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda, 
            N
        )
        lin_cost = linearized_eucl_traj_cost(
            q, u, tau, t_o, y_b, cos_theta_max, q_des,
            eta, xi, v,s1,
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda,
            N
        )
        Delta_J = pen_cost - trans_pen_cost
        Delta_L = pen_cost - lin_cost
        if np.abs(Delta_J) < eps_tol:
            break
        else:
            rho_k = np.abs(Delta_J)/Delta_L
        if rho_k < rho0:
            r = r/alpha
        else:
            q = q + eta
            u = u + xi
            if rho_k < rho1:
                r = r/alpha
            elif rho_k >= rho2:
                r = r*beta
            
            r = max(r, rl)
            k = k + 1
    time_end = time.time()

    tot_time = time_end - time_start
    geo_cost = geo_traj_cost(
        q, u, tau, t_o, y_b, cos_theta_max, q_des,
        penalty_lambda, 
        state_lambda, 
        control_lambda, 
        final_state_lambda,
        N
    )
    eucl_cost = eucl_traj_cost(
        q, u, tau, t_o, y_b, cos_theta_max, q_des,
        penalty_lambda, 
        state_lambda, 
        control_lambda, 
        final_state_lambda,
        N
    )

    return tot_time, k, geo_cost, eucl_cost
    
def run_iSCvx_experiment(
    q, u, q_des, cos_theta_max, tau, N,
    r, rl, alpha, beta, eps_tol, rho0, rho1, rho2,
    t_o, y_b,
    penalty_lambda,
    state_lambda,
    control_lambda,
    final_state_lambda,
):
    k = 0
    time_start = time.time()
    while True:

        # step 1
        eta, xi, v, s1 = convex_optimal_control_subproblem(
            q, u, tau, t_o, y_b, cos_theta_max, q_des,
            penalty_lambda,
            state_lambda, 
            control_lambda, 
            final_state_lambda,
            N,
            r
        )

        # step 2
        pen_cost = penalized_geo_traj_cost(
            q, u, tau, t_o, y_b, cos_theta_max, q_des,
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda,
            N
        )
        trans_pen_cost = penalized_geo_traj_cost(
            translate_traj(q, eta, N), u + xi, tau, t_o, y_b, cos_theta_max, q_des, 
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda, 
            N
        )
        lin_cost = linearized_geo_traj_cost(
            q, u, tau, t_o, y_b, cos_theta_max, q_des,
            eta, xi, v, s1,
            penalty_lambda, 
            state_lambda, 
            control_lambda, 
            final_state_lambda,
            N
        )
        Delta_J = pen_cost - trans_pen_cost
        Delta_L = pen_cost - lin_cost
        if np.abs(Delta_J) < eps_tol:
            break
        else:
            rho_k = np.abs(Delta_J)/Delta_L
        if rho_k < rho0:
            r = r/alpha
        else:
            q = translate_traj(q, eta, N)
            u = u + xi
            if rho_k < rho1:
                r = r/alpha
            elif rho_k >= rho2:
                r = r*beta
            
            r = max(r, rl)
            k = k + 1

    time_end = time.time()

    tot_time = time_end - time_start
    geo_cost = geo_traj_cost(
        q, u, tau, t_o, y_b, cos_theta_max, q_des,
        penalty_lambda, 
        state_lambda, 
        control_lambda, 
        final_state_lambda,
        N
    )
    eucl_cost = eucl_traj_cost(
        q, u, tau, t_o, y_b, cos_theta_max, q_des,
        penalty_lambda, 
        state_lambda, 
        control_lambda, 
        final_state_lambda,
        N
    )

    return tot_time, k, geo_cost, eucl_cost


if __name__ == "__main__":
    penalty_lambda = 1e5
    state_lambda = 1
    control_lambda = .1
    final_state_lambda = 10

    t_o = np.array([1,0,0])
    y_b = np.array([1,0,0])

    r = 1
    rl = 0
    alpha = 2
    beta = 3.2
    eps_tol = 1e-5
    rho0 = 0
    rho1 = .25
    rho2 = .7

    theta_max = 30/180*np.pi
    tau = .1
    N = 30
    cos_theta_max = np.cos(theta_max)
    q, u, q_des = experiment_init_traj(theta_max, t_o, y_b, N)

    tot_time, k, geo_cost, eucl_cost = run_iSCvx_experiment(
        q, u, q_des, cos_theta_max, tau, N,
        r, rl, alpha, beta, eps_tol, rho0, rho1, rho2,
        t_o, y_b,
        penalty_lambda,
        state_lambda,
        control_lambda,
        final_state_lambda,
    )
    print(f"iSCvx, time: {tot_time}, num iters: {k}, geo cost: {geo_cost}, eucl_cost: {eucl_cost}")

    tot_time, k, geo_cost, eucl_cost = run_SCvx_experiment(
        q, u, q_des, cos_theta_max, tau, N,
        r, rl, alpha, beta, eps_tol, rho0, rho1, rho2,
        t_o, y_b,
        penalty_lambda,
        state_lambda,
        control_lambda,
        final_state_lambda
    )
    print(f"SCvx, time: {tot_time}, num iters: {k}, geo cost: {geo_cost}, eucl_cost: {eucl_cost}")