import unittest
import numpy as np
from numpy.linalg import norm
import sympy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import *

TOL = 1e-8  # Tolerance from your code

class TestQuaternionFunctions(unittest.TestCase):

    def setUp(self):
        # Common setup for random inputs
        self.q = rand_quat()  # Random unit quaternion
        self.x = rand_state()  # Random state with unit quaternion at x[6:10]

        while True:
            self.w = np.random.randn(3)  # Random 3-vector for exp/log tests
            if norm(self.w) < np.pi:
                break

    def test_rand_tangent_quat_perpendicular(self):
        """Test that dq from rand_tangent_quat is perpendicular to q."""
        dq = rand_tangent_quat(self.q)
        dot_product = np.dot(self.q, dq)
        self.assertAlmostEqual(dot_product, 0, delta=1e-10, 
                              msg="dq is not perpendicular to q")

    def test_rand_tangent_state_perpendicular(self):
        """Test that dx[6:10] is perpendicular to x[6:10] in rand_tangent_state."""
        dx = rand_tangent_state(self.x)
        q = self.x[6:10]
        dq = dx[6:10]
        dot_product = np.dot(q, dq)
        self.assertAlmostEqual(dot_product, 0, delta=1e-10, 
                              msg="dx[6:10] is not perpendicular to x[6:10]")

    def test_exp_zero(self):
        """Test that exp([0,0,0]) = [1,0,0,0]."""
        result = exp(np.array([0, 0, 0]))
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=8,
            err_msg="exp([0,0,0]) != [1,0,0,0]")

    def test_log_exp_identity(self):
        """Test that log(exp(w)) = w, aligning dimensions."""
        w = self.w
        exp_w = exp(w)
        log_exp_w = log(exp_w)
        # log returns [0, wx, wy, wz], so compare w to log_exp_w[1:]
        np.testing.assert_array_almost_equal(log_exp_w[1:], w, decimal=8,
            err_msg="log(exp(w)) != w")

    def test_exp_log_identity(self):
        """Test that exp(log(q)) = q, aligning dimensions."""
        q = self.q
        log_q = log(q)  # Returns [0, wx, wy, wz]
        exp_log_q = exp(log_q[1:])  # Pass only the vector part to exp
        np.testing.assert_array_almost_equal(exp_log_q, q, decimal=8,
            err_msg="exp(log(q)) != q")

    def test_log_identity(self):
        """Test that log([1,0,0,0]) = [0,0,0,0]."""
        q = np.array([1, 0, 0, 0])
        result = log(q)
        expected = np.array([0, 0, 0, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=8,
                                            err_msg="log([1,0,0,0]) != [0,0,0,0]")

    def test_quat_retract_inv_quat_retract(self):
        """Test that quat_retract and inv_quat_retract are inverses."""
        q = self.q
        dq = rand_tangent_quat(q)
        q_retracted = quat_retract(q, dq)
        dq_recovered = inv_quat_retract(q, q_retracted)
        np.testing.assert_array_almost_equal(dq, dq_recovered, decimal=8,
            err_msg="inv_quat_retract(quat_retract(q, dq)) != dq")

    def test_sp_quat_retract_sp_inv_quat_retract(self):
        q_sym = sp.Matrix(self.q)
        dq_sym = sp.Matrix(rand_tangent_quat(self.q))
        q_retracted = sp_quat_retract(q_sym, dq_sym)
        q_retracted_num = np.array(q_retracted.evalf(), dtype=float).flatten()
        dq_recovered = sp_inv_quat_retract(q_sym, sp.Matrix(q_retracted_num))
        dq_recovered_num = np.array(dq_recovered.evalf(), dtype=float).flatten()
        dq_sym_num = np.array(dq_sym.evalf(), dtype=float).flatten()  # Flatten dq_sym
        np.testing.assert_array_almost_equal(dq_recovered_num, dq_sym_num, decimal=6,
            err_msg="sp_inv_quat_retract(sp_quat_retract(q, dq)) != dq")
    
    def test_sp_exp_sp_log_identity(self):
        """Test that sp_log(sp_exp(w)) = w, aligning dimensions."""
        w_sym = sp.Matrix(self.w)
        exp_w = sp_exp(w_sym)
        log_exp_w = sp_log(exp_w)
        log_exp_w_num = np.array(log_exp_w.evalf(), dtype=float).flatten()
        # Compare w to log_exp_w[1:] (vector part)
        np.testing.assert_array_almost_equal(log_exp_w_num[1:], self.w, decimal=6,
            err_msg="sp_log(sp_exp(w)) != w")

    def test_sp_log_sp_exp_identity(self):
        """Test that sp_exp(sp_log(q)) = q, aligning dimensions."""
        q_sym = sp.Matrix(self.q)
        log_q = sp_log(q_sym)
        exp_log_q = sp_exp(log_q[1:])  # Pass vector part only
        exp_log_q_num = np.array(exp_log_q.evalf(), dtype=float).flatten()
        np.testing.assert_array_almost_equal(exp_log_q_num, self.q, decimal=6,
            err_msg="sp_exp(sp_log(q)) != q")

    def test_retract_inv_retract(self):
        """Test that retract and inv_retract are inverses."""
        x = self.x
        dx = rand_tangent_state(x)
        x_retracted = retract(x, dx)
        dx_recovered = inv_retract(x, x_retracted)
        np.testing.assert_array_almost_equal(dx, dx_recovered, decimal=8,
            err_msg="inv_retract(retract(x, dx)) != dx")


    def test_sp_grad_sinc(self):
        """Test that sp_grad_sinc matches the numerical gradient of sp_sinc."""
        w = np.random.randn(3)  # Random 3-vector
        h = np.random.randn(3)  # Small perturbation

        
        # Convert to SymPy for symbolic evaluation
        w_sym = sp.Matrix(w)
        h_sym = sp.Matrix(h)
        
        # Numerical gradient: (sp_sinc(w + h) - sp_sinc(w)) / |h|
        sinc_w = sp_sinc(w_sym)
        sinc_w_plus_h = sp_sinc(w_sym + 1e-6*h_sym)
        num_grad = (sinc_w_plus_h - sinc_w) / 1e-6
        num_grad_num = float(num_grad.evalf())
        
        # Analytical gradient: sp_grad_sinc(w) @ h
        grad_sinc = sp_grad_sinc(w_sym)  # 3x1 Matrix
        anal_grad = grad_sinc.T @ h_sym  # 1x1 Matrix
        anal_grad_num = float(anal_grad[0].evalf())  # Extract scalar with [0]
        
        # Compare
        self.assertAlmostEqual(num_grad_num, anal_grad_num, delta=1e-5,
            msg="sp_grad_sinc does not match numerical gradient of sp_sinc"
        )

    def test_sp_dlog(self):
        """Test that sp_dlog matches the numerical differential of sp_log."""
        q = self.q  # Random unit quaternion
        dq = rand_tangent_quat(q)  # Perturbation direction
        
        # Convert to SymPy
        q_sym = sp.Matrix(q)
        dq_sym = sp.Matrix(dq)
        
        eps = 1e-8
        log_q = sp_log(q_sym)
        q_retracted = sp_quat_retract(q_sym, eps * dq_sym)
        log_q_retracted = sp_log(q_retracted)
        num_diff = (log_q_retracted - log_q) / eps
        num_diff_num = np.array(num_diff.evalf(), dtype=float).flatten()[1:]  # Vector part only
        
        # Analytical differential: sp_dlog(q, dq)
        dlog = sp_dlog(q_sym, dq_sym)
        anal_diff_num = np.array(dlog.evalf(), dtype=float).flatten()
        
        # Compare
        np.testing.assert_array_almost_equal(num_diff_num, anal_diff_num, decimal=5,
            err_msg=f"sp_dlog does not match numerical differential of sp_log"
        )


    def test_sp_dexp(self):
        """Test that sp_dexp matches the numerical differential of sp_exp."""
        w = np.random.randn(3)  # Random 3-vector
        eta = np.random.randn(3)  # Perturbation direction
        
        # Convert to SymPy
        w_sym = sp.Matrix(w)
        eta_sym = sp.Matrix(eta)
        
        # Numerical differential: (sp_exp(w + eps*eta) - sp_exp(w)) / eps
        eps = 1e-6
        exp_w = sp_exp(w_sym)
        exp_w_plus_eta = sp_exp(w_sym + eps * eta_sym)
        num_diff = (exp_w_plus_eta - exp_w) / eps
        num_diff_num = np.array(num_diff.evalf(), dtype=float).flatten()
        
        # Analytical differential: sp_dexp(w, eta)
        dexp = sp_dexp(w_sym, eta_sym)
        anal_diff_num = np.array(dexp.evalf(), dtype=float).flatten()
        
        # Compare
        np.testing.assert_array_almost_equal(num_diff_num, anal_diff_num, decimal=5,
            err_msg="sp_dexp does not match numerical differential of sp_exp"
        )

    def test_quat_inv_retract_identity(self):
        """Test that inv_quat_retract(q, q) = [0, 0, 0, 0]."""
        q = self.q  # Random unit quaternion from setUp
        dq = inv_quat_retract(q, q)
        expected = np.zeros(4)
        np.testing.assert_array_almost_equal(dq, expected, decimal=8,
            err_msg="inv_quat_retract(q, q) != [0, 0, 0, 0]")

    def test_sp_quat_inv_retract_identity(self):
        """Test that sp_inv_quat_retract(q, q) = [0, 0, 0, 0]."""
        q_sym = sp.Matrix(self.q)  # Random unit quaternion from setUp
        dq_sym = sp_inv_quat_retract(q_sym, q_sym)
        dq_num = np.array(dq_sym.evalf(), dtype=float).flatten()
        expected = np.zeros(4)
        np.testing.assert_array_almost_equal(dq_num, expected, decimal=6,
            err_msg="sp_inv_quat_retract(q, q) != [0, 0, 0, 0]")

    def test_inv_retract_identity(self):
        """Test that inv_retract(x, x) = zeros(13) with quaternion part [0, 0, 0, 0]."""
        x = self.x  # Random state vector from setUp
        dx = inv_retract(x, x)
        expected = np.zeros(13)
        np.testing.assert_array_almost_equal(dx, expected, decimal=8,
            err_msg="inv_retract(x, x) != zeros(13)")
        # Extra check for quaternion part
        np.testing.assert_array_almost_equal(dx[6:10], np.zeros(4), decimal=8,
            err_msg="inv_retract(x, x)[6:10] != [0, 0, 0, 0]")

    def test_sp_inv_retract_identity(self):
        """Test that sp_inv_retract(x, x) = zeros(13) with quaternion part [0, 0, 0, 0]."""
        x_sym = sp.Matrix(self.x)  # Random state vector from setUp
        dx_sym = sp_inv_retract(x_sym, x_sym)
        dx_num = np.array(dx_sym.evalf(), dtype=float).flatten()
        expected = np.zeros(13)
        np.testing.assert_array_almost_equal(dx_num, expected, decimal=6,
            err_msg="sp_inv_retract(x, x) != zeros(13)")
        # Extra check for quaternion part
        np.testing.assert_array_almost_equal(dx_num[6:10], np.zeros(4), decimal=6,
            err_msg="sp_inv_retract(x, x)[6:10] != [0, 0, 0, 0]")

    def test_inv_retract_retract_identity(self):
        """Test that inv_retract(x, y) = v implies y = retract(x, v)."""
        x = rand_state()  # Random state vector
        y = rand_state()  # Random target state vector
        v = inv_retract(x, y)
        y_reconstructed = retract(x, v)
        np.testing.assert_array_almost_equal(y, y_reconstructed, decimal=8,
            err_msg="retract(x, inv_retract(x, y)) != y")

    def test_d_inv_quat_retract_differential(self):
        """Test that d_inv_quat_retract is the differential of inv_quat_retract."""
        q = self.q  # Random unit quaternion from setUp
        p = rand_quat()  # Random target unit quaternion
        dp = rand_tangent_quat(p)  # Random tangent quaternion orthogonal to p
        eps = 1e-8
        # Numerical differential: (inv_quat_retract(q, p + eps*dp) - inv_quat_retract(q, p)) / eps
        p_perturbed = quat_retract(p, eps * dp)
        num_diff = (inv_quat_retract(q, p_perturbed) - inv_quat_retract(q, p)) / eps
        # Analytical differential: d_inv_quat_retract(q, p, dp)
        anal_diff = d_inv_quat_retract(q, p, dp)
        np.testing.assert_array_almost_equal(num_diff, anal_diff, decimal=5,
            err_msg="d_inv_quat_retract does not match numerical differential of inv_quat_retract")

    def test_d_inv_retract_differential(self):
        """Test that d_inv_retract is the differential of inv_retract."""
        x = self.x  # Random state vector from setUp
        z = rand_state()  # Random target state vector
        dz = rand_tangent_state(z)  # Random tangent state vector, with dz[6:10] orthogonal to z[6:10]
        eps = 1e-6
        # Numerical differential: (inv_retract(x, z + eps*dz) - inv_retract(x, z)) / eps
        z_perturbed = retract(z, eps * dz)
        num_diff = (inv_retract(x, z_perturbed) - inv_retract(x, z)) / eps
        # Analytical differential: d_inv_retract(x, z, dz)
        anal_diff = d_inv_retract(x, z, dz)
        np.testing.assert_array_almost_equal(num_diff, anal_diff, decimal=5,
            err_msg="d_inv_retract does not match numerical differential of inv_retract")

    def test_dlog_numerical(self):
        """Test that dlog matches the numerical differential of log."""
        while True:
            q = rand_quat()
            if abs(q[0] - 1) > 0.1 and np.linalg.norm(q[1:]) > 1e-6:  # Avoid edge cases
                break
        dq = rand_tangent_quat(q)
        eps = 1e-8  # Larger eps for numerical stability
        q_perturbed = quat_retract(q, eps * dq)
        num_diff = (log(q_perturbed)[1:] - log(q)[1:]) / eps
        anal_diff = dlog(q, dq)

        np.testing.assert_array_almost_equal(num_diff, anal_diff, decimal=5,
            err_msg="dlog does not match numerical differential of log")
        
    def test_log_correctness(self):
        """Test that log produces correct quaternion logarithms."""
        # Case 1: Identity quaternion q = [1, 0, 0, 0]
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        expected1 = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(log(q1), expected1, decimal=8,
            err_msg="log([1, 0, 0, 0]) != [0, 0, 0, 0]")

        # Case 2: 180-degree rotation around x-axis q = [0, 1, 0, 0]
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        expected2 = np.array([0.0, np.pi/2, 0.0, 0.0])  # theta = pi/2
        np.testing.assert_array_almost_equal(log(q2), expected2, decimal=8,
            err_msg="log([0, 1, 0, 0]) != [0, pi/2, 0, 0]")

        # Case 3: General quaternion q = [cos(theta/2), sin(theta/2)/sqrt(3), ...]
        theta = np.pi / 4  # 45 degrees
        q3 = np.array([np.cos(theta/2), np.sin(theta/2)/np.sqrt(3), np.sin(theta/2)/np.sqrt(3), np.sin(theta/2)/np.sqrt(3)])
        q3 = q3 / np.linalg.norm(q3)  # Ensure unit
        expected3 = np.array([0.0, theta/2/np.sqrt(3), theta/2/np.sqrt(3), theta/2/np.sqrt(3)])
        np.testing.assert_array_almost_equal(log(q3), expected3, decimal=8,
            err_msg="log(general quaternion) incorrect")

        # Case 4: Near identity with small vector part
        q4 = np.array([1.0 - 1e-6, 1e-6, 1e-6, 1e-6])
        q4 = q4 / np.linalg.norm(q4)  # Ensure unit
        result4 = log(q4)
        norm_v = np.linalg.norm(result4[1:])
        self.assertLess(norm_v, 1e-5, msg="log(near identity) produces large vector part")

if __name__ == "__main__":
    unittest.main()
