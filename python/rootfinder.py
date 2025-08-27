import numpy as np
from scipy import linalg
from scipy.fft import dctn, dct
from typing import Callable, Tuple, List
from cheby import *

def dct2_cheb_along_axis(arr, axis_len, axis):
    return (2.0 / axis_len) * dct(arr, type=2, axis=axis, norm=None)

def cayley_resultant(
    f: List[Callable], n: int, n_test_pts: int = None
) -> Tuple[np.ndarray, float]:
    """Computes the matricised 5D Cayley resultant and an accuracy estimate."""
    if n_test_pts is None:
        n_test_pts = n
    f1, f2, f3 = f

    def f_cayley(s1, s2, t1, t2, z):
        num = (f1(s1,s2,z)*f2(t1,s2,z)*f3(t1,t2,z) +
               f2(s1,s2,z)*f3(t1,s2,z)*f1(t1,t2,z) +
               f3(s1,s2,z)*f1(t1,s2,z)*f2(t1,t2,z) -
               f3(s1,s2,z)*f2(t1,s2,z)*f1(t1,t2,z) -
               f2(s1,s2,z)*f1(t1,s2,z)*f3(t1,t2,z) -
               f1(s1,s2,z)*f3(t1,s2,z)*f2(t1,t2,z))
        den = (s1 - t1) * (s2 - t2)
        return num / den

    n_s1, n_s2 = n, 2 * n
    n_t1, n_t2 = 2 * n, n
    n_z = 3 * n + 1
    dims = (n_s1, n_s2, n_t1, n_t2, n_z)

    s1_nodes = np.cos((2 * np.arange(1, n_s1 + 1) - 1) * np.pi / (2 * n_s1))
    s2_nodes = np.cos((2 * np.arange(1, n_s2 + 1) - 1) * np.pi / (2 * n_s2))
    t1_nodes, t2_nodes = s2_nodes, s1_nodes
    z_nodes = np.cos((2 * np.arange(1, n_z + 1) - 1) * np.pi / (2 * n_z))

    S1, S2, T1, T2, Z = np.meshgrid(s1_nodes, s2_nodes, t1_nodes, t2_nodes, z_nodes, indexing='ij')
    f_vals = f_cayley(S1, S2, T1, T2, Z)

    A = f_vals.astype(np.float64)
    A = dct2_cheb_along_axis(A, n_s1, axis=0)
    A = dct2_cheb_along_axis(A, n_s2, axis=1)
    A = dct2_cheb_along_axis(A, n_t1, axis=2)
    A = dct2_cheb_along_axis(A, n_t2, axis=3)
    A = dct2_cheb_along_axis(A, n_z,  axis=4)

    A[0, :, :, :, :] *= 0.5   # s1 axis, T0
    A[:, 0, :, :, :] *= 0.5   # s2 axis, T0
    A[:, :, 0, :, :] *= 0.5   # t1 axis, T0
    A[:, :, :, 0, :] *= 0.5   # t2 axis, T0
    A[:, :, :, :, 0] *= 0.5   # z  axis, T0

    R = A.reshape((n_s1 * n_s2, n_t1 * n_t2, n_z), order='F')

    s1t = np.linspace(-1, 1, n_test_pts)
    s2t = np.linspace(-1, 1, 2 * n_test_pts)
    zt = np.linspace(-1, 1, 3 * n_test_pts + 1)
    
    q1, q2, q3, q4, q5 = np.meshgrid(s1t, s2t, s2t, s1t, zt, indexing='ij')
    
    f_exact = f_cayley(q1, q2, q3, q4, q5)
    f_approx = cheby_5d_evaluate(A, [s1t, s2t, s2t, s1t, zt])
    
    bad_mask = ~np.isfinite(f_exact) | ~np.isfinite(f_approx)
    approx_err = np.max(np.abs(f_exact[~bad_mask] - f_approx[~bad_mask]))

    return R, approx_err

def roots_z(
    f: List[Callable], max_degree: int
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Finds the z-roots of the system using the Cayley resultant."""
    R, approx_err = cayley_resultant(f, max_degree)
    
    z_length = R.shape[2]
    for i in range(R.shape[2] - 1, -1, -1):
        if np.linalg.norm(R[:, :, i], ord='fro') == 0:
            z_length -= 1
        else:
            break
    if z_length <= 1:
        return np.array([]), np.array([]), approx_err, np.array([])
    R_eff = R[:, :, :z_length]

    n_dim = R_eff.shape[0]
    
    eig_err_all = np.array([])

    if z_length == 2:
        R0, R1 = R_eff[:, :, 0], R_eff[:, :, 1]
        try:
            D, V = linalg.eig(C2, -C1)
            residuals = C2 @ V + C1 @ V @ np.diag(D)
            eig_err_all = z_length * np.linalg.norm(residuals, axis=0) / np.maximum(np.linalg.norm(V, axis=0), 1e-16)
        except linalg.LinAlgError:
            D, V = np.array([]), np.array([])
    else:
        m = n_dim * (z_length - 1)
        C1 = -2 * np.eye(m)
        C1[:n_dim, :n_dim] = -np.eye(n_dim)
        C1[-n_dim:, -n_dim:] = 2 * R_eff[:, :, -1]
        C2 = np.zeros((m, m))
        if z_length > 2:
            C2[n_dim:, :-n_dim] += np.eye(m - n_dim)
            C2[:-n_dim, n_dim:] += np.eye(m - n_dim)
        C2[-n_dim:, -2*n_dim:-n_dim] = -R_eff[:, :, -1]
        for i in range(z_length - 1):
            C2[-n_dim:, i*n_dim:(i+1)*n_dim] += R_eff[:, :, i]
        try:
            D, V = linalg.eig(C2, -C1)
            residuals = C2 @ V - C1 @ V @ np.diag(D)
            eig_err_all = z_length * np.linalg.norm(residuals, axis=0) / np.maximum(np.linalg.norm(V, axis=0), 1e-16)
        except linalg.LinAlgError:
            D, V = np.array([]), np.array([])

    tol_imag = 1e-12
    valid_mask = (np.abs(np.imag(D)) <= tol_imag) & (np.abs(np.real(D)) <= 1.0)
    rts = np.real(D[valid_mask])
    V_filtered = V[:, valid_mask]
    eig_err = eig_err_all[valid_mask] if eig_err_all.size > 0 else np.array([])
    
    return rts, V_filtered, approx_err, eig_err, R