# rootfinder.py
import numpy as np
from typing import Callable, Tuple
from scipy import linalg
from scipy.fft import dct
from cheby import cheby_5d_evaluate, evaluate_cheb_poly_mat


def _eval_f_broadcast(f: Callable[[np.ndarray], np.ndarray],
                      X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Evaluate f on the broadcasted grid (X, Y, Z), where f expects inputs with
    shape (..., 3) and returns outputs with shape (..., 3).

    If f is not natively vectorised, we fall back to a batched loop that
    still keeps the (..., 3) contract and avoids splitting into components.
    """
    P = np.stack(np.broadcast_arrays(X, Y, Z), axis=-1)  # shape S + (3,)
    try:
        F = f(P)
        if F.shape != P.shape:
            raise ValueError("f returned wrong shape; expected (..., 3).")
        return F
    except Exception:
        P2 = P.reshape(-1, 3)
        out_list = [f(P2[i]) for i in range(P2.shape[0])]
        F = np.stack(out_list, axis=0).reshape(P.shape)
        return F


def _dct2_cheb_along_axis(arr: np.ndarray, axis_len: int, axis: int) -> np.ndarray:
    return (2.0 / axis_len) * dct(arr, type=2, axis=axis, norm=None)


def cayley_resultant(
    f: Callable[[np.ndarray], np.ndarray], n: int, n_test_pts: int = None
) -> Tuple[np.ndarray, float]:
    """
    Compute the matricised 5-D Cayley resultant R and an accuracy estimate.

    Parameters
    ----------
    f : callable
        Maps points with shape (..., 3) -> (..., 3). The last axis indexes components (f1,f2,f3).
        If f is not natively vectorised, it will be evaluated in a batched fallback.
    n : int
        Base degree parameter.
    n_test_pts : int, optional
        Test grid density per s1-dimension for accuracy check (default: n).

    Returns
    -------
    R : np.ndarray
        Resultant coefficient slices, shape (n_s1*n_s2, n_t1*n_t2, n_z).
    approx_err : float
        Max-abs interpolation error on a uniform test grid in [-1,1]^5.
    """
    if n_test_pts is None:
        n_test_pts = n

    n_s1, n_s2 = n, 2 * n
    n_t1, n_t2 = 2 * n, n
    n_z = 3 * n + 1

    s1 = np.cos((2 * np.arange(1, n_s1 + 1) - 1) * np.pi / (2 * n_s1))
    s2 = np.cos((2 * np.arange(1, n_s2 + 1) - 1) * np.pi / (2 * n_s2))
    t1, t2 = s2, s1
    z = np.cos((2 * np.arange(1, n_z + 1) - 1) * np.pi / (2 * n_z))

    S1, S2, T1, T2, Z = np.meshgrid(s1, s2, t1, t2, z, indexing="ij")

    F_s1s2z = _eval_f_broadcast(f, S1, S2, Z)    # (..., 3)
    F_t1s2z = _eval_f_broadcast(f, T1, S2, Z)    # (..., 3)
    F_t1t2z = _eval_f_broadcast(f, T1, T2, Z)    # (..., 3)

    f1_s1s2z = F_s1s2z[..., 0]
    f2_s1s2z = F_s1s2z[..., 1]
    f3_s1s2z = F_s1s2z[..., 2]

    f1_t1s2z = F_t1s2z[..., 0]
    f2_t1s2z = F_t1s2z[..., 1]
    f3_t1s2z = F_t1s2z[..., 2]

    f1_t1t2z = F_t1t2z[..., 0]
    f2_t1t2z = F_t1t2z[..., 1]
    f3_t1t2z = F_t1t2z[..., 2]

    num = (
        f1_s1s2z * f2_t1s2z * f3_t1t2z
        + f2_s1s2z * f3_t1s2z * f1_t1t2z
        + f3_s1s2z * f1_t1s2z * f2_t1t2z
        - f3_s1s2z * f2_t1s2z * f1_t1t2z
        - f2_s1s2z * f1_t1s2z * f3_t1t2z
        - f1_s1s2z * f3_t1s2z * f2_t1t2z
    )
    den = (S1 - T1) * (S2 - T2)
    f_vals = num / den  # shape (n_s1, n_s2, n_t1, n_t2, n_z)

    A = f_vals.astype(np.float64, copy=False)
    A = _dct2_cheb_along_axis(A, n_s1, axis=0)
    A = _dct2_cheb_along_axis(A, n_s2, axis=1)
    A = _dct2_cheb_along_axis(A, n_t1, axis=2)
    A = _dct2_cheb_along_axis(A, n_t2, axis=3)
    A = _dct2_cheb_along_axis(A, n_z,  axis=4)

    A[0, :, :, :, :] *= 0.5
    A[:, 0, :, :, :] *= 0.5
    A[:, :, 0, :, :] *= 0.5
    A[:, :, :, 0, :] *= 0.5
    A[:, :, :, :, 0] *= 0.5

    R = A.reshape((n_s1 * n_s2, n_t1 * n_t2, n_z), order="F")

    s1t = np.linspace(-1, 1, n_test_pts)
    s2t = np.linspace(-1, 1, 2 * n_test_pts)
    zt = np.linspace(-1, 1, 3 * n_test_pts + 1)
    Q1, Q2, Q3, Q4, Q5 = np.meshgrid(s1t, s2t, s2t, s1t, zt, indexing="ij")

    F_q1q2q5 = _eval_f_broadcast(f, Q1, Q2, Q5)
    F_q3q2q5 = _eval_f_broadcast(f, Q3, Q2, Q5)
    F_q3q4q5 = _eval_f_broadcast(f, Q3, Q4, Q5)

    num_t = (
        F_q1q2q5[..., 0] * F_q3q2q5[..., 1] * F_q3q4q5[..., 2]
        + F_q1q2q5[..., 1] * F_q3q2q5[..., 2] * F_q3q4q5[..., 0]
        + F_q1q2q5[..., 2] * F_q3q2q5[..., 0] * F_q3q4q5[..., 1]
        - F_q1q2q5[..., 2] * F_q3q2q5[..., 1] * F_q3q4q5[..., 0]
        - F_q1q2q5[..., 1] * F_q3q2q5[..., 0] * F_q3q4q5[..., 2]
        - F_q1q2q5[..., 0] * F_q3q2q5[..., 2] * F_q3q4q5[..., 1]
    )
    den_t = (Q1 - Q3) * (Q2 - Q4)
    f_exact = num_t / den_t

    f_approx = cheby_5d_evaluate(A, [s1t, s2t, s2t, s1t, zt])  # same shape
    mask = np.isfinite(f_exact) & np.isfinite(f_approx)
    approx_err = np.max(np.abs(f_exact[mask] - f_approx[mask])) if mask.any() else np.nan

    return R, approx_err


def roots_z(
    f: Callable[[np.ndarray], np.ndarray], max_degree: int
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Find z-roots via the Cayley resultant.

    Returns
    -------
    rts : np.ndarray
        Real roots in the unit z-domain, filtered to [-1,1].
    V_filtered : np.ndarray
        Right eigenvectors corresponding to retained roots.
    approx_err : float
        Interpolation error estimate from cayley_resultant.
    eig_err : np.ndarray
        Residual-based backward error for each retained root.
    R : np.ndarray
        The resultant coefficient slices.
    """
    R, approx_err = cayley_resultant(f, max_degree)

    z_len = R.shape[2]
    for i in range(R.shape[2] - 1, -1, -1):
        if np.linalg.norm(R[:, :, i], ord="fro") == 0:
            z_len -= 1
        else:
            break
    if z_len <= 1:
        return np.array([]), np.array([]), approx_err, np.array([]), R

    R_eff = R[:, :, :z_len]
    n = R_eff.shape[0]

    if z_len == 2:
        # Degree-1 in z: R0 + z R1
        R0, R1 = R_eff[:, :, 0], R_eff[:, :, 1]
        D, V = linalg.eig(R0, -R1)
        residuals = R0 @ V + R1 @ V @ np.diag(D)
        eig_err_all = z_len * np.linalg.norm(residuals, axis=0) / np.maximum(np.linalg.norm(V, axis=0), 1e-16)
    else:
        m = n * (z_len - 1)
        C1 = -2 * np.eye(m)
        C1[:n, :n] = -np.eye(n)
        C1[-n:, -n:] = 2 * R_eff[:, :, -1]

        C2 = np.zeros((m, m))
        C2[n:, :-n] += np.eye(m - n)
        C2[:-n, n:] += np.eye(m - n)
        C2[-n:, -2 * n : -n] = -R_eff[:, :, -1]
        for i in range(z_len - 1):
            C2[-n:, i * n : (i + 1) * n] += R_eff[:, :, i]

        D, V = linalg.eig(C2, -C1)
        residuals = C2 @ V - C1 @ V @ np.diag(D)
        eig_err_all = z_len * np.linalg.norm(residuals, axis=0) / np.maximum(np.linalg.norm(V, axis=0), 1e-16)

    tol_imag = 1e-12
    keep = (np.abs(np.imag(D)) <= tol_imag) & (np.abs(np.real(D)) <= 1.0)
    rts = np.real(D[keep])
    V_filtered = V[:, keep]
    eig_err = eig_err_all[keep] if eig_err_all.size else np.array([])

    return rts, V_filtered, approx_err, eig_err, R
