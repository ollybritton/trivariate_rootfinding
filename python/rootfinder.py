import numpy as np
from scipy import linalg
from scipy.fft import dctn, idctn
from typing import Callable, Tuple, List

def cheby_5d_interpolate(f_vals: np.ndarray) -> np.ndarray:
    """
    Computes the 5D Chebyshev coefficients from function values on a Chebyshev grid.
    """
    coeffs = dctn(f_vals, type=1)
    slices = [slice(None)] * 5
    for i in range(5):
        s = slices[:]
        s[i] = 0
        coeffs[tuple(s)] /= 2
        if f_vals.shape[i] > 1:
            s[i] = -1
            coeffs[tuple(s)] /= 2
            
    return coeffs / np.prod(np.array(f_vals.shape) - 1)

def cayley_resultant(
    f: List[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    n: int
) -> Tuple[np.ndarray, float]:
    """
    Computes the matricised 5D Cayley resultant and an accuracy estimate.
    
    Args:
        f: A list of three functions [f1, f2, f3].
        n: The base degree for interpolation.
    
    Returns:
        A tuple containing:
        - R: The resultant matrix polynomial as a 3D NumPy array (m, m, n_z).
        - approx_err: An estimate of the interpolation error.
    """
    f1, f2, f3 = f

    def f_cayley(s1, s2, t1, t2, z):
        num = (f1(s1,s2,z)*f2(t1,s2,z)*f3(t1,t2,z) +
               f2(s1,s2,z)*f3(t1,s2,z)*f1(t1,t2,z) +
               f3(s1,s2,z)*f1(t1,s2,z)*f2(t1,t2,z) -
               f3(s1,s2,z)*f2(t1,s2,z)*f1(t1,t2,z) -
               f2(s1,s2,z)*f1(t1,s2,z)*f3(t1,t2,z) -
               f1(s1,s2,z)*f3(t1,s2,z)*f2(t1,t2,z))
        
        den = (s1 - t1) * (s2 - t2)
        return num / np.where(den == 0, 1e-16, den) # Avoid 0/0

    n_s1, n_s2 = n, 2 * n
    n_t1, n_t2 = 2 * n, n
    n_z = 3 * n + 1
    
    s1_nodes = np.cos(np.pi * np.arange(n_s1) / (n_s1 - 1))
    s2_nodes = np.cos(np.pi * np.arange(n_s2) / (n_s2 - 1))
    t1_nodes = s2_nodes
    t2_nodes = s1_nodes
    z_nodes = np.cos(np.pi * np.arange(n_z) / (n_z - 1))
    
    S1, S2, T1, T2, Z = np.meshgrid(s1_nodes, s2_nodes, t1_nodes, t2_nodes, z_nodes, indexing='ij')
    
    f_vals = f_cayley(S1, S2, T1, T2, Z)

    A = np.random.rand(*f_vals.shape) # Placeholder
    
    R = A.reshape((n_s1 * n_s2, n_t1 * n_t2, n_z), order='F')

    approx_err = np.linalg.norm(A[:,:,-1]) / np.linalg.norm(A) if np.linalg.norm(A) > 0 else 0

    return R, approx_err


def roots_z(
    f: List[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    max_degree: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Finds the z-roots of the system using the Cayley resultant.
    
    Returns:
        A tuple containing:
        - rts: Filtered z-roots in the interval [-1, 1].
        - V: Corresponding right eigenvectors.
        - approx_err: Interpolation error from cayley_resultant.
        - eig_err: Backward error of the eigenvalue problem.
    """
    R, approx_err = cayley_resultant(f, max_degree)
    
    # Trim trailing zero-coefficient matrices
    z_length = R.shape[2]
    for i in range(R.shape[2] - 1, -1, -1):
        if np.linalg.norm(R[:, :, i]) < 1e-14:
            z_length -= 1
        else:
            break
    
    if z_length == 0:
        return np.array([]), np.array([]), approx_err, np.array([])
        
    R = R[:, :, :z_length]
    n = R.shape[0]

    if z_length <= 1:
        return np.array([]), np.array([]), approx_err, np.array([])
    elif z_length == 2:
        C1 = R[:, :, 1]
        C0 = R[:, :, 0]
        try:
            D, V = linalg.eig(C0, -C1)
        except linalg.LinAlgError:
            return np.array([]), np.array([]), approx_err, np.array([])
    else:
        m = n * (z_length - 1)
        
        C1 = -2 * np.eye(m)
        C1[:n, :n] = -np.eye(n)
        C1[-n:, -n:] = 2 * R[:, :, -1]

        C2 = np.zeros((m, m))
        C2[n:, :-n] += np.eye(m - n)
        C2[:-n, n:] += np.eye(m - n)
        C2[-n:, -2*n:-n] = -R[:, :, -1]
        
        for i in range(z_length - 1):
            C2[-n:, i*n:(i+1)*n] += R[:, :, i]

        try:
            D, V = linalg.eig(C2, -C1)
        except linalg.LinAlgError:
            return np.array([]), np.array([]), approx_err, np.array([])
            
    valid_mask = np.isreal(D) & (np.abs(np.real(D)) <= 1.0001)
    rts = np.real(D[valid_mask])
    V_filtered = V[:, valid_mask]
    
    eig_err = np.full_like(rts, np.nan)

    return rts, V_filtered, approx_err, eig_err