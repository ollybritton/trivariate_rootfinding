import numpy as np
from numpy.fft import fftn

def cheby_5D_interpolate(f):
    """
    5D Chebyshev interpolation: from values on Chebyshev points to
    Chebyshev coefficients, following the MATLAB implementation exactly.

    Args:
        f (np.ndarray): shape (n1, n2, n3, n4, n5).

    Returns:
        np.ndarray: shape (n1, n2, n3, n4, n5) of Chebyshev coefficients.
    """
    n1, n2, n3, n4, n5 = f.shape

    # Build symmetric extension
    f_ext = np.zeros((2*n1, 2*n2, 2*n3, 2*n4, 2*n5), dtype=complex)
    f_ext[:n1, :n2, :n3, :n4, :n5] = f

    # Mirror along axis 0 and 1
    f_ext[n1:2*n1, :n2, :n3, :n4, :n5] = np.flip(f, axis=0)
    f_ext[:n1, n2:2*n2, :n3, :n4, :n5] = np.flip(f, axis=1)
    f_ext[n1:2*n1, n2:2*n2, :n3, :n4, :n5] = np.flip(np.flip(f, axis=0), axis=1)

    # Mirror along axis 2, 3, 4
    f_ext[:, :, n3:2*n3, :, :n5] = np.flip(f_ext[:, :, :n3, :, :n5], axis=2)
    f_ext[:, :, :, n4:2*n4, :n5] = np.flip(f_ext[:, :, :, :n4, :n5], axis=3)
    f_ext[:, :, :, :, n5:2*n5] = np.flip(f_ext[:, :, :, :, :n5], axis=4)

    # Compute residuals
    def get_residuals(n):
        res = np.arange(n) * np.pi / (2 * n)
        return np.concatenate([res, [0], -np.flip(res[1:])])

    res1 = get_residuals(n1)
    res2 = get_residuals(n2)
    res3 = get_residuals(n3)
    res4 = get_residuals(n4)
    res5 = get_residuals(n5)

    # Construct residuals grid
    r1 = np.exp(-1j * res1)[:, None, None, None, None]
    r2 = np.exp(-1j * res2)[None, :, None, None, None]
    r3 = np.exp(-1j * res3)[None, None, :, None, None]
    r4 = np.exp(-1j * res4)[None, None, None, :, None]
    r5 = np.exp(-1j * res5)[None, None, None, None, :]
    residuals = r1 * r2 * r3 * r4 * r5

    # FFT to get Chebyshev coefficients
    cheby_coeffs = fftn(f_ext) / (32 * n1 * n2 * n3 * n4 * n5)
    cheby_coeffs *= residuals

    # Fold the coefficients (symmetry)
    cheby_coeffs[:, 1:n2, :, :, :] += np.flip(cheby_coeffs[:, n2+1:, :, :, :], axis=1)
    cheby_coeffs[1:n1, :n2, :, :, :] += np.flip(cheby_coeffs[n1+1:, :n2, :, :, :], axis=0)
    cheby_coeffs[:n1, :n2, 1:n3, :, :] += np.flip(cheby_coeffs[:n1, :n2, n3+1:, :, :], axis=2)
    cheby_coeffs[:n1, :n2, :n3, 1:n4, :] += np.flip(cheby_coeffs[:n1, :n2, :n3, n4+1:, :], axis=3)
    cheby_coeffs[:n1, :n2, :n3, :n4, 1:n5] += np.flip(cheby_coeffs[:n1, :n2, :n3, :n4, n5+1:], axis=4)

    return np.real(cheby_coeffs[:n1, :n2, :n3, :n4, :n5])

def cayley_resultant(f1, f2, f3, n):
    n_s1 = n
    n_s2 = 2 * n
    n_t1 = 2 * n
    n_t2 = n
    n_z = 3 * n + 1

    cg = lambda m: np.cos((2 * np.arange(1, m + 1) - 1) * np.pi / (2 * m))
    s1_pts = cg(n_s1)
    s2_pts = cg(n_s2)
    t1_pts = s2_pts
    t2_pts = s1_pts
    z_pts = cg(n_z)

    p1, p2, p3, p4, p5 = np.meshgrid(s1_pts, s2_pts, t1_pts, t2_pts, z_pts, indexing='ij')

    def f_cayley(s1, s2, t1, t2, z):
        num = (f1(s1, s2, z) * f2(t1, s2, z) * f3(t1, t2, z) +
               f2(s1, s2, z) * f3(t1, s2, z) * f1(t1, t2, z) +
               f3(s1, s2, z) * f1(t1, s2, z) * f2(t1, t2, z) -
               f3(s1, s2, z) * f2(t1, s2, z) * f1(t1, t2, z) -
               f2(s1, s2, z) * f1(t1, s2, z) * f3(t1, t2, z) -
               f1(s1, s2, z) * f3(t1, s2, z) * f2(t1, t2, z))
        return num / ((s1 - t1) * (s2 - t2))

    f_vals = f_cayley(p1, p2, p3, p4, p5)
    f_grid = f_vals.reshape(n_s1, n_s2, n_t1, n_t2, n_z)

    A = cheby_5D_interpolate(f_grid).copy(order='F')  # <-- new

    R = A.reshape(n_s1 * n_s2, n_s1 * n_s2, n_z, order='F')
    return R, n_s1, n_s2, n_t1, n_t2, n_z
