import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from tqdm import tqdm

from rootfinder import roots_z, evaluate_cheb_poly_mat, cayley_resultant
from jac import get_jacobian_func_numeric


def estimate_z_error(R: np.ndarray, z_hat: float, h: float) -> float:
    """Estimates the physical z-error from the resultant matrix polynomial."""
    R_eff = R 

    Rz = evaluate_cheb_poly_mat(R_eff, z_hat)

    try:
        U, S, Vh = svd(Rz)
        w = U[:, -1]
        v = Vh.conj().T[:, -1]
        smin = S[-1]
    except Exception:
        return np.nan

    dz = max(1e-7, np.sqrt(np.finfo(float).eps)) * max(1, abs(z_hat))
    Rp = evaluate_cheb_poly_mat(R_eff, z_hat + dz)
    Rm = evaluate_cheb_poly_mat(R_eff, z_hat - dz)
    R_prime_num = (Rp - Rm) / (2 * dz)

    den = abs(w.conj().T @ (R_prime_num @ v))
    den = max(den, np.finfo(float).eps)

    calibration_factor = 1e1
    dz_unit_est = smin / den
    dz_phys_est = (h / 2) * dz_unit_est * calibration_factor
    return dz_phys_est


PLOT_DIST = True
DEGREE_APPROX = 2
SHOULD_PRECONDITION = True
del_param = 1e-3
h_vals = np.logspace(-6, 4, 80)
NUM_Q, NUM_ROOT_LOC = 1, 50

f1 = lambda x, y, z: (1 / 3 + del_param) * x + 1 / 3 * y + (1 / 3 - del_param) * z
f2 = lambda x, y, z: (1 / 3 + del_param) * x + (1 / 3 - del_param) * y + 1 / 3 * z
f3 = lambda x, y, z: 1 / 3 * x + (1 / 3 + del_param) * y + (1 / 3 - del_param) * z
functions = [f1, f2, f3]

dist_vals = np.full((len(h_vals), NUM_Q, NUM_ROOT_LOC), np.nan)
predicted_dist_vals = np.full((len(h_vals), NUM_Q, NUM_ROOT_LOC), np.nan)

for i_Q in range(NUM_Q):
    J_func = get_jacobian_func_numeric(functions)
    J = J_func(0.0, 0.0, 0.0)
    J_inv = np.linalg.inv(J)
    cond_root = np.linalg.norm(J_inv)
    cond_eig = 1.0 / np.abs(np.linalg.det(J))

    progress_bar = tqdm(range(NUM_ROOT_LOC), desc=f"Q #{i_Q+1}/{NUM_Q}", leave=False)
    for i_root_loc in progress_bar:
        expected_root = (2 * np.random.rand(3) - 1) / 100.0

        for k, h in enumerate(h_vals):
            a, b = expected_root - h/2, expected_root + h/2
            scale, shift = (b - a) / 2.0, (b + a) / 2.0

            def remap(u, idx):
                return scale[idx] * u + shift[idx]

            e0, e1, e2 = expected_root

            f_remapped = [
                (lambda g: (lambda x, y, z, g=g, e0=e0, e1=e1, e2=e2:
                            g(remap(x, 0) - e0,
                              remap(y, 1) - e1,
                              remap(z, 2) - e2)))(g)
                for g in functions
            ]

            J_func_transformed = get_jacobian_func_numeric(f_remapped)
            J_transformed = J_func_transformed(0.0, 0.0, 0.0)
            J_inv_transformed = np.linalg.inv(J_transformed)
            kappa_root_transformed = np.linalg.norm(J_inv_transformed)
            kappa_eig_transformed  = 1.0 / abs(np.linalg.det(J_transformed))

            if SHOULD_PRECONDITION:
                def lin_combo(row, g_list):
                    return (lambda x, y, z, row=row, g_list=g_list:
                            row[0]*g_list[0](x,y,z) + row[1]*g_list[1](x,y,z) + row[2]*g_list[2](x,y,z))
                p = [lin_combo(J_inv_transformed[i, :], f_remapped) for i in range(3)]
            else:
                p = f_remapped

            roots_z_unit, _, _, _, R = roots_z(p, DEGREE_APPROX)

            if roots_z_unit.size == 0:
                continue

            roots_z_physical = remap(roots_z_unit, 2)
            dists = np.abs(roots_z_physical - expected_root[2])
            min_dist_idx = np.argmin(dists)

            dist_vals[k, i_Q, i_root_loc] = dists[min_dist_idx]
            z_unit_closest = roots_z_unit[min_dist_idx]

            pred = estimate_z_error(R, z_unit_closest, h)
            u = np.finfo(float).eps
            predicted_dist_vals[k, i_Q, i_root_loc] = max(pred, u * kappa_root_transformed)

# --- Plotting ---
if PLOT_DIST:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_dist = np.nanmean(dist_vals, axis=(1, 2))
    mean_pred_dist = np.nanmean(predicted_dist_vals, axis=(1, 2))

    ax.loglog(h_vals, mean_dist, "o-", label=r"Observed Error $|z - z_*|$")
    ax.loglog(h_vals, mean_pred_dist, "s--", label=r"Predicted Error")

    ax.axhline(np.finfo(float).eps * kappa_root_transformed, linestyle='--', color='r',
               label=r'Error floor $\approx u\cdot\kappa_{\mathrm{root,transformed}}$')
    ax.axhline(
        cond_eig * 1e-16,
        color="g",
        linestyle="-.",
        label=r"Error Floor $\approx u \cdot \kappa_{eig}$",
    )

    ax.set_ylim(bottom=1e-18)  # Ensure floors are visible
    ax.set_xlabel("Box Width $h$")
    ax.set_ylabel("Error in $z$-component")
    ax.set_title(f"Effect of Domain Shrinking (δ = {del_param:.0e})")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
