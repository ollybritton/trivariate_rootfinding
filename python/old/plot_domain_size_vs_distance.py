"""
Python version of the Noferini–Townsend “devastating example”
that studies how shrinking the domain affects the z–root accuracy
and the size of the Cayley coordinate matrices.

Requirements
------------
* numpy
* matplotlib
* the fixed `roots_z` (and its helpers) from the previous answer
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Utilities ------------------------------------------------------------
# ---------------------------------------------------------------------
def remapper(a, b):
    """
    Return a vectorised function  r(x, idx)  that maps x ∈ [-1,1] into
    the interval [a[idx], b[idx]].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0

    def r(x, idx):
        return scale[idx] * x + shift[idx]

    return r, scale, shift
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Problem set‑up -------------------------------------------------------
# ---------------------------------------------------------------------
# Random orthogonal matrix  (frozen to match MATLAB numbers exactly)
Q = np.array([
    [-0.30565725,  0.94407778, -0.12365595],
    [-0.94346673, -0.31779850, -0.09420534],
    [-0.12823484,  0.08787073,  0.98784339]
])

sigma = 1e-1                                 # easiest: sigma = 1e‑1

# Trivariate polynomials
f1 = lambda x, y, z: x**2 + sigma * (Q[0, 0]*x + Q[0, 1]*y + Q[0, 2]*z)
f2 = lambda x, y, z: y**2 + sigma * (Q[1, 0]*x + Q[1, 1]*y + Q[1, 2]*z)
f3 = lambda x, y, z: z**2 + sigma * (Q[2, 0]*x + Q[2, 1]*y + Q[2, 2]*z)

# Chosen root; we will *translate* the problem so that this is the origin
expected = np.array([0.9834675, -0.2374, -0.36024])

# Translated polynomials (root at the new origin)
p1 = lambda x, y, z: f1(x - expected[0], y - expected[1], z - expected[2])
p2 = lambda x, y, z: f2(x - expected[0], y - expected[1], z - expected[2])
p3 = lambda x, y, z: f3(x - expected[0], y - expected[1], z - expected[2])

# ---------------------------------------------------------------------
# Conditioning at the root (analytic Jacobian) ------------------------
# ---------------------------------------------------------------------
J = sigma * Q                               # Jacobian  J(0,0,0)
cond = np.linalg.norm(np.linalg.inv(J), 2)  # == 1 / sigma  (orthogonal Q)
print(f"Condition number (2‑norm) at the root: {cond:.1e}")

# ---------------------------------------------------------------------
# Experiments for shrinking boxes -------------------------------------
# ---------------------------------------------------------------------
h_vals = np.logspace(0, -9, 25)             # widths of sub‑regions

dist_vals                       = np.full_like(h_vals, np.nan)
predicted_dist_vals             = np.full_like(h_vals, np.nan)
coord_mat_norms                 = np.full_like(h_vals, np.nan)
predicted_coord_mat_norms       = np.full_like(h_vals, np.nan)

# import the root finder AFTER we have set up the environment
from roots import roots_z                  # <- your earlier fixed version

for k, h in enumerate(h_vals):
    # --------------------------------------------------------------
    # Axis‑aligned cube that is *not* perfectly centred on the root
    a = expected + np.array([-h/3,  +h/3, -h/4])
    b = expected + np.array([+2*h/3, -2*h/3, +3*h/4])

    remap, scale, shift = remapper(a, b)

    # --------------------------------------------------------------
    # Scale each polynomial so that on the eight vertices of the cube
    # its magnitude is ≤ 1 (this helps numerical stability)
    v = np.array([-1.0, 1.0])
    X, Y, Z = np.meshgrid(v, v, v, indexing='ij')
    xv = remap(X.ravel(), 0)
    yv = remap(Y.ravel(), 1)
    zv = remap(Z.ravel(), 2)

    c1 = np.max(np.abs(p1(xv, yv, zv)))
    c2 = np.max(np.abs(p2(xv, yv, zv)))
    c3 = np.max(np.abs(p3(xv, yv, zv)))

    p1_u = lambda x, y, z: p1(remap(x, 0), remap(y, 1), remap(z, 2)) / c1
    p2_u = lambda x, y, z: p2(remap(x, 0), remap(y, 1), remap(z, 2)) / c2
    p3_u = lambda x, y, z: p3(remap(x, 0), remap(y, 1), remap(z, 2)) / c3

    # --------------------------------------------------------------
    # Locate the roots in unit coordinates
    roots_unit, R = roots_z(p1_u, p2_u, p3_u,
                            a=[-1, -1, -1],
                            b=[ 1,  1,  1],
                            max_degree=3)

    # --------------------------------------------------------------
    # ‖A_i‖₂ of the Cayley pencil
    if R.size:
        Ai_norms = [np.linalg.norm(R[:, :, i], 2) for i in range(R.shape[2])]
        coord_mat_norms[k] = max(Ai_norms)
    predicted_coord_mat_norms[k] = 1.0       # place‑holder (as in MATLAB)

    # --------------------------------------------------------------
    # Distance of computed z‑roots from the expected value
    if roots_unit.size == 0:
        warnings.warn(f"No roots found for h = {h:.3g} – recording NaN")
    else:
        z_physical = remap(roots_unit, 2)
        dist_vals[k] = np.min(np.abs(z_physical - expected[2]))
        predicted_dist_vals[k] = h            # simple “h‑line” prediction

# ---------------------------------------------------------------------
# Plot 1 :  box width  vs  z‑error ------------------------------------
# ---------------------------------------------------------------------
plt.figure(figsize=(6.0, 4.0))
plt.loglog(h_vals, dist_vals, "o-", lw=1.2, ms=6, label="observed error")
plt.loglog(h_vals, predicted_dist_vals, "o-", lw=1.2, ms=6,
           label="predicted $\\propto h$")

plt.grid(True, which="both")
plt.xlabel(r"box width $h$")
plt.ylabel(r"error in $z$–component")
plt.title(fr"Effect of shrinking the domain ( $\sigma = {sigma:.0e}$ )")

plt.axvline(1/cond, ls="--", color="r",
            label=r"$h \approx 1/\mathrm{cond}$")
plt.axhline(cond * 1e-15,  ls="--", color="r",
            label=r"$\mathrm{err} \approx u\,\mathrm{cond}$")
plt.axhline(cond**3 * 1e-15, ls="--", color="r",
            label=r"$\mathrm{err} \approx u\,\mathrm{cond}^3$")

plt.legend()
plt.tight_layout()

# ---------------------------------------------------------------------
# Plot 2 :  box width  vs  max‖A_i‖₂ -----------------------------------
# ---------------------------------------------------------------------
plt.figure(figsize=(6.0, 4.0))
plt.loglog(h_vals, coord_mat_norms, "o-", lw=1.2, ms=6,
           label=r"observed $\max\|A_i\|_2$")
plt.loglog(h_vals, predicted_coord_mat_norms, "o-", lw=1.2, ms=6,
           label=r"predicted (placeholder)")

plt.grid(True, which="both")
plt.xlabel(r"box width $h$")
plt.ylabel(r"$\max\|A_i\|_2$")
plt.title("Effect of shrinking the domain on $\\max\\|A_i\\|_2$")
plt.legend()
plt.tight_layout()

plt.show()
