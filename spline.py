import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, List, Tuple


# ----------------------------
# Data structure
# ----------------------------
@dataclass
class PSplineFit:
    degree: int
    n_basis: int
    knots: np.ndarray
    coef: np.ndarray
    lam: float
    metric: float          # chi2 if yerr provided; RSS otherwise
    edf: float             # effective degrees of freedom
    bic: float             # BIC using edf
    cov: Optional[np.ndarray]

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        B = bspline_design_matrix(np.asarray(x_new, float), self.knots, self.degree)
        return B @ self.coef


# ----------------------------
# B-spline utilities (same as before)
# ----------------------------
def _clamped_knot_vector(x: np.ndarray, degree: int, n_basis: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        raise ValueError("x must span a non-zero range.")
    if n_basis < degree + 1:
        raise ValueError(f"n_basis must be >= degree+1 (= {degree+1}).")

    n_internal = n_basis - degree - 1
    if n_internal > 0:
        qs = np.linspace(0, 1, n_internal + 2)[1:-1]
        internal = np.quantile(x, qs)
        internal = np.unique(internal)
        if internal.size < n_internal:
            internal = np.linspace(xmin, xmax, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)

    t = np.concatenate([
        np.full(degree + 1, xmin),
        internal,
        np.full(degree + 1, xmax),
    ])
    return t


def bspline_design_matrix(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = int(degree)

    n_basis = len(t) - k - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector/degree combination.")

    B = np.zeros((x.size, n_basis), dtype=float)
    for j in range(n_basis):
        left, right = t[j], t[j + 1]
        if j == n_basis - 1:
            B[:, j] = (x >= left) & (x <= right)
        else:
            B[:, j] = (x >= left) & (x < right)

    for d in range(1, k + 1):
        B_new = np.zeros_like(B)
        for j in range(n_basis):
            denom1 = t[j + d] - t[j]
            denom2 = t[j + d + 1] - t[j + 1]

            term1 = 0.0
            if denom1 > 0:
                term1 = ((x - t[j]) / denom1) * B[:, j]

            term2 = 0.0
            if denom2 > 0 and (j + 1) < n_basis:
                term2 = ((t[j + d + 1] - x) / denom2) * B[:, j + 1]

            B_new[:, j] = term1 + term2
        B = B_new

    return B


# ----------------------------
# P-spline penalty + fitting
# ----------------------------
def second_difference_matrix(n: int) -> np.ndarray:
    """D such that (D @ c) are 2nd differences of coefficient vector c. Shape (n-2, n)."""
    if n < 3:
        raise ValueError("Need n >= 3 for second differences.")
    D = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _weights_from_yerr(y: np.ndarray, yerr: Optional[np.ndarray]) -> np.ndarray:
    if yerr is None:
        return np.ones_like(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if np.any(yerr <= 0):
        raise ValueError("All yerr must be > 0.")
    return 1.0 / yerr


def fit_pspline_fixed(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    degree: int,
    n_basis: int,
    lam: float,
) -> Tuple[np.ndarray, float, float, float, Optional[np.ndarray]]:
    """
    Fit penalized weighted least squares:
      minimize ||W(y - Bc)||^2 + lam * ||D c||^2
    Returns: coef, metric(chi2 or RSS), edf, bic, cov
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of the same length.")
    if lam < 0:
        raise ValueError("lam must be >= 0.")

    # knots + design
    knots = _clamped_knot_vector(x, degree=degree, n_basis=n_basis)
    B = bspline_design_matrix(x, knots, degree=degree)

    n = x.size
    p = B.shape[1]

    # weights (w = 1/sigma). We'll use W2 = diag(w^2)
    w = _weights_from_yerr(y, yerr)
    W2 = (w**2)

    # Penalized normal equations:
    # (B^T W2 B + lam D^T D) c = B^T W2 y
    BtW2 = B.T * W2  # each row i multiplied by W2
    A = BtW2 @ B
    b = BtW2 @ y

    if lam > 0 and p >= 3:
        D = second_difference_matrix(p)
        P = D.T @ D
        A = A + lam * P
    elif lam > 0 and p < 3:
        # Not enough params for 2nd-diff penalty; fall back to ridge-like identity
        A = A + lam * np.eye(p)

    # Solve
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Residual metric
    resid = y - (B @ coef)
    if yerr is None:
        metric = float(np.sum(resid**2))  # RSS
    else:
        metric = float(np.sum((resid / yerr) ** 2))  # chi^2

    # EDF = trace(S), where S = B (B^T W2 B + lam P)^-1 B^T W2
    # EDF = trace( (B^T W2 B) * A^-1 )
    # (This is a standard identity for penalized least squares smoothers.)
    # Use stable trace via solving A^-1 times (B^T W2 B).
    try:
        A_inv = np.linalg.inv(A)
        edf = float(np.trace((BtW2 @ B) @ A_inv))
        cov = A_inv if yerr is not None else None
    except np.linalg.LinAlgError:
        # compute edf via solves column-by-column
        M = (BtW2 @ B)
        edf_acc = 0.0
        for j in range(p):
            col = M[:, j]
            sol, *_ = np.linalg.lstsq(A, col, rcond=None)
            edf_acc += sol[j]
        edf = float(edf_acc)
        cov = None

    # BIC using edf
    if yerr is None:
        # unknown sigma; constants dropped; compare via n ln(RSS/n) + edf ln n
        bic = n * np.log(max(metric / n, 1e-300)) + edf * np.log(n)
    else:
        # known sigma; -2 log L differs by chi2; add edf ln n
        bic = metric + edf * np.log(n)

    return knots, coef, metric, edf, float(bic), cov


def fit_pspline_bic(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    degree: int = 3,
    n_basis_grid: Optional[Sequence[int]] = None,
    lam_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Grid-search over (n_basis, lam) and pick minimum BIC (using EDF).

    Returns dict with:
      - 'best': PSplineFit
      - 'all_fits': list[PSplineFit]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of same length.")

    # Sort by x
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    if yerr is not None:
        yerr = np.asarray(yerr, float)[idx]

    n = x.size
    if n_basis_grid is None:
        nmin = degree + 1
        nmax = min(max(nmin + 6, n // 3), n - 1)
        n_basis_grid = list(range(nmin, nmax + 1))

    if lam_grid is None:
        # Broad default; adjust as needed
        lam_grid = (10.0 ** np.linspace(-6, 8, 60)).tolist()
        # include lam=0 explicitly
        lam_grid = [0.0] + list(lam_grid)

    fits: List[PSplineFit] = []
    for n_basis in n_basis_grid:
        if n_basis >= n:
            continue
        for lam in lam_grid:
            knots, coef, metric, edf, bic, cov = fit_pspline_fixed(
                x=x, y=y, yerr=yerr,
                degree=degree, n_basis=int(n_basis), lam=float(lam),
            )
            fits.append(PSplineFit(
                degree=degree,
                n_basis=int(n_basis),
                knots=knots,
                coef=coef,
                lam=float(lam),
                metric=float(metric),
                edf=float(edf),
                bic=float(bic),
                cov=cov,
            ))

    if not fits:
        raise ValueError("No valid fits produced; check grids and data length.")

    best = min(fits, key=lambda f: f.bic)
    return {"best": best, "all_fits": fits}


# ------------------------
# Example fit (synthetic line)
# ------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Velocity grid (km/s)
    x = np.linspace(-300, 300, 400)

    # Synthetic broad emission line + mild asymmetry + noise
    true = (
        1.0
        + 3.0 * np.exp(-0.5 * (x / 120.0) ** 2)
        + 0.3 * np.exp(-0.5 * ((x - 80.0) / 70.0) ** 2)
    )
    yerr = 0.06 * np.ones_like(x)
    y = true + rng.normal(0, yerr)

    # Fit
    out = fit_pspline_bic(
        x, y, yerr=yerr,
        degree=3,
        n_basis_grid=range(6, 45),              # candidate basis sizes
        lam_grid=[0.0] + list(10.0 ** np.linspace(-4, 7, 40)),  # candidate smoothness
    )
    best = out["best"]

    print("Best model:")
    print("  degree   :", best.degree)
    print("  n_basis  :", best.n_basis)
    print("  lam      :", best.lam)
    print("  EDF      :", best.edf)
    print("  BIC      :", best.bic)
    print("  metric   :", best.metric, "(chi2)" if yerr is not None else "(RSS)")

    # Predict on a dense grid (here same grid)
    yhat = best.predict(x)

    # Optional: quick scalar diagnostics
    rmse = np.sqrt(np.mean((yhat - true) ** 2))
    print("RMSE vs true:", rmse)