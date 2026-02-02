from math import inf

import numpy as np
from astropy.modeling.functional_models import Voigt1D
from scipy import optimize


_voigt = Voigt1D()


def _center_crop(array, target_len):
    if len(array) == target_len:
        return array
    start = (len(array) - target_len) // 2
    end = start + target_len
    return array[start:end]


def airglow_broadened_boxcar_row(
    x_grid,
    centroid,
    width,
    amplitude,
    fwhm_g,
    fwhm_l,
    lsf_kernel=None,
):
    """
    Build a broadened boxcar airglow model for a single row.

    Parameters
    ----------
    x_grid : array-like
        Pixel coordinates along dispersion for this row.
    centroid : float
        Center of the boxcar in x_grid coordinates.
    width : float
        Boxcar width in x_grid units.
    amplitude : float
        Amplitude scaling for the resulting profile.
    fwhm_g : float
        Gaussian FWHM for the Voigt broadening.
    fwhm_l : float
        Lorentzian FWHM for the Voigt broadening.
    lsf_kernel : array-like, optional
        Additional LSF kernel to convolve with the Voigt kernel.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    boxcar = np.where(np.abs(x_grid - centroid) <= width / 2.0, 1.0, 0.0)

    offsets = x_grid - x_grid[len(x_grid) // 2]
    voigt_kernel = _voigt.evaluate(offsets, x_0=0.0, amplitude_L=1.0, fwhm_L=fwhm_l, fwhm_G=fwhm_g)
    voigt_kernel = voigt_kernel / np.sum(voigt_kernel)

    if lsf_kernel is not None:
        kernel = np.convolve(voigt_kernel, np.asarray(lsf_kernel, dtype=float), mode="full")
    else:
        kernel = voigt_kernel
    kernel = kernel / np.sum(kernel)

    full = np.convolve(boxcar, kernel, mode="full")
    convolved = _center_crop(full, len(x_grid))
    return amplitude * convolved


def estimate_airglow_initial_guess(x_roi, row_roi, keep):
    if np.sum(row_roi[keep]) > 0:
        centroid_guess = np.sum(x_roi[keep] * row_roi[keep]) / np.sum(row_roi[keep])
    else:
        centroid_guess = np.nanmean(x_roi[keep])

    def estimate_fwhm(x, y):
        y = np.array(y, copy=True)
        y -= np.nanmin(y)
        y_sum = np.nansum(y)
        if y_sum == 0:
            return 5.0
        half_max = np.nanmax(y) / 2.0
        above = np.where(y >= half_max)[0]
        if len(above) < 2:
            return 5.0
        return x[above[-1]] - x[above[0]]

    width_guess = estimate_fwhm(x_roi[keep], row_roi[keep])
    amplitude_guess = np.nanmax(row_roi[keep])
    return centroid_guess, width_guess, amplitude_guess


def fit_airglow_row(
    row,
    x_grid,
    roi_x0,
    roi_x1,
    dq_row=None,
    dq_mask_value=0,
    lsf_kernel=None,
    init=None,
    bounds=([350, 2, 0, 0.1, 0.1], [450, 40, 1e6, 5.0, 5.0]),
):
    """
    Fit a broadened boxcar airglow model to a single row within an ROI.

    Parameters
    ----------
    row : array-like
        1D pixel values for the row.
    x_grid : array-like
        1D pixel coordinate grid matching row.
    roi_x0, roi_x1 : int
        Inclusive/exclusive ROI bounds for fitting.
    dq_row : array-like, optional
        DQ values for the row; masked by dq_mask_value if provided.
    dq_mask_value : int, optional
        Bitwise mask value indicating bad pixels.
    lsf_kernel : array-like, optional
        LSF kernel for additional convolution.
    init : tuple, optional
        Initial parameter guess: (centroid, width, amplitude, fwhm_g, fwhm_l).
        If None, uses estimate_airglow_initial_guess.
    bounds : tuple, optional
        Bounds for parameters as (lower, upper). Used as a penalty with Nelder-Mead.

    Returns
    -------
    fit : ndarray
        Best-fit model evaluated on the ROI x-grid.
    params : ndarray
        Best-fit parameters.
    keep : ndarray
        Boolean mask of pixels used in the fit (ROI indexing).
    """
    row = np.asarray(row, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    x_roi = x_grid[roi_x0:roi_x1]
    row_roi = row[roi_x0:roi_x1]

    if dq_row is not None and dq_mask_value != 0:
        dq_roi = np.asarray(dq_row, dtype=int)[roi_x0:roi_x1]
        bad = (dq_roi & dq_mask_value) != 0
    else:
        bad = np.zeros_like(row_roi, dtype=bool)

    keep = (~bad) & np.isfinite(row_roi)
    if not np.any(keep):
        raise ValueError("No valid pixels to fit in the requested ROI.")

    def model_row(params):
        centroid, width, amplitude, fwhm_g, fwhm_l = params
        return airglow_broadened_boxcar_row(
            x_roi, centroid, width, amplitude, fwhm_g, fwhm_l, lsf_kernel=lsf_kernel
        )

    def residuals(params):
        model = model_row(params)
        return model[keep] - row_roi[keep]

    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    if init is None:
        centroid_guess, width_guess, amplitude_guess = estimate_airglow_initial_guess(
            x_roi, row_roi, keep
        )
        init = (centroid_guess, width_guess, amplitude_guess, 1.0, 0.5)
    init = np.asarray(init, dtype=float)

    def objective(params):
        params = np.asarray(params, dtype=float)
        if np.any(params < lower) or np.any(params > upper):
            return inf
        res = residuals(params)
        return np.sum(res ** 2)

    result = optimize.minimize(
        objective,
        x0=init,
        method="Nelder-Mead",
    )
    fit = model_row(result.x)
    return fit, result.x, keep


def fit_airglow_rows(
    rows,
    x_grid,
    roi_x0,
    roi_x1,
    row_indices=None,
    dq_rows=None,
    dq_mask_value=0,
    lsf_kernel=None,
    init=None,
    bounds=([350, 2, 0, 0.1, 0.1], [450, 40, 1e6, 5.0, 5.0]),
):
    """
    Fit the broadened boxcar model for multiple rows, seeding each fit
    with the previous row's best-fit parameters.

    Parameters
    ----------
    rows : array-like
        2D array of rows to fit (shape: N_rows x N_x).
    x_grid : array-like
        1D pixel coordinate grid matching row length.
    roi_x0, roi_x1 : int
        Inclusive/exclusive ROI bounds for fitting.
    row_indices : array-like, optional
        Row indices corresponding to each row (for bookkeeping).
    dq_rows : array-like, optional
        DQ values for rows (shape: N_rows x N_x).
    dq_mask_value : int, optional
        Bitwise mask value indicating bad pixels.
    lsf_kernel : array-like, optional
        LSF kernel for additional convolution.
    init : tuple, optional
        Initial parameter guess for the first row.
    bounds : tuple, optional
        Bounds for parameters as (lower, upper).

    Returns
    -------
    results : dict
        Dictionary with keys: fits, params, keeps, row_indices.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.ndim == 1:
        rows = rows[None, :]
    if dq_rows is not None:
        dq_rows = np.asarray(dq_rows, dtype=int)
    if row_indices is None:
        row_indices = list(range(rows.shape[0]))

    fits = []
    params_list = []
    keeps = []
    current_init = init

    for i, row in enumerate(rows):
        dq_row = dq_rows[i] if dq_rows is not None else None
        fit, params, keep = fit_airglow_row(
            row,
            x_grid,
            roi_x0,
            roi_x1,
            dq_row=dq_row,
            dq_mask_value=dq_mask_value,
            lsf_kernel=lsf_kernel,
            init=current_init,
            bounds=bounds,
        )
        fits.append(fit)
        params_list.append(params)
        keeps.append(keep)
        current_init = params

    return {
        "fits": np.array(fits),
        "params": np.array(params_list),
        "keeps": np.array(keeps),
        "row_indices": np.array(row_indices),
    }

