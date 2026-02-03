import numpy as np
from astropy.modeling.functional_models import Voigt1D
import george
from george import kernels
from scipy import optimize
from scipy.interpolate import Rbf
from scipy.special import gammaln


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
    bounds=([0, 0, 0, 1e-5, 1e-5], [1024, 500, 1e10, 100, 100]),
    fit_fwhm=True,
    fixed_fwhm_g=1.0,
    fixed_fwhm_l=0.5,
    minimize_options=None,
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
        The fit uses a Poisson log-likelihood with model values as the mean counts.
        Data are rounded to nearest integer and clipped at zero for Poisson evaluation.
    fit_fwhm : bool, optional
        If True, fit fwhm_g and fwhm_l. If False, use fixed values.
    fixed_fwhm_g : float, optional
        Fixed Gaussian FWHM when fit_fwhm is False.
    fixed_fwhm_l : float, optional
        Fixed Lorentzian FWHM when fit_fwhm is False.
    minimize_options : dict, optional
        Options passed to scipy.optimize.minimize.

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
        if fit_fwhm:
            centroid, width, amplitude, fwhm_g, fwhm_l = params
            return airglow_broadened_boxcar_row(
                x_roi, centroid, width, amplitude, fwhm_g, fwhm_l, lsf_kernel=lsf_kernel
            )
        centroid, width, amplitude = params
        return airglow_broadened_boxcar_row(
            x_roi, centroid, width, amplitude, fixed_fwhm_g, fixed_fwhm_l, lsf_kernel=lsf_kernel
        )

    def neg_loglike(params):
        model = model_row(params)[keep]
        data = row_roi[keep]
        # handle non-integer and negative counts for Poisson stats
        counts = np.rint(data)
        counts = np.clip(counts, 0, None)
        if np.any(model <= 0):
            return 1e300
        loglike = counts * np.log(model) - model - gammaln(counts + 1)
        return -np.sum(loglike)

    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    if init is None:
        centroid_guess, width_guess, amplitude_guess = estimate_airglow_initial_guess(
            x_roi, row_roi, keep
        )
        if fit_fwhm:
            init = (centroid_guess, width_guess, amplitude_guess, fixed_fwhm_g, fixed_fwhm_l)
        else:
            init = (centroid_guess, width_guess, amplitude_guess)
    init = np.asarray(init, dtype=float)
    if not fit_fwhm:
        lower = lower[:3]
        upper = upper[:3]
        if init.shape[0] != 3:
            init = init[:3]

    def objective(params):
        params = np.asarray(params, dtype=float)
        if np.any(params < lower) or np.any(params > upper):
            return 1e300
        return neg_loglike(params)

    result = optimize.minimize(
        objective,
        x0=init,
        method="Nelder-Mead",
        options=minimize_options,
    )
    fit = model_row(result.x)
    return fit, result.x, keep, result


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
    fit_fwhm=True,
    fixed_fwhm_g=1.0,
    fixed_fwhm_l=0.5,
    minimize_options=None,
):
    """
    Fit the broadened boxcar model for multiple rows.

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
    init : tuple or array-like, optional
        Initial parameter guess for each row. If a 1D tuple/array is provided,
        it will be used for every row. If a 2D array is provided, it must
        have shape (N_rows, N_params).
    bounds : tuple, optional
        Bounds for parameters as (lower, upper).
    fit_fwhm : bool, optional
        If True, fit fwhm_g and fwhm_l. If False, use fixed values.
    fixed_fwhm_g : float, optional
        Fixed Gaussian FWHM when fit_fwhm is False.
    fixed_fwhm_l : float, optional
        Fixed Lorentzian FWHM when fit_fwhm is False.
    minimize_options : dict, optional
        Options passed to scipy.optimize.minimize.

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
    results = []

    if init is not None:
        init = np.asarray(init, dtype=float)
        if init.ndim == 1:
            init_list = [init for _ in range(rows.shape[0])]
        elif init.ndim == 2 and init.shape[0] == rows.shape[0]:
            init_list = [init[i] for i in range(rows.shape[0])]
        else:
            raise ValueError("init must be 1D or (N_rows, N_params).")
    else:
        init_list = [None for _ in range(rows.shape[0])]

    for i, row in enumerate(rows):
        dq_row = dq_rows[i] if dq_rows is not None else None
        fit, params, keep, result = fit_airglow_row(
            row,
            x_grid,
            roi_x0,
            roi_x1,
            dq_row=dq_row,
            dq_mask_value=dq_mask_value,
            lsf_kernel=lsf_kernel,
            init=init_list[i],
            bounds=bounds,
            fit_fwhm=fit_fwhm,
            fixed_fwhm_g=fixed_fwhm_g,
            fixed_fwhm_l=fixed_fwhm_l,
            minimize_options=minimize_options,
        )
        fits.append(fit)
        params_list.append(params)
        keeps.append(keep)
        results.append(result)
    return {
        "fits": np.array(fits),
        "params": np.array(params_list),
        "keeps": np.array(keeps),
        "row_indices": np.array(row_indices),
    }


def fit_poly_surface_2d(data, roi, order=2, mask=None, weights=None):
    """
    Fit a 2D polynomial surface to an ROI.

    Parameters
    ----------
    data : array-like
        2D image data.
    roi : tuple
        ROI bounds as (y0, y1, x0, x1) or ((y0, y1), (x0, x1)).
    order : int, optional
        Polynomial order (total degree). Default is 2.
    mask : array-like, optional
        Boolean mask for data; True values are excluded from the fit.
        Can be full-image shape or ROI shape.
    weights : array-like, optional
        Weights per pixel. Can be full-image shape or ROI shape.

    Returns
    -------
    result : dict
        Dictionary with keys: coeffs, powers, model, center, roi, keep.
    """
    data = np.asarray(data, dtype=float)
    if len(roi) == 2 and len(roi[0]) == 2 and len(roi[1]) == 2:
        (y0, y1), (x0, x1) = roi
    else:
        y0, y1, x0, x1 = roi

    roi_data = data[y0:y1, x0:x1]
    y_grid, x_grid = np.indices(roi_data.shape, dtype=float)
    y_center = np.nanmean(y_grid)
    x_center = np.nanmean(x_grid)
    y = y_grid - y_center
    x = x_grid - x_center

    keep = np.isfinite(roi_data)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == data.shape:
            mask_arr = mask_arr[y0:y1, x0:x1]
        elif mask_arr.shape != roi_data.shape:
            raise ValueError("Mask must match ROI shape or full image shape.")
        keep &= ~mask_arr

    roi_weights = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape == data.shape:
            roi_weights = weights_arr[y0:y1, x0:x1]
        elif weights_arr.shape == roi_data.shape:
            roi_weights = weights_arr
        else:
            raise ValueError("Weights must match ROI shape or full image shape.")
        keep &= np.isfinite(roi_weights) & (roi_weights > 0)

    if not np.any(keep):
        raise ValueError("No valid pixels to fit in the requested ROI.")

    powers = []
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            powers.append((i, j))
            terms.append((x ** i) * (y ** j))

    A = np.stack([term[keep] for term in terms], axis=1)
    b = roi_data[keep]

    if roi_weights is not None:
        w = roi_weights[keep]
        A = A * w[:, None]
        b = b * w

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    model = np.zeros_like(roi_data)
    for coeff, (i, j) in zip(coeffs, powers):
        model += coeff * (x ** i) * (y ** j)

    return {
        "coeffs": coeffs,
        "powers": np.array(powers, dtype=int),
        "model": model,
        "center": (y_center, x_center),
        "roi": (y0, y1, x0, x1),
        "keep": keep,
    }


def fit_rbf_surface_2d(
    data,
    roi,
    mask=None,
    function="thin_plate",
    smooth=0.0,
    epsilon=None,
    subsample=None,
):
    """
    Fit a 2D radial basis function (RBF) surface to an ROI.

    Parameters
    ----------
    data : array-like
        2D image data.
    roi : tuple
        ROI bounds as (y0, y1, x0, x1) or ((y0, y1), (x0, x1)).
    mask : array-like, optional
        Boolean mask for data; True values are excluded from the fit.
        Can be full-image shape or ROI shape.
    function : str, optional
        RBF kernel type (e.g., "thin_plate", "multiquadric", "linear").
    smooth : float, optional
        Smoothing parameter passed to scipy.interpolate.Rbf.
    epsilon : float, optional
        Shape parameter for some RBF kernels.
    subsample : tuple or int, optional
        If provided, fit the RBF on a subsampled grid within the ROI.
        Use (ny, nx) to specify target grid size or an int for both axes.

    Returns
    -------
    result : dict
        Dictionary with keys: rbf, model, roi, keep.
    """
    data = np.asarray(data, dtype=float)
    if len(roi) == 2 and len(roi[0]) == 2 and len(roi[1]) == 2:
        (y0, y1), (x0, x1) = roi
    else:
        y0, y1, x0, x1 = roi

    roi_data = data[y0:y1, x0:x1]
    y_grid, x_grid = np.indices(roi_data.shape, dtype=float)

    keep = np.isfinite(roi_data)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == data.shape:
            mask_arr = mask_arr[y0:y1, x0:x1]
        elif mask_arr.shape != roi_data.shape:
            raise ValueError("Mask must match ROI shape or full image shape.")
        keep &= ~mask_arr

    if not np.any(keep):
        raise ValueError("No valid pixels to fit in the requested ROI.")

    if subsample is not None:
        if isinstance(subsample, int):
            subsample = (subsample, subsample)
        if len(subsample) != 2:
            raise ValueError("subsample must be an int or a (ny, nx) tuple.")
        ny, nx = roi_data.shape
        ny_sub, nx_sub = subsample
        ny_sub = max(2, min(ny_sub, ny))
        nx_sub = max(2, min(nx_sub, nx))
        y_idx = np.linspace(0, ny - 1, ny_sub).round().astype(int)
        x_idx = np.linspace(0, nx - 1, nx_sub).round().astype(int)
        y_sub, x_sub = np.meshgrid(y_idx, x_idx, indexing="ij")
        sub_mask = keep[y_sub, x_sub]
        x = x_sub[sub_mask].ravel()
        y = y_sub[sub_mask].ravel()
        z = roi_data[y_sub, x_sub][sub_mask].ravel()
    else:
        x = x_grid[keep]
        y = y_grid[keep]
        z = roi_data[keep]

    if epsilon is None:
        rbf = Rbf(x, y, z, function=function, smooth=smooth)
    else:
        rbf = Rbf(x, y, z, function=function, smooth=smooth, epsilon=epsilon)

    model = rbf(x_grid, y_grid)

    return {
        "rbf": rbf,
        "model": model,
        "roi": (y0, y1, x0, x1),
        "keep": keep,
    }


def fit_gp_1d(x, y, yerr=None, smoothing=1.0, white_noise=1e-6, fit_white_noise=False):
    """
    Fit a 1D Gaussian Process to data with a tunable smoothing length scale.

    Parameters
    ----------
    x : array-like
        1D coordinate array.
    y : array-like
        1D data values.
    yerr : array-like, optional
        1D uncertainties; if None, uses white_noise.
    smoothing : float, optional
        Length scale for the kernel. Higher values => smoother GP.
    white_noise : float, optional
        White noise level when yerr is not provided.
    fit_white_noise : bool, optional
        If True, fit a white-noise term along with the GP kernel.

    Returns
    -------
    gp : george.GP
        Configured GP object.
    mean : ndarray
        GP mean prediction at x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if yerr is None:
        yerr = np.full_like(y, white_noise, dtype=float)
    else:
        yerr = np.asarray(yerr, dtype=float)

    amplitude = np.var(y) if np.var(y) > 0 else 1.0
    kernel = amplitude * kernels.ExpSquaredKernel(smoothing ** 2)

    if fit_white_noise:
        gp = george.GP(
            kernel,
            mean=np.mean(y),
            fit_mean=True,
            white_noise=np.log(white_noise ** 2),
            fit_white_noise=True,
        )
        def nll(p):
            try:
                gp.set_parameter_vector(p)
                gp.compute(x)
                nll_val = -gp.log_likelihood(y)
            except Exception as exc:
                return 1e300
            return nll_val

        p0 = gp.get_parameter_vector()
        # Set conservative bounds to avoid numerical blow-ups
        y_std = np.nanstd(y) if np.nanstd(y) > 0 else 1.0
        mean_bounds = (np.nanmin(y) - 5.0 * y_std, np.nanmax(y) + 5.0 * y_std)
        white_noise_floor = max(white_noise, 1e-8)
        white_noise_ceil = max(y_std * 10.0, white_noise_floor * 10.0)
        log_wn_bounds = (np.log(white_noise_floor ** 2), np.log(white_noise_ceil ** 2))
        amp = np.var(y) if np.var(y) > 0 else 1.0
        log_amp_bounds = (np.log(amp * 1e-6), np.log(amp * 1e6))
        ls_floor = max(smoothing / 10.0, 1e-3)
        ls_ceil = max(smoothing * 10.0, 1e-1)
        log_ls_bounds = (np.log(ls_floor ** 2), np.log(ls_ceil ** 2))
        bounds = [mean_bounds, log_wn_bounds, log_amp_bounds, log_ls_bounds]
        result = optimize.minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        gp.set_parameter_vector(result.x)
        mean, _ = gp.predict(y, x, return_var=True)
        return gp, mean, result

    gp = george.GP(kernel)
    gp.compute(x, yerr)
    mean, _ = gp.predict(y, x, return_var=True)
    return gp, mean


def emd_decompose_1d(y, max_imf=None):
    """
    Perform empirical mode decomposition (EMD) on a 1D series.

    Parameters
    ----------
    y : array-like
        1D input series.
    max_imf : int, optional
        Maximum number of IMFs to extract.

    Returns
    -------
    imfs : ndarray
        Array of intrinsic mode functions (IMFs).
    residue : ndarray
        Residual after extracting IMFs.
    """
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise ImportError(
            "PyEMD is required for emd_decompose_1d. Install with `pip install EMD-signal`."
        ) from exc

    y = np.asarray(y, dtype=float)
    emd = EMD()
    if max_imf is not None:
        imfs = emd.emd(y, max_imf=max_imf)
    else:
        imfs = emd.emd(y)
    residue = y - np.sum(imfs, axis=0) if imfs.size else y
    return imfs, residue

