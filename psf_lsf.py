import os
import urllib.request

import numpy as np


LSF_URLS = {
    "G140M_1200": (
        "https://www.stsci.edu/files/live/sites/www/files/home/hst/"
        "instrumentation/stis/performance/spectral-resolution/_documents/LSF/"
        "LSF_G140M_1200.txt"
    )
}


def _data_dir():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "data")


def ensure_lsf_cached(name="G140M_1200", data_dir=None, url_override=None):
    if data_dir is None:
        data_dir = _data_dir()
    os.makedirs(data_dir, exist_ok=True)
    filename = f"LSF_{name}.txt"
    target_path = os.path.join(data_dir, filename)
    if os.path.exists(target_path):
        return target_path
    url = url_override or LSF_URLS.get(name)
    if not url:
        raise ValueError(f"Unknown LSF name '{name}'.")
    urllib.request.urlretrieve(url, target_path)
    return target_path


def load_lsf(name="G140M_1200", data_dir=None, url_override=None, aperture="52x0.2"):
    path = ensure_lsf_cached(name=name, data_dir=data_dir, url_override=url_override)
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Unexpected LSF format in {path}.")
    header = lines[1].split()
    if aperture not in header:
        raise ValueError(f"Aperture '{aperture}' not found in LSF header: {header}")
    col_idx = header.index(aperture)

    offsets = []
    values = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) <= col_idx:
            continue
        offsets.append(float(parts[0]))
        values.append(float(parts[col_idx]))
    offsets = np.asarray(offsets)
    values = np.asarray(values)
    values = values / np.sum(values)
    return offsets, values


def resample_lsf(offsets, values, step=1.0):
    grid = np.arange(offsets.min(), offsets.max() + step, step)
    interp_vals = np.interp(grid, offsets, values, left=0.0, right=0.0)
    interp_vals = interp_vals / np.sum(interp_vals)
    return grid, interp_vals


def psf_profile_y(offsets, fwhm=2.0, beta=3.5, profile="moffat"):
    offsets = np.asarray(offsets, dtype=float)
    if profile == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    else:
        alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        kernel = (1.0 + (offsets / alpha) ** 2) ** (-beta)
    kernel = kernel / np.sum(kernel)
    return kernel
