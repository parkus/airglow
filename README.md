# Airglow

Airglow modeling and STIS trace extraction utilities for HST spectroscopic data. This package supports:

- Multi-trace airglow modeling ( Voigt profiles for Lyα and related lines)
- Trace stack extraction from STIS FLT files
- Empirical background error estimation from trace stacks
- Revised error columns for x1d files

## Installation

### Option 1: Install from source (recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/airglow.git
cd airglow

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with dependencies
pip install -e .
```

### Option 2: Install dependencies only

If you prefer to run from the repository without installing the package:

```bash
pip install -r requirements.txt
```

Then add the project root to your Python path when running notebooks (see Usage below).

## Requirements

- Python 3.10 or newer
- **stistools**: HST/STIS calibration tools. For full STIS pipeline support, STScI recommends using their `stenv` conda environment:
  ```bash
  conda create -n stenv -c conda-forge -c stsci stenv
  conda activate stenv
  pip install -e .
  ```

## Usage

### Running the notebooks

After installing, start Jupyter from the repository root:

```bash
cd airglow
jupyter notebook
```

Open any notebook in the `notebooks/` directory. The notebooks use:

```python
from airglow import extraction_utils as extract_utils
from airglow import MultiTraceAirglowModel
```

### Test data

The notebooks expect a `test-data/` directory with STIS FITS files (e.g., `of9b05010_flt.fits`, `of9b05010_x1d_traces.fits`). The package looks for it:

1. Relative to the current working directory: `./test-data`, `../test-data`, `../../test-data`
2. Or set the `AIRGLOW_TEST_DATA` environment variable to the full path

```bash
export AIRGLOW_TEST_DATA=/path/to/your/test-data
```

### Example: Estimate background errors

```python
from pathlib import Path
from astropy.io import fits
from airglow import extraction_utils as extract_utils

test_data_dir = extract_utils.find_test_data_dir()
trace_file = test_data_dir / "of9b05010_x1d_traces.fits"

with fits.open(trace_file) as hdul:
    trace_table = hdul["TRACES"].data
    traces = extract_utils._get_column(trace_table, "flux")
    wavelength = extract_utils._get_column(trace_table, "wavelength")
    y_positions = extract_utils._get_column(trace_table, "a2center")

result = extract_utils.estimate_background_sigma_from_traces(
    wavelength_stack=wavelength,
    trace_stack=traces,
    y_positions=y_positions,
    y_windows=[(40, 160), (240, 750), (870, 960)],
)
```

### Example: Airglow model

```python
from airglow import MultiTraceAirglowModel

model = MultiTraceAirglowModel(
    pixgrids=[pixgrid1, pixgrid2],
    tolerances=[0.1, 0.2, 4e-14, 0.2, 0.1, 1.0e-16],
    midpt_rng=[center - 5, center + 5],
)
```

## Project layout

```
airglow/
├── pyproject.toml      # Package metadata and dependencies
├── requirements.txt    # Pip-installable dependencies
├── README.md
├── src/
│   └── airglow/        # Main package
│       ├── __init__.py
│       ├── model.py           # MultiTraceAirglowModel
│       ├── extraction_utils.py
│       └── utilities.py
├── notebooks/         # Jupyter notebooks
├── test-data/         # STIS FITS files (include or set AIRGLOW_TEST_DATA)
└── tests.py           # Unit tests
```

## License

BSD-3-Clause
