import numpy as np
from scipy import optimize
from astropy.modeling.functional_models import Voigt1D
from astropy.io import fits
from matplotlib import pyplot as plt
import utilities as utils


# airglow function runs about 5x faster if the code instantiates this object now rather than within the function
voigt = Voigt1D()

class AirglowModel(object):
    parameter_order = ['midpts', 'widths', 'fluxes', 'fwhm_Gs', 'fwhm_Ls', 'darkrates']
    n_params_per_trace = len(parameter_order)
    _1d_organization_string = (f"[param1_trace1, param1_trace2, ..., paramn_trace1, paramn_trace2] for an example with two "
                               f"traces.\nThe order of the grouped parmeters is {str(parameter_order)}.")

    def __init__(self, wavegrids, dw_sample, plate_scale, tolerances, midpt_rng):
        """
        Initialize an AirglowModel to simultaneously model the airglow of multiple traces, intended for use with STIS
        modes that observe Lya.

        Parameters
        ----------
        wavegrids : list of wavelength grids to be used in model evaluations during MCMC or other optimization
        dw_sample : grid spacing for supersampling of the airglow profile
        plate_scale : factor enabling the scaling the slit width into a width in AA on the dispersion axis
            This is the dispersion (AA/pixel) divided by the conventional plate scale (arcsec / pixel) yielding
            units of AA/arcsec
        tolerances : list of sigmas for priors tying the parameters of the fits to each trace to each other.
            constrains how tightly the various parameters of the fit to each trace will be pinned to each
            other during fitting. smaller values mean the fits to each trace will not be allowed to differ as much.
        midpt_rng : where the airglow should be centered, used to define a uniform prior that can keep a sampler
            from getting stuck trying to fit the continuum

        Returns
        -------
        AirglowModel object

        Notes
        -----
        At Lya for STIS/G140M the dispersion / plate scale ratio is 1.838 AA/arcsec according to the IHB and for
        E140M it is 0.357 AA/arcsec
        These values roughly agree with a visual inspection of _flt files.
        """
        self.wavegrids = wavegrids
        self.dw_sample = dw_sample
        self.num_traces = len(wavegrids)
        self.plate_scale = plate_scale
        self.tolerances = np.asarray(tolerances)
        self.midpt_rng = midpt_rng

        # helpful for parsing 1d parameter set
        slices = [slice(3*i, 3*(i+1)) for i in range(self.n_params_per_trace)]
        self.param_slices = dict(zip(self.parameter_order, slices))

        # supersample the wavelength grids
        self.wavesup = self._wave_supersample(wavegrids, dw_sample)
        self.supbins = utils.mids2edges(self.wavesup)

        # setting these to None here mainly to enable introspection
        self.midpts = None
        self.widths = None
        self.fluxes = None
        self.fwhm_Gs = None
        self.fwhm_Ls = None
        self.darkrates = None

    def _wave_supersample(self, wavegrids, dw_sample):
        # supersample the wavelength grid
        wmin = min(min(wavegrid) for wavegrid in wavegrids)
        wmax = max(max(wavegrid) for wavegrid in wavegrids)
        return np.arange(wmin, wmax+dw_sample, dw_sample)

    # region convenience functions for parsing and setting parameters
    def tile_params(self, params_single_set):
        """Copy a single set of parameters into a set for all of the traces."""
        ary = np.tile(params_single_set, (self.num_traces,1))
        return ary.T.ravel()

    def params_2d_to_1d(self, params_2d):
        if params_2d.shape[0] == self.num_traces:
            params_2d = params_2d.T
        return np.ravel(params_2d)

    def params_1d_to_2d(self, params):
        """Transform a 1d list of params into a 2d array where each row represents a parameter and each column a set
        of parameters for a given trace. Parameters should be organized as
        {}"""
        params = np.asarray(params)
        return params.reshape((self.n_params_per_trace, self.num_traces))
    params_1d_to_2d.__doc__.format(_1d_organization_string)

    def params_dict(self, params1d):
        params2d = self.params_1d_to_2d(params1d)
        return dict(*zip(self.parameter_order, params2d.T))

    def set_params(self, params1d):
        """
        Parameters
        ----------
        params1d : paremeters of the airglow model. This should be a list of parameters grouped by trace, e.g.,
            {}

        Returns
        -------
        None. Parameters of the object set in place.

        """
        params2d = self.params_1d_to_2d(params1d)
        for i, name in enumerate(self.parameter_order):
            setattr(self, name, params2d[i])
    set_params.__doc__.format(_1d_organization_string)

    @property
    def param_sets(self):
        return [getattr(self, name) for name in self.parameter_order]

    @property
    def params_1d(self):
        return np.hstack(self.param_sets)

    @property
    def params_2d(self):
        return np.array(self.param_sets)
    # endregion

    def evaluate(self, params=None, wavegrids=None, dw_sample=None):
        """
        Generate airglow profiles for each trace on the pre-defined grid with the given parameters. This will update
        the parameters of the AirglowModel object in-place. If None, the object's current parameters are used.

        Parameters
        ----------
        params : parameters of the airglow model. This should be a list of parameters grouped by trace, e.g.,
            {}

        Returns
        -------
        airglow flux in erg s-1 cm-2 AA-1 within the bins

        """
        if params is not None:
            self.set_params(params)
        if wavegrids is None:
            wavegrids = self.wavegrids
            wavesup = self.wavesup
            supbins = self.supbins
        else:
            wavesup = self._wave_supersample(wavegrids, dw_sample)
            supbins = utils.mids2edges(wavesup)

        # boxcar widths in AA based on plate scale
        widths_AA = self.widths * self.plate_scale

        # boxcars for the image of the aperture
        yboxes = utils.boxcars_to_bins(self.midpts, widths_AA, self.fluxes, supbins)

        # voigt profile for the natural and thermal broadening, convolve with boxcar
        x_0 = 0
        amplitude_Ls = 2 / (np.pi * self.fwhm_Ls)
        voigt_span = 3 * (max(self.fwhm_Gs) + max(self.fwhm_Ls))
        n = 2 * int(voigt_span // self.dw_sample) + 1 # ensures the grid is centered on 0
        voigt_grid = np.linspace(-voigt_span, voigt_span, num=n)
        nextra = len(voigt_grid) - len(self.wavesup)
        if nextra > 0: # in case the sampler tries out a voigt profile larger than the range being fit
            nclip = nextra // 2 + 1
            voigt_grid = voigt_grid[nclip:-nclip]
        sets = zip(yboxes, amplitude_Ls, self.fwhm_Ls, self.fwhm_Gs, self.darkrates)
        ys = []
        for ybox, A, L, G, darkrate in sets:
            yvoigt = voigt.evaluate(voigt_grid, x_0=x_0, amplitude_L=A, fwhm_L=L, fwhm_G=G)
            y = np.convolve(ybox, yvoigt, mode='same')
            y += darkrate
            ys.append(y)

        # bin
        ys_binned = [utils.bin_average(w, wavesup, y, left=None, right=None) for w, y in zip(wavegrids, ys)]

        return ys_binned
    params_1d_to_2d.__doc__.format(_1d_organization_string)

    def loglike_tolerance_prior(self, params1d):
        """Return the log prior likelihood of the given set of parameters. The parameters should be organized as
        {}

        Models are penalized based on how much the parameters of the airglow fits to different traces vary, scaled
        by the tolerances.
        """
        params2d = self.params_1d_to_2d(params1d)
        means = np.mean(params2d, axis=1)
        terms = -(params2d - means[:,None])**2/2/self.tolerances[:,None]**2
        return np.sum(terms)
    loglike_tolerance_prior.__doc__.format(_1d_organization_string)

    def loglike_physical_prior(self, params1d):
        if np.any(params1d < 0):
            return -np.inf
        return 0

    def loglike_midpoint_prior(self, params1d):
        midpts = params1d[self.param_slices['midpts']]
        lo = midpts < self.midpt_rng[0]
        hi = midpts > self.midpt_rng[1]
        if np.any(lo | hi):
            return -np.inf
        return 0

    def __call__(self, wavegrids, dw_sample=None):
        """Generate airglow profiles across the supplied wavegrid."""
        return self.evaluate(params=None, wavegrids=wavegrids, dw_sample=dw_sample)


# code snippet to get files for testing on Parke's machine
from pathlib import Path
folder = Path('/Users/parke/Google Drive/Research/STELa/scratch/52x2 airglow fit test data')
base_name = 'o46j010k0_x1d{}.fits'
suffixes = ('bk1', 'bk2', 'trace')
three_trace_files = [folder / base_name.format(suffix) for suffix in suffixes]

def test_airglow_models(three_trace_files):
    import emcee
    import corner
    from scipy.special import gamma

    # load the data and put it into lists
    fit_range = [1213, 1218]
    waves = []
    wavegrids = []
    fluxes = []
    errors = []
    counts = []
    cts_per_flux_factors = []
    for file in three_trace_files:
        h = fits.open(file)
        w, f, e, cps = [h[1].data[s][0] for s in ['wavelength', 'flux', 'error', 'gross']]
        exptime = h[1].header['exptime']
        keep = (w > fit_range[0]) & (w < fit_range[1])
        fluxes.append(f[keep])
        errors.append(e[keep])
        waves.append(w[keep])
        wgrid = utils.mids2edges(w[keep])
        wavegrids.append(wgrid)

        # record factor to estimate counts for a given flux
        cts = cps * exptime
        counts.append(cts[keep])
        cts_per_flux = cts / f
        cts_per_flux_factors.append(cts_per_flux[keep])
        assert np.allclose(cts_per_flux[keep], np.mean(cts_per_flux[keep]), rtol=0.2)
        # if the assertion above fails, it probably means there are some nans or odd pixels that need filling

    # play with purposefully varying the fluxes to see how the fit handles it
    fluxes[0] *= 0.9
    fluxes[2] *= 1.1

    sets = ((10, 'loose tolerances'),
            (0.001, 'tight tolerances'))
    for tol_rel, tol_label in sets:

        # set up a model with tight tolerances
        tolerances = np.array([0.1, 0.2, 4e-14, 0.2, 0.1, 1.0e-16]) * tol_rel
        model = AirglowModel(wavegrids, 0.01, 1.838,
                             tolerances=tolerances, midpt_rng=[1215.3, 1216.3])

        ctstack = np.hstack(counts)
        errorstack = np.hstack(errors)
        cts_per_flux_stack = np.hstack(cts_per_flux_factors)
        def loglike(params):
            physical_prior = model.loglike_physical_prior(params)
            if physical_prior == -np.inf:
                return -np.inf
            ys = model.evaluate(params)
            ystack = np.hstack(ys)

            # estimate poisson likelihoods based on the *expected* number of counts predicted from the model
            expected_counts = ystack * cts_per_flux_stack
            # continuous poisson likelihood
            assert np.all(expected_counts > 0)
            loglike_terms = ctstack * np.log(expected_counts) - expected_counts - np.log(gamma(ctstack + 1))

            loglike_data = np.sum(loglike_terms)
            loglike_midpts = model.loglike_midpoint_prior(params)
            loglike_tolerance = model.loglike_tolerance_prior(params)
            return loglike_data + loglike_tolerance + loglike_midpts + physical_prior
            # return loglike_data + loglike_midpts

        # guess_single = [1215.54, 0.2, 5.5e-13, 0.2, 0.1]
        # guess_all_traces = model.tile_params(guess_single)

        np.random.seed(42)

        p0_2d = np.array(((1215.6, 2.0, 4e-14, 0.1, 0.3, 1.0e-16),
                          (1215.6, 2.0, 4e-14, 0.1, 0.3, 1.0e-16),
                          (1215.6, 2.0, 4e-14, 0.1, 0.3, 1.0e-16)))
        p0 = model.params_2d_to_1d(p0_2d)
        jitter_amplitude = np.tile((0.01, 0.01, 1e-15, 0.01, 0.01, 1e-18), (3,1))
        jitter_amplitude = model.params_2d_to_1d(jitter_amplitude)
        ndim = len(p0)
        nwalkers = ndim * 3
        jitter = np.random.randn(nwalkers, ndim) * jitter_amplitude
        p0 = p0[None,:] + jitter

        sampler = emcee.EnsembleSampler(nwalkers,  ndim, loglike)
        state = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        finalstate = sampler.run_mcmc(state, 1000, progress=True)

        # plot corner for fluxes
        for name, slc in model.param_slices.items():
            fig = plt.figure()
            labels = [f'{name} {i+1}' for i in range(3)]
            # corner.corner(np.log10(sampler.flatchain[:,slc]), labels=labels, fig=fig)
            corner.corner(sampler.flatchain[:,slc], labels=labels, fig=fig)
            fig.suptitle(f'{tol_label} {name} posteriors')

        # plot median fits
        p_median = np.median(sampler.flatchain, axis=0)
        ys = model.evaluate(p_median)
        for i, (wave, flux, y) in enumerate(zip(waves, fluxes, ys)):
            plt.figure()
            plt.step(wave, flux, where='mid')
            plt.step(wave, y, where='mid')
            plt.title(f'{tol_label} | trace {i+1}')