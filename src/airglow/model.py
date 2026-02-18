import numpy as np
from astropy.modeling.functional_models import Voigt1D
from matplotlib import pyplot as plt
from . import utilities as utils


# airglow function runs about 5x faster if the code instantiates this object now rather than within the function
voigt = Voigt1D()

class MultiTraceAirglowModel(object):
    parameter_order = ['midpts', 'widths', 'fluxes', 'fwhm_Gs', 'fwhm_Ls', 'darkrates']
    n_params_per_trace = len(parameter_order)
    _1d_organization_string = (f"[param1_trace1, param1_trace2, ..., paramn_trace1, paramn_trace2] for an example with two "
                               f"traces.\nThe order of the grouped parmeters is {str(parameter_order)}.")

    def __init__(self, pixgrids, tolerances, midpt_rng):
        """
        Initialize an AirglowModel to simultaneously model the airglow of multiple traces, intended for use with STIS
        modes that observe Lya.

        Parameters
        ----------
        pixgrids : list of pixel grids to be used in model evaluations during MCMC or other optimization.
            Each grid is interpreted as pixel centers.
        tolerances : list of sigmas for priors tying the parameters of the fits to each trace to each other.
            constrains how tightly the various parameters of the fit to each trace will be pinned to each
            other during fitting. smaller values mean the fits to each trace will not be allowed to differ as much.
        midpt_rng : where the airglow should be centered, used to define a uniform prior that can keep a sampler
            from getting stuck trying to fit the continuum

        Returns
        -------
        AirglowModel object

        """
        self.pixgrids = pixgrids
        self.num_traces = len(pixgrids)
        self.tolerances = np.asarray(tolerances)
        self.midpt_rng = midpt_rng

        # helpful for parsing 1d parameter set
        slices = [slice(3*i, 3*(i+1)) for i in range(self.n_params_per_trace)]
        self.param_slices = dict(zip(self.parameter_order, slices))

        # precompute bin edges for each trace
        self.pixbins = [utils.mids2edges(pixgrid) for pixgrid in pixgrids]

        # setting these to None here mainly to enable introspection
        self.midpts = None
        self.widths = None
        self.fluxes = None
        self.fwhm_Gs = None
        self.fwhm_Ls = None
        self.darkrates = None

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

    def evaluate(self, params=None, pixgrids=None):
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
        if pixgrids is None:
            pixgrids = self.pixgrids
            pixbins_list = self.pixbins
        else:
            pixbins_list = [utils.mids2edges(pixgrid) for pixgrid in pixgrids]

        # voigt profile for the natural and thermal broadening, convolve with boxcar
        ys = []
        for i, (pixgrid, pixbins) in enumerate(zip(pixgrids, pixbins_list)):
            ybox = utils.boxcars_to_bins([self.midpts[i]], [self.widths[i]], [self.fluxes[i]], pixbins)[0]
            voigt_span = 3 * (self.fwhm_Gs[i] + self.fwhm_Ls[i])
            n = 2 * int(voigt_span) + 1  # ensures the grid is centered on 0
            voigt_grid = np.linspace(-voigt_span, voigt_span, num=n)
            nextra = len(voigt_grid) - len(pixgrid)
            if nextra > 0:  # in case the sampler tries out a voigt profile larger than the range being fit
                nclip = nextra // 2 + 1
                voigt_grid = voigt_grid[nclip:-nclip]
            yvoigt = voigt.evaluate(
                voigt_grid,
                x_0=0,
                amplitude_L=1.0,
                fwhm_L=self.fwhm_Ls[i],
                fwhm_G=self.fwhm_Gs[i],
            )
            yvoigt = yvoigt / np.sum(yvoigt)
            y = np.convolve(ybox, yvoigt, mode='same')
            y += self.darkrates[i]
            ys.append(y)

        return ys
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

    def __call__(self, pixgrids=None):
        """Generate airglow profiles across the supplied pixel grids."""
        return self.evaluate(params=None, pixgrids=pixgrids)
