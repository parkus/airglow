import numpy as np
from astropy.io import fits
from airglow import MultiTraceAirglowModel
from matplotlib import pyplot as plt


def test_airglow_models(three_trace_files):
    import emcee
    import corner
    from scipy.special import gamma

    # load the data and put it into lists
    fit_range = [325, 475]
    pixgrids = []
    fluxes = []
    errors = []
    counts = []
    cts_per_flux_factors = []
    for file in three_trace_files:
        h = fits.open(file)
        f, e, cps = [h[1].data[s][0] for s in ['flux', 'error', 'gross']]
        x = np.arange(len(f)) + 1
        keep = (x < fit_range[0]) & (x > fit_range[1])
        exptime = h[1].header['exptime']
        fluxes.append(f[keep])
        errors.append(e[keep])
        pixgrid = np.arange(np.sum(keep), dtype=float)
        pixgrids.append(pixgrid)

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
    min_len = min(len(pixgrid) for pixgrid in pixgrids)
    centers = [
        np.average(pixgrid, weights=flux) if np.any(flux > 0) else np.median(pixgrid)
        for pixgrid, flux in zip(pixgrids, fluxes)
    ]
    center_guess = float(np.mean(centers))
    half_width = max(2.0, 0.05 * min_len)
    midpt_rng = [center_guess - half_width, center_guess + half_width]

    for tol_rel, tol_label in sets:

        # set up a model with tight tolerances
        tolerances = np.array([0.1, 0.2, 4e-14, 0.2, 0.1, 1.0e-16]) * tol_rel
        model = MultiTraceAirglowModel(
            pixgrids,
            tolerances=tolerances,
            midpt_rng=midpt_rng,
        )

        ctstack = np.hstack(counts)
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

        p0_2d = np.array(((center_guess, 2.0, 4e-14, 0.1, 0.3, 1.0e-16),
                          (center_guess, 2.0, 4e-14, 0.1, 0.3, 1.0e-16),
                          (center_guess, 2.0, 4e-14, 0.1, 0.3, 1.0e-16)))
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
        for i, (pixgrid, flux, y) in enumerate(zip(pixgrids, fluxes, ys)):
            plt.figure()
            plt.step(pixgrid, flux, where='mid')
            plt.step(pixgrid, y, where='mid')
            plt.title(f'{tol_label} | trace {i+1}')