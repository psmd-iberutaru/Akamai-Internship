import copy

import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import scipy.signal as sp_sig
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import gaussian_fitting as gaussfit
import bessel_fitting as bessfit
import misc_functions as misc


def gaussian_bessel_fit(x_values, y_values,
                        # Arbitrary values.
                        arbitrary={'fft_cutoff': 0.01, 'prominence': 0.25,
                                   'prom_height_ratio': 0.25, 'gauss_toler': 0.1,
                                   'bess_mask_toler': 0.01}):
    """
    This function detects and fits multiple gaussian functions on a bessel 
    function.
    """

    # Type check
    x_values = valid.validate_float_array(x_values)
    y_values = valid.validate_float_array(y_values)

    # Detect potential gaussian locations.
    def _detect_gaussians_and_mask(x_values, y_values,
                                   # Arbitrary
                                   fft_cutoff, prominence, prom_height_ratio,
                                   gauss_toler,
                                   *args, **kwargs):
        """
        This function detects for possible locations of gaussians using 
        arbitrary fft methods. After detected, they are masked.
        """
        # Type check
        x_values = valid.validate_float_array(x_values)
        y_values = valid.validate_float_array(y_values)
        n_datapoints = len(x_values)

        # Sort
        sort_index = np.argsort(x_values)
        x_values = x_values[sort_index]
        y_values = y_values[sort_index]

        # Perform a fft and cut the first and last percents of values.
        y_fft = np.fft.fft(y_values)
        y_fft[int(n_datapoints*fft_cutoff):-int(n_datapoints*fft_cutoff)] = 0

        # Revert the fft transform
        y_ifft = np.fft.ifft(y_fft)

        # Find and estimate x values of gaussian peaks.
        peak_index = sp_sig.find_peaks(y_ifft, prominence=prominence)[0]
        center_guesses = x_values[peak_index]

        # Determine Gaussian bounds using half peak width as a weak
        # approximation for FWHF
        peak_widths = sp_sig.peak_widths(np.abs(y_ifft),
                                         peaks=peak_index,
                                         rel_height=prom_height_ratio)
        peak_lower_bounds = np.array(np.floor(peak_widths[2]), dtype=int)
        peak_upper_bounds = np.array(np.ceil(peak_widths[3]), dtype=int)

        # Mask the entire set within the gaussian bounds. True passes.
        passed = np.ones(n_datapoints, dtype=bool)
        for lowerdex, upperdex in zip(peak_lower_bounds, peak_upper_bounds):
            passed[lowerdex:upperdex] = False

        # Also mask those less than some tolerance.
        passed[np.where(y_ifft < gauss_toler)] = False

        # Return only the valid values via the valid index.
        return x_values[np.where(passed)], y_values[np.where(passed)]

    # Get Bessel only data to attempt to fit a Bessel function to.
    clear_bessel_x, clear_bessel_y = \
        _detect_gaussians_and_mask(x_values=x_values,
                                   y_values=y_values,
                                   **arbitrary)

    fitted_order = bessfit.fit_bessel_function_1st_integer(clear_bessel_x,
                                                           clear_bessel_y)

    def _bessel_mask(x_values, y_values, order, bess_mask_toler,
                     *args, **kwargs):
        """
        This masks values from a bessel function.
        """
        # Type check
        x_values = valid.validate_float_array(x_values)
        y_values = valid.validate_float_array(y_values)

        # Once there is the fitted order, mask all the bessel function values
        # from the points.
        bessel_x = copy.deepcopy(x_values)
        bessel_y = bessfit.bessel_function_1st(bessel_x, fitted_order)

        # Mask the entire set within the gaussian bounds. True passes.
        passed_index = np.where(np.abs(y_values - bessel_y) > bess_mask_toler)

        # Return only the valid values via the valid index.
        return x_values[passed_index], y_values[passed_index]

    # Get gaussian only data.
    clear_gauss_x, clear_gauss_y = _bessel_mask(x_values=x_values,
                                                y_values=y_values,
                                                order=fitted_order,
                                                **arbitrary)

    # Attempt to fit gaussians
    center_array, std_dev_array, height_array, covariance = \
        gaussfit.fit_multigaussian(clear_gauss_x, clear_gauss_y)

    # It should be the case that it is all. Return order first.
    return fitted_order, center_array, std_dev_array, height_array, covariance
