import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import scipy.signal as sp_sig
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import misc_functions as misc


def gaussian_function(x_input, center, std_dev, height):
    """
    Equation for a single gaussian.

    Input:
    x_input = x-values to be input into the gaussian function.
    """

    # Type check.
    x_input = valid.validate_float_array(x_input)
    center = valid.validate_float_value(center)
    # Standard deviations can't be negative.
    std_dev = valid.validate_float_value(std_dev, greater_than=0)
    height = valid.validate_float_value(height)

    # Use the equation of a gaussian from Wikipedia:
    y_output = (((1 / std_dev * np.sqrt(2 * np.pi))
                 * np.exp(-0.5 * ((x_input - center)/std_dev)**2))
                + height)
    return np.array(y_output, dtype=float)


def multigaussian_function(x_input, center_array, std_dev_array, height_array,
                           gaussian_count=1):
    """
    Equation for a multigaussian, where the arrays are parallel, denoting the
    properties of each gaussian. Assuming all of the gaussians are linearly combined.
    """

    # Type check
    gaussian_count = valid.validate_int_value(gaussian_count, greater_than=0)
    x_input = valid.validate_float_array(x_input)
    center_array = valid.validate_float_array(
        center_array, size=gaussian_count)
    std_dev_array = valid.validate_float_array(std_dev_array, 
                                               size=gaussian_count, 
                                               deep_validate=True, 
                                               greater_than=0)
    height_array = valid.validate_float_array(
        height_array, size=gaussian_count)

    # Define initial variables.
    n_datapoints = len(x_input)
    y_output_array = np.zeros(n_datapoints)

    # Loop and sum over all gaussian values.
    for gaussiandex in range(gaussian_count):
        y_output_array += gaussian_function(x_input,
                                            center_array[gaussiandex],
                                            std_dev_array[gaussiandex],
                                            height_array[gaussiandex])

    return np.array(y_output_array, dtype=float)


def generate_gaussian(center, std_dev, height,
                      x_domain=[-3, 3], n_datapoints=10000):
    """
    Generate a gaussian curve with datapoints.

    Input:
        center = central x value
        std_dev = standard deviation of the function
        height = height (y-off set) of the function
        x_range = absolute domain of the gaussian function 
        n_datapoints = total number of input datapoints of gaussian function
    Output: x_values,y_values
        x_values = the x-axial array of the gaussian function within the domain
        y_values = the y-axial array of the gaussian function within the domain
    """

    # Make the x-axis value array given the domain and the number of points.
    x_values = np.random.uniform(x_domain[0], x_domain[-1], n_datapoints)

    # Generate the gaussian function and map to an output with the input
    # parameters.
    y_values = gaussian_function(x_values, center, std_dev, height)

    return x_values, y_values


def generate_multigaussian(center_list, std_dev_list, height_list,
                           x_domain_list, gaussian_count=None,
                           n_datapoints_list=None,
                           n_datapoints=None):
    """
    Generates a multigaussian of datapoints.
    """

    # Assume the center list is the highest priority (but check the
    # std_dev) for double checking the gaussian count.
    if (gaussian_count is None):
        gaussian_count = len(center_list)
        # Double check with std_dev
        if ((gaussian_count != len(std_dev_list)) or
                (len(std_dev_list) != len(center_list))):
            raise InputError('The number of gaussians to generate is not '
                             'known, nor can it be accurately derived from '
                             'the inputs given.    --Kyubey')

    # Type check
    gaussian_count = valid.validate_int_value(gaussian_count, greater_than=0)
    center_list = valid.validate_float_array(center_list, size=gaussian_count)
    std_dev_list = valid.validate_float_array(std_dev_list,
                                              size=gaussian_count,
                                              deep_validate=True,
                                              greater_than=0)
    height_list = valid.validate_float_array(height_list, size=gaussian_count)

    # Initial variables.
    x_points = []
    y_points = []

    # Check if the user provides individual datapoint count values.
    if (n_datapoints_list is not None):
        n_datapoints_list = valid.validate_int_array(n_datapoints_list, 
                                                     size=gaussian_count, 
                                                     deep_validate=True, 
                                                     greater_than=0)
    elif (n_datapoints is not None):
        # Validate
        n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)
        # Define subpoints
        n_subpoints = np.ceil(n_datapoints/gaussian_count)
        n_datapoints_list = np.full(gaussian_count, n_subpoints)

    for gaussiandex in range(gaussian_count):
        # Generate one gaussian.
        temp_x_points, temp_y_points = \
            generate_gaussian(center_list[gaussiandex],
                              std_dev_list[gaussiandex],
                              height_list[gaussiandex],
                              x_domain_list[gaussiandex],
                              n_datapoints=n_datapoints_list[gaussiandex])
        # Propagate one dimension out.
        temp_y_points = np.array([temp_y_points])

        for sub_gaussiandex in range(gaussian_count):
            # Prevent double dipping.
            if (sub_gaussiandex == gaussiandex):
                continue

            # Extract the maximum value from all of the gaussians.
            added_y_points = gaussian_function(temp_x_points,
                                               center_list[sub_gaussiandex],
                                               std_dev_list[sub_gaussiandex],
                                               height_list[sub_gaussiandex])
            # Propagate one dimension out.
            added_y_points = np.array([added_y_points])
            temp_y_points = np.concatenate((temp_y_points, added_y_points),
                                           axis=0)
        # Extract only the maximum values.
        temp_y_points = np.amax(temp_y_points, axis=0)

        # Append the values for storage.
        x_points.append(temp_x_points)
        y_points.append(temp_y_points)

    # Morph to numpy arrays.
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)
    # Flatten arrays just in case.
    x_points = np.ravel(x_points)
    y_points = np.ravel(y_points)

    # Double check that they are the same size.
    if (x_points.size != y_points.size):
        raise ShapeError('x_points and y_points do not seem to be the same '
                         'size.    --Kyubey')
    elif (x_points.shape != y_points.shape):
        raise ShapeError('x_points and y_points do not seem to be the same '
                         'shape.    --Kyubey')

    # Ensure that the total number of points is not exceeded.
    if (n_datapoints is not None):
        if (x_points.size != n_datapoints):
            delta_size = x_points.size - n_datapoints
            # It should not be the case that delta_size is negative.
            if (delta_size < 0):
                raise OutputError(
                    'The number of points is less than expected.')

            removed_indexes = np.random.randint(0, x_points.size - 1,
                                                delta_size)
            x_points = np.delete(x_points, removed_indexes)
            y_points = np.delete(y_points, removed_indexes)

    # Sort the values before returning.
    sort_index = np.argsort(x_points)
    x_points = x_points[sort_index]
    y_points = y_points[sort_index]

    return np.array(x_points, dtype=float), np.array(y_points, dtype=float)


def generate_noisy_gaussian(center, std_dev, height, x_domain, noise_domain, 
                            n_datapoints):
    """
    Generate a gaussian with some aspect of noise.

    Input:
        center = central x value
        std_dev = standard deviation of the function
        height = height (y-off set) of the function
        noise_range = uniform random distribution of noise from perfect gauss function
        x_range = absolute domain of the gaussian function 
        n_datapoints = total number of input datapoints of gaussian function
    Output: x_values,y_values
        x_values = the x-axial array of the gaussian function within the domain
        y_values = the y-axial array of the gaussian function within the domain
    """
    # Type check.
    center = valid.validate_float_value(center)
    std_dev = valid.validate_float_value(std_dev,greater_than=0)
    height = valid.validate_float_value(height)
    x_domain = valid.validate_float_array(x_domain,shape=(2,),size=2)
    noise_domain = valid.validate_float_array(noise_domain,shape=(2,),size=2)
    n_datapoints = valid.validate_int_value(n_datapoints,greater_than=0)

    # Make the x-axis value array given the domain and the number of points.
    x_values = np.random.uniform(x_domain[0], x_domain[-1], n_datapoints)

    # Generate the gaussian function and map to an output with the input
    # parameters.
    y_values = gaussian_function(x_values, center, std_dev, height)

    # Imbue the gaussian with random noise.
    y_values = misc.generate_noise(y_values, noise_domain,
                                   distribution='uniform')

    return x_values, y_values


def generate_noisy_multigaussian(center_list, std_dev_list, height_list,
                                 noise_domain_list, x_domain_list,
                                 n_datapoints=None, n_datapoints_list=None,
                                 gaussian_count=None, cumulative_noise=False):
    """
    Generate multiple gaussians with some aspect of noise within one 
    dataset.

    Input:
    center_list = list of central x values
    std_dev_list = list of standard deviations of the functions
    height_list = list of heights (y-off set) of the functions
    noise_domain_list = list of uniform random distribution of noise 
        from perfect gauss function
    x_domain_list = absolute domains of the gaussian functions
    n_datapoints = total number of datapoints
    n_datapoints_list = list of number of datapoints (overrides 
        n_datapoints)
    gaussian_count = the number of gaussian functions to be made
    cumulative_noise = if each gaussian has noise (True), or just the 
         entire set (False).

    Output: x_values,y_values
    x_values = the x-axial array of the gaussian function within the 
        domain
    y_values = the y-axial array of the gaussian function within the 
        domain
    """

    # Assume the center list is the highest priority (but check the
    # std_dev) for double checking the gaussian count.
    if (gaussian_count is None):
        gaussian_count = len(center_list)
        # Double check with std_dev
        if ((gaussian_count != len(std_dev_list)) or
                (len(std_dev_list) != len(center_list))):
            raise InputError('The number of gaussians to generate is not '
                             'known, nor can it be accurately derived from '
                             'the inputs given.    --Kyubey')

    # Type check.
    center_list = valid.validate_float_array(center_list, size=gaussian_count)
    std_dev_list = valid.validate_float_array(
        std_dev_list, size=gaussian_count)
    height_list = valid.validate_float_array(height_list, size=gaussian_count)
    noise_domain_list = valid.validate_float_array(noise_domain_list,
                                                   shape=(gaussian_count, 2))
    x_domain_list = valid.validate_float_array(x_domain_list,
                                               shape=(gaussian_count, 2))
    cumulative_noise = valid.validate_boolean_value(cumulative_noise)

    # Type check optional elements
    if (n_datapoints is not None):
        # Type check.
        n_datapoints = valid.validate_float_value(n_datapoints, greater_than=0)

        # Assume equal distribution of datapoints over all gaussian functions.
        n_subpoints = np.ceil(n_datapoints/gaussian_count)
        n_datapoints_list = np.full(gaussian_count, n_subpoints)
        # Track the total number of datapoints.
        total_datapoints = n_datapoints
    elif (n_datapoints_list is not None):
        # Type check the datapoints list.
        n_datapoints_list = valid.validate_float_array(n_datapoints_list,
                                                       size=gaussian_count,
                                                       deep_validate=True,
                                                       greater_than=0)
        # Track the total number of datapoints.
        total_datapoints = np.sum(n_datapoints_list)
    else:
        raise InputError('n_datapoints and n_datapoints_list are empty. One '
                         'must be provided.    --Kyubey')

    # Initialize initial variables.
    x_values = []
    y_values = []

    # Check how to distribute noise.
    if (cumulative_noise):
        # Each gaussian must be generated on its own.
        for gaussiandex in range(gaussian_count):
            # Generate gaussian.
            temp_x_points, temp_y_points = \
                generate_gaussian(center_list[gaussiandex],
                                  std_dev_list[gaussiandex],
                                  height_list[gaussiandex],
                                  x_domain_list[gaussiandex],
                                  n_datapoints_list[gaussiandex])
            temp_y_points = misc.generate_noise(temp_y_points,
                                                noise_domain_list[gaussiandex],
                                                distribution='uniform')
            # Store for return
            x_values = np.append([x_values], [temp_x_points], axis=0)
            y_values = np.append([y_values], [temp_y_points], axis=0)

        # Maximize the values, discarding everything lower than.
        x_values = np.amax(x_values, axis=0)
        y_values = np.amax(y_values, axis=0)

    else:
        # Generate noise of every point after gaussian generation.
        # Generate gaussian
        x_values, y_values = generate_multigaussian(center_list, std_dev_list,
                                                    height_list, x_domain_list,
                                                    gaussian_count,
                                                    n_datapoints_list)

        # Generate noise. Warn the user that only the first noise domain is
        # being used.
        kyubey_warning(OutputWarning, ('Only the first element of the '
                                       'noise_domian_list is used if '
                                       'cumulative_noise is False.'
                                       '    --Kyubey'))
        y_values = misc.generate_noise(y_values, noise_domain_list[0],
                                       distribution='uniform')

    return np.array(x_values, dtype=float), np.array(y_values, dtype=float)


def fit_gaussian(x_values, y_values,
                 center_guess=None, std_dev_guess=None, height_guess=None,
                 center_bounds=None, std_dev_bounds=None, height_bounds=None):
    """
    Fit a gaussian function with 3 degrees of freedom.

    Input:
        x_values = the x-axial array of the values
        y_values = the y-axial array of the values
        center_guess = a starting point for the center
        std_dev_guess = a starting point for the std_dev
        height_guess = a starting point for the height

    Returns: center,std_dev,height,covariance
        center = the central value of the gaussian
        std_dev = the standard deviation of the gaussian
        height = the height of the gaussian function along the x-axis
        covariance = a convariance matrix of the fit
    """
    # Type check
    x_values = valid.validate_float_array(x_values)
    y_values = valid.validate_float_array(y_values)

    # Type check optional issues.
    # Type check the guesses
    if (center_guess is not None):
        center_guess = valid.validate_float_value(center_guess)
    else:
        # The default of scipy's curve fit.
        center_guess = 1
    if (std_dev_guess is not None):
        std_dev_guess = valid.validate_float_value(
            std_dev_guess, greater_than=0)
    else:
        # The default of scipy's curve fit.
        std_dev_guess = 1
    if (height_guess is not None):
        height_guess = valid.validate_float_value(height_guess)
    else:
        # The default of scipy's curve fit.
        height_guess = 1
    # Type check bounds.
    if (center_bounds is not None):
        center_bounds = valid.validate_float_array(center_bounds, size=2)
        center_bounds = np.sort(center_bounds)
    else:
        center_bounds = np.array([-np.inf, np.inf])
    if (std_dev_bounds is not None):
        std_dev_bounds = valid.validate_float_array(std_dev_bounds, size=2,
                                                    deep_validate=True,
                                                    greater_than=0)
        std_dev_bounds = np.sort(std_dev_bounds)
    else:
        std_dev_bounds = np.array([0, np.inf])
    if (height_bounds is not None):
        height_bounds = valid.validate_float_array(height_bounds)
        height_bounds = np.sort(height_bounds)
    else:
        height_bounds = np.array([-np.inf, np.inf])

    # Compiling the guesses.
    guesses = np.array([center_guess, std_dev_guess, height_guess])

    # Compiling the bounds
    lower_bounds = (center_bounds[0], std_dev_bounds[0], height_bounds[0])
    upper_bounds = (center_bounds[1], std_dev_bounds[1], height_bounds[1])
    bounds = (lower_bounds, upper_bounds)

    # Use scipy's curve optimization function for the gaussian function.
    fit_parameters, covariance = sp_opt.curve_fit(gaussian_function,
                                                  x_values, y_values,
                                                  p0=guesses, bounds=bounds)

    # For ease.
    center = fit_parameters[0]
    std_dev = fit_parameters[1]
    height = fit_parameters[2]

    return center, std_dev, height, covariance


def fit_multigaussian(x_points, y_points,
                      gaussian_count=None, masking=False,
                      prominence=None, fft_keep=0.01, prom_height_ratio=None):
    """
    Fit a gaussian function with 3 degrees of freedom but with many gaussians.

    Input:
        x_values = the x-axial array of the values
        y_values = the y-axial array of the values
        gaussian_count = the number of expected gaussian functions
        masking = mask out known gaussians to prevent overfitting.
        fft_keep = the percentage kept by the fft truncation, use a lower 
            fft_keep if there is a lot of noise
        prom_height_ratio = the ratio of prominence to height for width 
            detection, a lower value increases accuracy until there are too
            little patterns.


    Returns: center_array,std_dev_array,height_array,covariance_array
        center_array = the central value of the gaussian
        std_dev_array = the standard deviation of the gaussian
        height_array = the height of the gaussian function along the x-axis
        covariance_array = a convariance matrix of the fit
    """
    # Initial variables.
    n_datapoints = len(x_points)
    center_array = []
    std_dev_array = []
    height_array = []
    covariance_array = []

    # Type check.
    x_points = valid.validate_float_array(x_points, size=n_datapoints)
    y_points = valid.validate_float_array(y_points, size=n_datapoints)
    masking = valid.validate_boolean_value(masking)
    if (gaussian_count is not None):
        # Gaussian count can't be less than 0.
        gaussian_count = valid.validate_int_value(
            gaussian_count, greater_than=0)
    fft_keep = valid.validate_float_value(
        fft_keep, greater_than=0, less_than=1)
    if (prominence is not None):
        prominence = valid.validate_float_value(prom_height_ratio,
                                                greater_than=0)
    else:
        prominence = 0.1
    if (prom_height_ratio is not None):
        prom_height_ratio = valid.validate_float_value(prom_height_ratio,
                                                       greater_than=0)
    else:
        prom_height_ratio = 0.25

    # Detect the approximate center values of the gaussians. Using a smoothing
    # fft and ifft.
    fourier_y_points = np.fft.fft(y_points)
    fourier_y_points[int(n_datapoints*fft_keep):] = 0
    inv_fourier_y_points = np.fft.ifft(fourier_y_points)

    # Find the peaks of the smooth fourier transform. Only the values are
    # desired.
    peak_index = sp_sig.find_peaks(np.abs(inv_fourier_y_points),
                                   prominence=0.1)[0]
    peak_index = np.array(peak_index, dtype=int)
    center_estimates = x_points[peak_index]

    # Test if the center estimate found the right amount of gaussians. If not,
    # warn the user, but carry on, it is likely a more or less minor issue.
    if (gaussian_count is not None):
        if (len(center_estimates) < gaussian_count):
            kyubey_warning(OutputError, ('Less gaussians were detected than '
                                         'input as the number of gaussians. '
                                         'Consider raising fft_keep or '
                                         'lowering prominence.'
                                         '    --Kyubey'))
        elif (len(center_estimates) > gaussian_count):
            kyubey_warning(OutputError, ('More gaussians were detected than '
                                         'input as the number of gaussians. '
                                         'Consider lowering fft_keep or '
                                         'raising prominence.'
                                         '    --Kyubey'))

    # Do an initial fit over all possible center estimates. Only attempt to fit
    for guessdex in range(len(center_estimates)):
        # Find the range between to evaluate a fit using half peak width as an
        # approximation for FWHF
        peak_widths = sp_sig.peak_widths(np.abs(inv_fourier_y_points),
                                         peaks=peak_index,
                                         rel_height=prom_height_ratio)
        peak_lower_bounds = np.array(np.floor(peak_widths[2]), dtype=int)
        peak_upper_bounds = np.array(np.ceil(peak_widths[3]), dtype=int)

        # Fit a gaussian using only the points between 2 peak bounds.
        temp_x_points = x_points[peak_lower_bounds[guessdex]:
                                 peak_upper_bounds[guessdex]]
        temp_y_points = y_points[peak_lower_bounds[guessdex]:
                                 peak_upper_bounds[guessdex]]

        temp_center, temp_std_dev, temp_height, temp_covariance = \
            fit_gaussian(temp_x_points, temp_y_points,
                         center_guess=center_estimates[guessdex])

        # Append the values of the fit.
        center_array.append(temp_center)
        std_dev_array.append(temp_std_dev)
        height_array.append(temp_height)
        covariance_array.append(temp_covariance)

    # Test if the user did not want masking, if so, then return data as masking
    # is the only next step.
    if (not masking):
        # Turn into numpy arrays just in case.
        center_array = np.array(center_array, dtype=float)
        std_dev_array = np.array(std_dev_array, dtype=float)
        height_array = np.array(height_array, dtype=float)
        covariance_array = np.array(covariance_array, dtype=float)
        return center_array, std_dev_array, height_array, covariance_array
