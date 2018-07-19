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
    y_output = (((1 / (std_dev * np.sqrt(2 * np.pi)))
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
        x_domain = absolute domain of the gaussian function 
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
                           x_domain, gaussian_count=None,
                           n_datapoints=None):
    """
    Generates a multigaussian arrangement of datapoints.
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
    x_domain = valid.validate_float_array(x_domain,
                                          shape=(2,),
                                          size=2)
    n_datapoints = valid.validate_int_value(n_datapoints)

    # Initial parameters.
    x_values = np.random.uniform(x_domain[0], x_domain[-1],
                                 size=n_datapoints)
    y_values = []

    # Compile the parameters into a concentric list for the usage of the
    # envelope function.
    parameters = []
    for gaussiandex in range(gaussian_count):
        temp_parameter_dict = {'center': center_list[gaussiandex],
                               'std_dev': std_dev_list[gaussiandex],
                               'height': height_list[gaussiandex]}
        parameters.append(temp_parameter_dict)
    parameters = tuple(parameters)

    # Compile the list of functions for the concentric list. As this is multi-
    # gaussian fitting, it is expected to only be gaussian functions.
    functions = []
    for gaussiandex in range(gaussian_count):
        functions.append(gaussian_function)
    functions = tuple(functions)

    # Execute the envelope function.
    y_values = misc.generate_function_envelope(x_values, functions, parameters)

    # Sort the values.
    sort_index = np.argsort(x_values)
    x_values = x_values[sort_index]
    y_values = y_values[sort_index]

    return np.array(x_values, dtype=float), np.array(y_values, dtype=float)


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
    std_dev = valid.validate_float_value(std_dev, greater_than=0)
    height = valid.validate_float_value(height)
    x_domain = valid.validate_float_array(x_domain, shape=(2,), size=2)
    noise_domain = valid.validate_float_array(noise_domain, shape=(2,), size=2)
    n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)

    # Generate the gaussian function and map to an output with the input
    # parameters.
    x_values, y_values = generate_gaussian(center, std_dev, height,
                                           x_domain=x_domain,
                                           n_datapoints=n_datapoints)

    # Imbue the gaussian with random noise.
    y_values = misc.generate_noise(y_values, noise_domain,
                                   distribution='uniform')

    return x_values, y_values


def generate_noisy_multigaussian(center_list, std_dev_list, height_list,
                                 noise_domain_list, x_domain, n_datapoints,
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
    x_domain = valid.validate_float_array(x_domain,
                                          shape=(2,), size=2)
    cumulative_noise = valid.validate_boolean_value(cumulative_noise)
    n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)

    # Type check optional elements
    gaussian_count = valid.validate_int_value(gaussian_count, greater_than=0)
    cumulative_noise = valid.validate_boolean_value(cumulative_noise)

    # Initialize initial variables.
    x_values = []
    y_values = []

    # Check how to distribute noise.
    if (cumulative_noise):
        # Each gaussian must be generated on its own.
        for gaussiandex in range(gaussian_count):
            # Generate gaussian.
            temp_x_values, temp_y_values = \
                generate_gaussian(center_list[gaussiandex],
                                  std_dev_list[gaussiandex],
                                  height_list[gaussiandex],
                                  x_domain,
                                  np.ceil(n_datapoints / gaussian_count))
            temp_y_values = misc.generate_noise(temp_y_values,
                                                noise_domain_list[gaussiandex],
                                                distribution='uniform')
            # Store for return
            x_values = np.append([x_values], [temp_x_values], axis=0)
            y_values = np.append([y_values], [temp_y_values], axis=0)

        # Maximize the values, discarding everything lower than.
        x_values = np.amax(x_values, axis=0)
        y_values = np.amax(y_values, axis=0)

    else:
        # Generate noise of every point after gaussian generation.
        # Generate gaussian
        x_values, y_values = generate_multigaussian(center_list, std_dev_list,
                                                    height_list, x_domain,
                                                    gaussian_count,
                                                    np.ceil(n_datapoints /
                                                            gaussian_count))

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
    # The total number of points, useful.
    try:
        n_datapoints = len(x_values)
    except:
        raise InputError('It does not make sense to try and fit a '
                         'single point.'
                         '    --Kyubey')
    else:
        n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)

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


def fit_multigaussian(x_values, y_values,
                      gaussian_count=None,
                      window_len_ratio=0.1, sg_polyorder=3,
                      prominence=0.10,
                      *args, **kwargs):
    """
    Fit a gaussian function with 3 degrees of freedom but with many gaussians.

    Input:
        x_values = the x-axial array of the values
        y_values = the y-axial array of the values
        gaussian_count = the number of expected gaussian functions
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
    # The total number of points, useful.
    try:
        n_datapoints = len(x_values)
    except:
        raise InputError('It does not make sense to try and fit a '
                         'single point.'
                         '    --Kyubey')
    else:
        n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)

    # Initial variables.
    center_array = []
    std_dev_array = []
    height_array = []
    covariance_array = []

    # Type check.
    x_values = valid.validate_float_array(x_values, size=n_datapoints)
    y_values = valid.validate_float_array(y_values, size=n_datapoints)
    if (gaussian_count is not None):
        # Gaussian count can't be less than 0.
        gaussian_count = valid.validate_int_value(
            gaussian_count, greater_than=0)
    window_len_ratio = valid.validate_float_value(window_len_ratio)
    sg_polyorder = valid.validate_int_value(sg_polyorder)
    prominence = valid.validate_float_value(prominence)

    # Implement the Savitzky-Golay filtering algorithm.
    # Window width needs to be an odd interger by Scipy and algorithm
    # stipulation.
    window_width = int(window_len_ratio * n_datapoints)
    if (window_width % 2 == 0):
        # It is even, make odd.
        window_width += 1
    elif (window_width % 2 == 1):
        # It is odd, it should be good.
        pass

    filtered_y_values = sp_sig.savgol_filter(y_values,
                                             window_width,
                                             sg_polyorder)

    # Detect possible peaks of Gaussian functions.
    peak_index, peak_properties = \
        sp_sig.find_peaks(filtered_y_values, prominence=prominence)
    left_bases = peak_properties['left_bases']
    right_bases = peak_properties['right_bases']

    # Attempt to fit a gaussian curve between the ranges of each peak.
    for peakdex, left_basedex, right_basedex in \
            zip(peak_index, left_bases, right_bases):
        # Separate each of the gaussians and try to find parameters.
        center, std_dev, height, covariance = \
            fit_gaussian(x_values[left_basedex:right_basedex],
                         y_values[left_basedex:right_basedex],
                         center_guess=x_values[peakdex])

        # Append the values to the arrays of information.
        center_array.append(center)
        std_dev_array.append(std_dev)
        height_array.append(height)
        covariance_array.append(covariance)

    # Type check before returning, just in case.
    center_array = valid.validate_float_array(center_array)
    std_dev_array = valid.validate_float_array(std_dev_array)
    height_array = valid.validate_float_array(height_array)
    covariance_array = valid.validate_float_array(covariance_array)

    return center_array, std_dev_array, height_array, covariance_array
