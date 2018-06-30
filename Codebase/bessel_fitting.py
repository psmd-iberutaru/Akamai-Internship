import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import misc_functions as misc


def bessel_function_1st(x_input, order):
    """
    This is a wrapper function around scipy's bessel function of the
    first kind, given any real order value. This wrapper also includes
    input verification.

    Input:
    x_input = The x value input of the bessel function.
    order = The value(s) of orders wanting to be computed.

    Output:
    y_output = The y value(s) from the bessel function.
    """

    # Type check.
    x_input = valid.validate_float_array(x_input)
    if isinstance(order, (float, int)):
        order = valid.validate_float_value(order)
    else:
        order = valid.validate_float_array(order, deep_validate=True)

    # Compute the value(s) of the bessel_function
    y_output = sp_spcl.jv(order, x_input)

    return np.array(y_output, dtype=float)


def bessel_function_2nd(x_input, order):
    """
    This is a wrapper function around scipy's bessel function of the
    second kind, given any real order value. This wrapper also includes
    input verification.

    Input:
    x_input = The x value input of the bessel function.
    order = The value(s) of orders wanting to be computed.
    center = The x displacement from center (i.e. the new center)
    height = The y displacement from center (i.e. the new height offset)

    Output:
    y_output = The y value(s) from the bessel function.
    """

    # Type check.
    x_input = valid.validate_float_array(x_input)
    if isinstance(order, (float, int)):
        order = valid.validate_float_value(order)
    else:
        order = valid.validate_float_array(order, deep_validate=True)

    # Compute the value(s) of the bessel_function
    y_output = sp_spcl.yv(order, x_input)

    return np.array(y_output, dtype=float)


def generate_noisy_bessel_1st(order, x_domain, noise_domain, n_datapoints,
                              distribution='uniform'):
    """
    This function generates a noisy bessel function of the first kind given
    a real order.

    Input:
    order = The real order of the bessel function
    x_domain = The range of x_values that should be plotted.
    noise_domain = The domain the noise is allowed to spread around.
    n_datapoints = The number of data points that is desired.
    distribution = The method of noise distribution.

    Output:
    y_output = The values of the function after noise.    
    """

    # Type check.
    order = valid.validate_float_value(order)
    x_domain = valid.validate_float_array(x_domain, shape=(2,), size=2)
    noise_domain = valid.validate_float_array(noise_domain, shape=(2,), size=2)
    distribution = valid.validate_string(distribution)

    # Generate the input values. Make sure the first element is the lower
    # element.
    x_domain = np.sort(x_domain)
    x_input = np.random.uniform(x_domain[0], x_domain[-1], n_datapoints)

    # Generate values for the bessel function.
    y_output = bessel_function_1st(x_input, order)

    # Imbue the values with noise.
    y_output = misc.generate_noise(y_output, noise_domain,
                                   distribution=distribution)

    # Sort the values for ease of plotting and computation.
    sort_index = np.argsort(x_input)
    x_input = x_input[sort_index]
    y_output = y_output[sort_index]

    return np.array(x_input, dtype=float), np.array(y_output, dtype=float)


def generate_noisy_bessel_2nd(order, x_domain, noise_domain, n_datapoints,
                              distribution='uniform'):
    """
    This function generates a noisy bessel function of the second kind given
    a real order.

    Input:
    order = The real order of the bessel function
    x_domain = The range of x_values that should be plotted.
    noise_domain = The domain the noise is allowed to spread around.
    n_datapoints = The number of data points that is desired.
    distribution = The method of noise distribution.

    Output:
    y_output = The values of the function after noise.    
    """

    # Type check.
    order = valid.validate_float_value(order)
    x_domain = valid.validate_float_array(x_domain, shape=(2,), size=2)
    noise_domain = valid.validate_float_array(noise_domain, shape=(2,), size=2)
    distribution = valid.validate_string(distribution)

    # Generate the input values. Make sure the first element is the lower
    # element.
    x_domain = np.sort(x_domain)
    x_input = np.random.uniform(x_domain[0], x_domain[-1], n_datapoints)

    # Generate values for the bessel function.
    y_output = bessel_function_2nd(x_input, order)

    # Imbue the values with noise.
    y_output = misc.generate_noise(y_output, noise_domain,
                                   distribution=distribution)

    # Sort the values for ease of plotting and computation.
    sort_index = np.argsort(x_input)
    x_input = x_input[sort_index]
    y_output = y_output[sort_index]

    return np.array(x_input, dtype=float), np.array(y_output, dtype=float)


def fit_bessel_function_1st(x_points, y_points,
                            order_guess=None, order_bounds=None):
    """
    This function returns the order of a Bessel function of the second kind 
    that fits the data points according to a least squares fitting algorithm.

    Input:
    x_points = The x values of the points to fit.
    y_points = The y values of the points to fit.
    order_guess = A starting point for order guessing.
    order_bounds = The min and max values the order can be.

    Output:
    fit_order = The value of the order of the fit bessel function.
    """
    # The total number of points, useful.
    try:
        n_datapoints = len(x_points)
    except:
        raise InputError('It does not make sense to try and fit a '
                         'single point.'
                         '    --Kyubey')
    else:
        n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)

    # Type check
    x_points = valid.validate_float_array(x_points, size=n_datapoints)
    y_points = valid.validate_float_array(y_points, size=n_datapoints)
    if (order_guess is not None):
        order_guess = valid.validate_float_value(order_guess)
    else:
        order_guess = 1
    if (order_bounds is not None):
        order_bounds = valid.validate_float_array(order_bounds,
                                                  shape=(2,), size=2)
    else:
        order_bounds = (-np.inf, np.inf)

    # Function fitting, Scipy's module is likely the better method to go.
    fit_parameters = sp_opt.curve_fit(bessel_function_1st, x_points, y_points,
                                      p0=order_guess, bounds=order_bounds)

    # Split the fitted order and covariance array.
    fit_order = float(fit_parameters[0])
    covariance = float(fit_parameters[1])

    return fit_order, covariance


def fit_bessel_function_2nd(x_points, y_points,
                            order_guess=None, order_bounds=None):
    """
    This function returns the order of a Bessel function of the second kind 
    that fits the data points according to a least squares fitting algorithm.

    Input:
    x_points = The x values of the points to fit.
    y_points = The y values of the points to fit.
    order_guess = A starting point for order guessing.
    order_bounds = The min and max values the order can be.

    Output:
    fit_order = The value of the order of the fit bessel function.
    """
    # The total number of points, useful.
    try:
        n_datapoints = len(x_points)
    except:
        raise InputError('It does not make sense to try and fit a '
                         'single point.'
                         '    --Kyubey')
    else:
        n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)

    # Type check
    x_points = valid.validate_float_array(x_points, size=n_datapoints)
    y_points = valid.validate_float_array(y_points, size=n_datapoints)
    if (order_guess is not None):
        order_guess = valid.validate_float_value(order_guess)
    else:
        order_guess = 1
    if (order_bounds is not None):
        order_bounds = valid.validate_float_array(order_bounds,
                                                  shape=(2,), size=2)
    else:
        order_bounds = (-np.inf, np.inf)

    # Function fitting, Scipy's module is likely the better method to go.
    fit_parameters = sp_opt.curve_fit(bessel_function_2nd, x_points, y_points,
                                      p0=order_guess, bounds=order_bounds)

    # Split the fitted order and covariance array.
    fit_order = float(fit_parameters[0])
    covariance = float(fit_parameters[1])

    return fit_order, covariance


def fit_bessel_function_1st_integer(x_values, y_values,
                                    max_order=1000, negative_order=False):
    """
    This is the main bessel fitting algorithm. It can fit, and then extract, 
    the order of the bessel function according to the best fit of the data 
    points.

    However, it can only do integer orders, and will not attempt float orders or
    complex orders.

    Input:
    x_values = Input x values
    y_values = Input y values
    max_order = The absolute value of the max order that is to be checked.
    negative_order = If the user prefers the negative order if the positive and
        negative order is the same.

    Output:
    fit_order = The primary positive fit order
    """
    # Important information and validate.
    try:
        n_datapoints = len(x_values)
    except:
        x_values = valid.validate_float_array(x_values)
        n_datapoints = len(x_values)
    finally:
        # Type check
        x_values = valid.validate_float_array(x_values, size=n_datapoints)
        y_values = valid.validate_float_array(y_values, size=n_datapoints)
    negative_order = valid.validate_boolean_value(negative_order)

    # Because an order of 0 is a thing.
    searchable_orders = range(max_order + 1)

    # Generate list of output values, given inputs and searching through all
    # given possible orders.
    output_ranges = []
    for orderdex in searchable_orders:
        output_ranges.append(bessel_function_1st(x_values, orderdex))

    # Copy the positive domain and transpose to the negative order domain via
    # the relationship of non-linearity of Bessel functions of the the first
    # kind of interger order.
    positive_order_outputs = np.array(output_ranges)
    negative_order_outputs = np.array(output_ranges)
    for orderdex, output_array in enumerate(negative_order_outputs):
        negative_order_outputs[orderdex] *= (-1)**orderdex

    # Use the least squares fitting algorithm.

    # Define square residuals for both positive and negative.
    positive_order_sq_residuals = \
        np.sum((y_values - positive_order_outputs[:])**2, axis=1)
    negative_order_sq_residuals = \
        np.sum((y_values - negative_order_outputs[:])**2, axis=1)

    # Find the lowest square residual of both the positive and negative values.
    positive_order_fit = np.argmin(positive_order_sq_residuals)
    negative_order_fit = np.argmin(negative_order_sq_residuals)

    # It might be the case that the values of the negative and the positive
    # order is the same via the non-linearity relationship, if so, send the
    # positive value if the negative order is not wanted.
    if (positive_order_fit == negative_order_fit):
        if (negative_order):
            return int(negative_order_fit)
        else:
            return int(positive_order_fit)
    elif (positive_order_sq_residuals[positive_order_fit]
          < negative_order_sq_residuals[negative_order_fit]):
        return int(positive_order_fit)
    elif (positive_order_sq_residuals[positive_order_fit]
          > negative_order_sq_residuals[negative_order_fit]):
        return int(negative_order_fit)
    else:
        raise OutputError('There seems to be something wrong, it is not known'
                          'what is wrong.'
                          '    --Kyubey')
