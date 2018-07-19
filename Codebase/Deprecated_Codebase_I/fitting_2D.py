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


def dual_dimensional_gauss_equation(input_points, center, std_dev, height,
                                    dimensions):
    """
    This function generates gaussians of multiple dimensions/variables given
    the center's coordinates and the covariance matrix.
    """
    try:
        n_datapoints = len(input_points[0])
    except:
        input_points = valid.validate_float_array(input_points)
        n_datapoints = len(input_points[0])

    # Validate, dimensions must go first.
    dimensions = valid.validate_int_value(dimensions, greater_than=0)
    input_points = valid.validate_float_array(input_points,
                                              shape=(2, n_datapoints))
    center = valid.validate_float_array(center,
                                        shape=(dimensions,), size=dimensions)
    std_dev = valid.validate_float_array(std_dev,
                                         shape=(dimensions,), size=dimensions,
                                         deep_validate=True, greater_than=0)
    height = valid.validate_float_value(height)

    # For two dimensions.
    normalization_term = 1 / (2 * np.pi * std_dev[0] * std_dev[1])

    exp_x_term = (input_points[0] - center[0])**2 / (2 * std_dev[0]**2)
    exp_y_term = (input_points[1] - center[1])**2 / (2 * std_dev[1]**2)

    z_points = normalization_term * np.exp(-(exp_x_term + exp_y_term)) + height

    output_points = np.append(input_points, np.array([z_points]), axis=0)

    return z_points, output_points


def dual_dimensional_gauss_equation_rot(input_points, centers, std_devs, height,
                                        theta, dimensions):
    """
    This is the general gaussian equation for a rotatable gaussian for some 
    angle theta (in radians).
    """
    try:
        n_datapoints = len(input_points[0])
    except:
        input_points = valid.validate_float_array(input_points)
        n_datapoints = len(input_points[0])

    # Validate, dimensions must go first.
    dimensions = valid.validate_int_value(dimensions, greater_than=0)
    input_points = valid.validate_float_array(input_points,
                                              shape=(2, n_datapoints))
    centers = valid.validate_float_array(centers,
                                         shape=(dimensions,), size=dimensions)
    std_devs = valid.validate_float_array(std_devs,
                                          shape=(dimensions,), size=dimensions,
                                          deep_validate=True, greater_than=0)
    height = valid.validate_float_value(height)
    # Adapt for over/under rotation of theta.
    try:
        theta = valid.validate_float_value(theta,
                                           greater_than=0, less_than=2*np.pi)
    except ValueError:
        # A loop is to be done. Have an insurance policy.
        loopbreak = 0
        while ((theta < 0) or (theta > 2 * np.pi)):
            if (theta < 0):
                theta += 2*np.pi
            elif (theta > 0):
                theta = theta % (2 * np.pi)
            # Ensure that the loop does not get stuck in the event of
            # unpredicted behavior.
            loopbreak += 1
            if (loopbreak > 100):
                raise InputError('The value of theta cannot be '
                                 'nicely confined to 0 <= θ <= 2π '
                                 '    --Kyubey')

    # Following Wikipedia's parameter definitions.
    a = ((np.cos(theta)**2 / (2*std_devs[0]**2))
         + (np.sin(theta)**2 / (2*std_devs[1]**2)))
    b = (-(np.sin(2*theta)/(4*std_devs[0]**2))
         + (np.sin(2*theta)/(4*std_devs[1]**2)))
    c = ((np.sin(theta)**2 / (2*std_devs[0]**2))
         + (np.cos(theta)**2 / (2*std_devs[1]**2)))

    # Amplitude or normalization
    normalization_term = 1 / (2 * np.pi * std_devs[0] * std_devs[1])

    # General equation
    z_values = (normalization_term
                * np.exp(-(a * (input_points[0] - centers[0])**2
                           + (2 * b * ((input_points[0] - centers[0])
                                       * (input_points[1] - centers[1])))
                           + (c * (input_points[1] - centers[1])**2))))

    # Return values.
    output_points = np.append(input_points, np.array([z_values]), axis=0)

    return z_values, output_points


def generate_dual_dimension_gaussian(centers, std_devs, height, n_datapoints,
                                     x_domain, y_domain,
                                     dimensions=2):
    """
    This generates random points for a 2D dimensional gaussian. 
    """

    # Type check
    dimensions = valid.validate_int_value(dimensions, greater_than=0)
    centers = valid.validate_float_array(centers,
                                         shape=(dimensions,), size=dimensions)
    std_devs = valid.validate_float_array(std_devs,
                                          shape=(dimensions,), size=dimensions,
                                          deep_validate=True, greater_than=0)
    height = valid.validate_float_value(height)
    n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)
    x_domain = valid.validate_float_array(x_domain,
                                          shape=(2,), size=2)
    y_domain = valid.validate_float_array(y_domain,
                                          shape=(2,), size=2)

    # Generate x and y points at random.
    x_values = np.random.uniform(x_domain[0], x_domain[-1], size=n_datapoints)
    y_values = np.random.uniform(y_domain[0], y_domain[-1], size=n_datapoints)

    # Compile into a parallel pair of (x,y)
    input_points = np.append([x_values], [y_values], axis=0)

    # Compute the z_values, desire only the output points.
    z_values, output_points = dual_dimensional_gauss_equation(
        input_points, centers, std_devs, height, dimensions)

    return output_points


def generate_noisy_dual_dimension_gaussian(centers, std_devs, height,
                                           n_datapoints, x_domain, y_domain,
                                           noise_domain,
                                           dimensions=2):
    """
    This generates a noisy 2D gaussian.
    """

    # Type check
    dimensions = valid.validate_int_value(dimensions, greater_than=0)
    centers = valid.validate_float_array(centers,
                                         shape=(dimensions,), size=dimensions)
    std_devs = valid.validate_float_array(std_devs,
                                          shape=(dimensions,), size=dimensions,
                                          deep_validate=True, greater_than=0)
    height = valid.validate_float_value(height)
    n_datapoints = valid.validate_int_value(n_datapoints, greater_than=0)
    x_domain = valid.validate_float_array(x_domain,
                                          shape=(2,), size=2)
    y_domain = valid.validate_float_array(y_domain,
                                          shape=(2,), size=2)
    noise_domain = valid.validate_float_array(noise_domain,
                                              shape=(2,), size=2)

    # Generate the 2D gaussian.
    points = generate_dual_dimension_gaussian(centers, std_devs, height,
                                              n_datapoints, x_domain, y_domain)

    # Imbue the z points (2 index) with noise.
    points[2] = misc.generate_noise(points[2], noise_domain)

    return points


def fit_dual_dimension_gaussian(points,
                                center_cutoff_factor=0.05,
                                height_cutoff_factor=0.42,
                                strip_width=0.2):
    """
    This function computes the values that describe a given 2D elliptical 
    gaussian.
    """
    try:
        n_datapoints = len(points[0])
    except:
        points = valid.validate_float_array(points)
        n_datapoints = len(points[0])

    # Type check, three dimensional points are expected.
    points = valid.validate_float_array(points, shape=(3, n_datapoints))
    center_cutoff_factor = valid.validate_float_value(center_cutoff_factor,
                                                      greater_than=0,
                                                      less_than=1)
    height_cutoff_factor = valid.validate_float_value(height_cutoff_factor,
                                                      greater_than=0,
                                                      less_than=1)
    strip_width = valid.validate_float_value(strip_width, greater_than=0)

    # Sort based off of z-values for convince.
    sort_index = np.argsort(points[2])
    points = points[:, sort_index]

    # Attempt to find the height of the gaussian. A weighted average of the
    # lowest z-points should be alright. The lower ceil(42%) is used just for
    # fun. The weight function is arbitrary, but used as it favors small values,
    # the points that might be considered at the bottom of the gaussian.
    height_cutoff = int(np.ceil(height_cutoff_factor * n_datapoints))
    fit_height = np.average(points[2, :height_cutoff],
                            weights=(1/points[2, :height_cutoff]**2))

    # Do a translation to "zero-out" the datapoints along the z-axis
    points[2] -= fit_height

    # Attempt to find the center of the gaussian through weighted averages over
    # both axis. The cut off is such that the very low valued points do not
    # over power the average and the weights through attrition. The value of
    # ceil(5%) is arbitrary.
    center_cutoff = int(np.ceil(center_cutoff_factor * n_datapoints))
    x_center_fit = np.average(points[0, -center_cutoff:],
                              weights=points[2, -center_cutoff:]**2)
    y_center_fit = np.average(points[1, -center_cutoff:],
                              weights=points[2, -center_cutoff:]**2)
    fit_center = np.array([x_center_fit, y_center_fit], dtype=float)

    # Do a translation to center the datapoints along the xy-plane.
    points[0] -= x_center_fit
    points[1] -= y_center_fit

    # Determine the standard deviation. The normal fitting gaussian function
    # does not work as well because of the built in normalization factor.
    def subgauss(x_input, std_dev, amp, height):
        """
        This is a modified superset of gaussians.
        """
        # The centers should have already been detected and shifted. A priori
        # value
        center = 0
        # amp being the amplitude
        return ((amp * np.exp(-0.5 * ((x_input - center)/std_dev)**2))
                + height)

    # Extract a strip of values along the x and y axes.
    x_valid_points = np.where(np.abs(points[1]) <= strip_width / 2.0)
    x_strip_points = points[:, x_valid_points[0]]

    y_valid_points = np.where(np.abs(points[0]) <= strip_width / 2.0)
    y_strip_points = points[:, y_valid_points[0]]

    x_gauss_ans = sp_opt.curve_fit(subgauss,
                                   x_strip_points[0], x_strip_points[2])
    y_gauss_ans = sp_opt.curve_fit(subgauss,
                                   y_strip_points[1], y_strip_points[2])

    # The only value desired is the standard deviation.
    x_std_dev = float(x_gauss_ans[0][0])
    y_std_dev = float(y_gauss_ans[0][0])
    fit_std_dev = np.array([x_std_dev, y_std_dev],dtype=float)

    # Package all of the obtained values. And return.
    return fit_center, fit_std_dev, fit_height
