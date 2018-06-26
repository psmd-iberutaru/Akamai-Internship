import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid


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
                 * np.exp(-0.5 * (x_input - (center/std_dev))**2))
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


def generate_multigaussian(center_list, std_dev_list, height_list,x_domain_list,                           gaussian_count=None,n_datapoints_list=None,   
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
        n_datapoints = valid.validate_int_value(n_datapoints,greater_than=0)
        # Define subpoints
        n_subpoints = np.ceil(n_datapoints/gaussian_count)
        n_datapoints_list = np.full(gaussian_count, n_subpoints)
    
    for gaussiandex in range(gaussian_count):
        # Generate one gaussian.
        temp_x_points,temp_y_points = \
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
            temp_y_points = np.concatenate((temp_y_points,added_y_points),
                                           axis=0)
        # Extract only the maximum values.
        temp_y_points = np.amax(temp_y_points,axis=0)

        # Append the values for storage.
        x_points.append(temp_x_points)
        y_points.append(temp_y_points)

    # Morph to numpy arrays.
    x_points = np.array(x_points,dtype=float)
    y_points = np.array(y_points,dtype=float)
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
                raise OutputError('The number of points is less than expected.')
            
            removed_indexes = np.random.randint(0,x_points.size - 1,
                                                delta_size)
            x_points = np.delete(x_points,removed_indexes)
            y_points = np.delete(y_points,removed_indexes)
    
    # Sort the values before returning.
    sort_index = np.argsort(x_points)
    x_points = x_points[sort_index]
    y_points = y_points[sort_index]

    return np.array(x_points,dtype=float),np.array(y_points,dtype=float)


def generate_noise(input_array, noise_domain, distribution='uniform',
                   center=None, std_dev=None,  # Normal distribution terms.
                   debug=False):
    """
    Takes a set of 'perfect' datapoints and scatters them based on some 
    randomly generated noise. The generated noise can be distributed in a number
    of ways.

    Input:
    input_array = array of datapoints to be scattered from the original value
    noise_domain = (2,) shaped array of the upper and lower bounds of scattering
    distribution = method of random number distribution
                    - 'uniform'
                    - 'gaussian'
    debug = Debug mode

    Output:
    output_array = scattered 'noisy' datapoints
    """
    # Type check. 
    input_array = valid.validate_float_array(input_array,size=len(input_array))
    noise_domain = valid.validate_float_array(noise_domain, shape=(2,), size=2)
    distribution = valid.validate_string(distribution)

    # Initial conditions
    n_datapoints = len(input_array)

    # Ensure the lower bound of the noise domain is the first element.
    if (noise_domain[0] < noise_domain[-1]):
        # This is correct behavior.
        pass
    elif (noise_domain[0] > noise_domain[-1]):
        # Warn and change, the array seems to be reversed.
        noise_domain = np.flip(noise_domain, axis=0)
    elif (noise_domain[0] == noise_domain[-1]):
        raise ValueError('Noise domain range is detected to be zero. There is '
                         'no functional use of this function.    --Kyubey')

    # Check for distribution method, generate noise array from method.
    if (distribution == 'uniform'):
        if (debug):
            print('Noise distribution set to "uniform".')
        noise_array = np.random.uniform(noise_domain[0], noise_domain[1],
                                        size=n_datapoints)
    elif ((distribution == 'gaussian') or (distribution == 'normal')):
        if (debug):
            print('Noise distribution set to "gaussian".')
            kyubey_warning(OutputWarning, ('Noise domain is ignored under '
                                           'gaussian distribution.    --Kyubey'))
        # Type check center and standard deviation.
        if (std_dev is None):
            raise InputError('Noise distribution is set to gaussian, there is '
                             'no standard deviation input.')
        else:
            # Standard deviation cannot be negative
            center = valid.validate_float_value(center)
            std_dev = valid.validate_float_value(std_dev, greater_than=0)
            noise_array = np.random.normal(center, std_dev, size=n_datapoints)

    # Noise array plus standard values.
    return input_array + noise_array


def generate_noisy_gaussian(center, std_dev, height,
                            noise_domain=[-0.1, 0.1], x_domain=[-3, 3],
                            n_datapoints=10000):
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

    # Make the x-axis value array given the domain and the number of points.
    x_values = np.random.uniform(x_domain[0], x_domain[-1], n_datapoints)

    # Generate the gaussian function and map to an output with the input
    # parameters.
    y_values = gaussian_function(x_values, center, std_dev, height)

    # Imbue the gaussian with random noise.
    y_values = generate_noise(y_values, noise_domain, distribution='uniform')

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
    center_list = valid.validate_float_array(center_list,size=gaussian_count)
    std_dev_list = valid.validate_float_array(std_dev_list,size=gaussian_count)
    height_list = valid.validate_float_array(height_list,size=gaussian_count)
    noise_domain_list = valid.validate_float_array(noise_domain_list,
                                                   shape=(gaussian_count,2))
    x_domain_list = valid.validate_float_array(x_domain_list,
                                               shape=(gaussian_count,2))
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
            temp_x_points,temp_y_points = \
                generate_gaussian(center_list[gaussiandex],
                                  std_dev_list[gaussiandex],
                                  height_list[gaussiandex],
                                  x_domain_list[gaussiandex],
                                  n_datapoints_list[gaussiandex])
            temp_y_points = generate_noise(temp_y_points,
                                           noise_domain_list[gaussiandex],
                                           distribution='uniform')
            # Store for return
            x_values = np.append([x_values],[temp_x_points],axis=0)
            y_values = np.append([y_values],[temp_y_points],axis=0)

        # Maximize the values, discarding everything lower than.
        x_values = np.amax(x_values,axis=0)
        y_values = np.amax(y_values,axis=0)

    else:
        # Generate noise of every point after gaussian generation.
        # Generate gaussian
        x_values,y_values = generate_multigaussian(center_list,std_dev_list,
                                                   height_list,x_domain_list,
                                                   gaussian_count,
                                                   n_datapoints_list)
        
        # Generate noise. Warn the user that only the first noise domain is
        # being used.
        kyubey_warning(OutputWarning,('Only the first element of the '
                                      'noise_domian_list is used if ' 
                                      'cumulative_noise is False.'
                                      '    --Kyubey'))
        y_values = generate_noise(y_values,noise_domain_list[0],
                                  distribution='uniform')

    return np.array(x_values,dtype=float),np.array(y_values,dtype=float)


def fit_gaussian(x_values, y_values):
    """
    Fit a gaussian function with 4 degrees of freedom.

    Input:
        x_values = the x-axial array of the values
        y_values = the y-axial array of the values

    Returns: fit_parameters[center,std_dev,height],covariance
        fit_parameters = an array containing the values of the fit
        center = the central value of the gaussian
        std_dev = the standard deviation of the gaussian
        height = the height of the gaussian function along the x-axis
        covariance = a convariance matrix of the fit
    """

    # Use scipy's curve optimization function for the gaussian function.
    fit_parameters, covariance = sp_opt.curve_fit(gaussian_function, x_values,
                                                  y_values)

    return fit_parameters, covariance