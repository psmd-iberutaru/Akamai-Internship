import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import scipy.signal as sp_sig
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid

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

