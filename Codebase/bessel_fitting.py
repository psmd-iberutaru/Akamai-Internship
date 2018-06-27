import numpy as np 
import scipy as sp 
import scipy.optimize as sp_opt
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import misc_functions as misc

def bessel_function_1st(x_input,order):
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
    order = valid.validate_float_array(order,deep_validate=True)

    # Compute the value(s) of the bessel_function
    y_output = sp_spcl.jv(order,x_input)

    return np.array(y_output,dtype=float)

def bessel_function_2nd(x_input,order):
    """
    This is a wrapper function around scipy's bessel function of the
    second kind, given any real order value. This wrapper also includes
    input verification.

    Input:
    x_input = The x value input of the bessel function.
    order = The value(s) of orders wanting to be computed.

    Output:
    y_output = The y value(s) from the bessel function.
    """

    # Type check.
    x_input = valid.validate_float_array(x_input)
    order = valid.validate_float_array(order,deep_validate=True)

    # Compute the value(s) of the bessel_function
    y_output = sp_spcl.yv(order,x_input)

    return np.array(y_output,dtype=float)


def generate_noisy_bessel_1st(x_input,order,noise_domain,
                              distribution='uniform'):
    """
    This function generates a noisy bessel function of the first kind given
    a real order.

    Input:
    x_input = The input values of the function
    order = The real order of the bessel function
    noise_domain = The domain the noise is allowed to spread around.

    Output:
    y_output = The values of the function after noise.    
    """

    # Type check.
    x_input = valid.validate_float_array(x_input)
    order = valid.validate_float_value(order)
    noise_domain = valid.validate_float_array(noise_domain,shape=(2,),size=2)
    distribution = valid.validate_string(distribution)

    # Generate values for the bessel function.
    y_output = bessel_function_1st(x_input,order)

    # Imbue the values with noise.
    y_output = misc.generate_noise(y_output,noise_domain, 
                                   distribution=distribution)

    return np.array(y_output,dtype=float)

def generate_noisy_bessel_2nd(x_input,order,noise_domain,
                              distribution='uniform'):
    """
    This function generates a noisy bessel function of the second kind given
    a real order.

    Input:
    x_input = The input values of the function
    order = The real order of the bessel function
    noise_domain = The domain the noise is allowed to spread around.

    Output:
    y_output = The values of the function after noise.    
    """

    # Type check.
    x_input = valid.validate_float_array(x_input)
    order = valid.validate_float_value(order)
    noise_domain = valid.validate_float_array(noise_domain,shape=(2,),size=2)
    distribution = valid.validate_string(distribution)

    # Generate values for the bessel function.
    y_output = bessel_function_2nd(x_input,order)

    # Imbue the values with noise.
    y_output = misc.generate_noise(y_output,noise_domain, 
                                   distribution=distribution)

    return np.array(y_output,dtype=float)
