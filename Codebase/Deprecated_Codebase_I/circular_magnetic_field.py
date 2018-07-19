import inspect

import numpy as np
import scipy as sp
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import gaussian_fitting as gaussfit
import bessel_fitting as bessfit
import misc_functions as misc


def circular_magnetic_field_cyln(r, phi, z, propagation_function,
                                 ):
    """
    This makes circular magnetic fields, in a way, fields without any 
    divergence. However, this measures not the tangential vector, but rotational
    vector.
    """

    # Type check
    r = valid.validate_float_array(r, deep_validate=True, greater_than=0)
    phi = valid.validate_float_array(phi, deep_validate=True,
                                         greater_than=0, less_than=2*np.pi)
    z = valid.validate_float_array(z)

    # Because of its invariantness in phi and z, also, only the r value 
    # matters in this function.
    B_r = np.zeros_like(r) 
    B_phi = propagation_function(r)
    B_z = np.zeros_like(r) 

    # Return
    return B_r, B_phi, B_z

    
def circular_magnetic_field_cart(x, y, propagation_function,
                                 tangential_axis='z'):
    """
    This makes circular magnetic fields, in a way, fields without any 
    divergence. However, this measures not the rotational vector, but tangential
    vector.

    The tangential axis is the axis of which is the axis of rotation for the
    field. Assume that the positive direction is pointing to the user.
    """

    # Type check
    x = valid.validate_float_array(x)
    y = valid.validate_float_array(y)

    # Convert to a polar system for tangential vector.
    r_subaxis = np.hypot(x,y)
    phi_subaxis = np.arctan2(y,x)

    # Calculate the magnitude of the tangential vector.
    B_t = propagation_function(r_subaxis)

    # The vector is tangent to a circle made by r, thus the angle is related to
    # phi, but is not phi.
    B_angle = phi_subaxis + np.pi/2

    # Calculate the components of the magnetic field vector based on the 
    # magnitude and the angle.
    B_x = B_t * np.cos(B_angle)
    B_y = B_t * np.sin(B_angle)

    # Return
    return B_x, B_y