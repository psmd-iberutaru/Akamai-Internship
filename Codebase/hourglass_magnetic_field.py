# Pylint has some problems with scipy.special.erfc
# pylint: skip-file

import numpy as np
import scipy as sp
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import gaussian_fitting as gaussfit
import bessel_fitting as bessfit
import misc_functions as misc


def bessel_zeros(order, n_zero):
    """
    These are the positive zeros of the bessel functions. 
    This function returns the nth zero of a bessel function based on the order.
    """

    # Type check.
    order = valid.validate_int_value(order, greater_than=0)
    n_zero = valid.validate_int_value(n_zero, greater_than=0)

    # For some reason, scipy wants to return all zeros from 1 to n. This
    # function only wants the last one.
    zeros = sp_spcl.jn_zeros(order, n_zero)
    return valid.validate_float_value(zeros[-1])


def Ewer_Basu__eigenvalues(index_m, radius):
    """
    This is the values of the eigenvalues of some integer index m as it
    pertains to Ewertiwski & Basu 2013.
    """
    # Type check.
    index_m = valid.validate_int_value(index_m, greater_than=0)
    radius = valid.validate_float_value(radius)

    eigenvalue = (bessel_zeros(1, index_m)/radius)**2

    return valid.validate_float_value(eigenvalue)


def Ewer_Basu__B_r(r, z, h, k_array, disk_radius):
    """
    This implements equation 45 of Ewertiwski & Basu 2013. The k_array (values
    of k) determine the number of summation terms that will be computed.
    """
    # Type check
    r = valid.validate_float_array(r)
    z = valid.validate_float_array(z)
    k_array = valid.validate_float_array(k_array, deep_validate=True)
    disk_radius = valid.validate_float_value(disk_radius, greater_than=0)
    # Shorthand for the squareroot of the eigenvalues. Account for 0 indexing
    def evsq(m): return np.sqrt(Ewer_Basu__eigenvalues(m + 1, disk_radius))
    # Shorthand for bessel function of order 1.
    def bess1(x): return sp_spcl.jv(1, x)

    Bfield_r = 0
    for kdex, k_value in enumerate(k_array):
        # Dividing the equation into smaller chunks for readability.
        coefficient = k_value * evsq(kdex) * bess1(evsq(kdex) * r)
        # Left and right erfc functions of the equation, respectively.
        minus_erfc = sp_spcl.erfc((0.5*evsq(kdex)*h) - z/h)
        plus_erfc = sp_spcl.erfc((0.5*evsq(kdex)*h) + z/h)
        # Exponent values
        neg_exp = np.exp(-evsq(kdex) * z)
        pos_exp = np.exp(evsq(kdex) * z)

        Bfield_r = (Bfield_r 
                    + (coefficient * (minus_erfc * neg_exp 
                                      - plus_erfc * pos_exp)))

    return Bfield_r


def Ewer_Basu__B_z(r, z, h, k_array, disk_radius, uniform_B0):
    """
    This implements equation 46 of Ewertiwski & Basu 2013. The k_array (values
    of k) determine the number of summation terms that will be computed.
    """
    # Type check
    r = valid.validate_float_array(r)
    z = valid.validate_float_array(z)
    k_array = valid.validate_float_array(k_array, deep_validate=True)
    disk_radius = valid.validate_float_value(disk_radius, greater_than=0)
    uniform_B0 = valid.validate_float_value(uniform_B0)

    # Shorthand for the squareroot of the eigenvalues. Account for 0 indexing.
    def evsq(m): return np.sqrt(Ewer_Basu__eigenvalues(m + 1, disk_radius))
    # Shorthand for bessel function of order 1.
    def bess0(x): return sp_spcl.jv(0, x)

    Bfeild_z = 0
    for kdex, k_value in enumerate(k_array):
        # Dividing the equation into smaller chunks for readability.
        coefficient = k_value * evsq(kdex) * bess0(evsq(kdex) * r)
        # Left and right erfc functions of the equation, respectively.
        plus_erfc = sp_spcl.erfc((0.5*evsq(kdex)*h) + z/h)
        minus_erfc = sp_spcl.erfc((0.5*evsq(kdex)*h) - z/h)
        # Exponent values
        pos_exp = np.exp(evsq(kdex) * z)
        neg_exp = np.exp(-evsq(kdex) * z)

        Bfeild_z = (Bfeild_z 
                    + (coefficient * (plus_erfc * pos_exp 
                                      + minus_erfc * neg_exp)))

    return Bfeild_z + uniform_B0


def hourglass_magnetic_field_cyln(r, phi, z,  # Cylindrical cords
                                  h, k_array, disk_radius, uniform_B0):
    """
    This function retruns the magnitude of an hourglass magnetic field in
    cylindrical cords given a location in cylindrical cords.
    """

    # Type check
    r = valid.validate_float_array(r, greater_than=0)
    phi = valid.validate_float_array(phi, deep_validate=True,
                                         greater_than=0, less_than=2*np.pi)
    z = valid.validate_float_array(z)
    h = valid.validate_float_value(h)
    k_array = valid.validate_float_array(k_array, deep_validate=True)
    disk_radius = valid.validate_float_value(disk_radius, greater_than=0)
    uniform_B0 = valid.validate_float_value(uniform_B0)

    # The phi component of the field is agnostic as the values of the magnetic
    # field is independent of phi, see equation 3 and 4 in Ewertiwski & Basu
    # 2013
    B_r = Ewer_Basu__B_r(r, z, h, k_array, disk_radius)
    B_phi = 0
    B_z = Ewer_Basu__B_z(r, z, h, k_array, disk_radius, uniform_B0)

    return B_r, B_phi, B_z


def hourglass_magnetic_field_cart(x, y, z,  # Cartesian cords
                                  h, k_array, disk_radius, uniform_B0):
    """
    This function retruns the magnitude of an hourglass magnetic field in
    cartesian cords given a location in cartesian cords.
    """
    # Type check
    x = valid.validate_float_array(x)
    y = valid.validate_float_array(y)
    z = valid.validate_float_array(z)
    h = valid.validate_float_value(h)
    k_array = valid.validate_float_array(k_array, deep_validate=True)
    disk_radius = valid.validate_float_value(disk_radius, greater_than=0)
    uniform_B0 = valid.validate_float_value(uniform_B0)

    # Convert to cylindrical cords.
    r = np.hypot(x,y)
    phi = np.arctan2(y,x)
    phi[phi < 0] += 2*np.pi
    z = z 

    # Find the values of the magnetic field.
    B_r, B_phi, B_z = \
        hourglass_magnetic_field_cyln(r, phi, z,  
                                      h, k_array, disk_radius, uniform_B0)

    # Convert to cartesian.
    B_x = B_r * np.cos(phi)
    B_y = B_r * np.sin(phi)
    B_z = B_z

    # Return cartesian
    return B_x, B_y, B_z
