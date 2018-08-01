"""
This document highlights 2d versions of magnetic field functions. Thus, these
functions take in 2 parameters, the x,y values or rho,phi values, and return 
the u,v vector components of the given magnetic field

"""

import numpy as np
import scipy as sp
import scipy.special as sp_spcl

import Robustness as Robust
import Backend as _Backend
import magnetic_field_functions_2d as mff2d


# In order from least complex field to most complex field, first in cartesian
# cords.

########################################################################
#####             3D Magnetic Field Cartesian Functions            #####
########################################################################

def circular_magnetic_field_cart_3d(x, y, z,
                                    center=[0, 0, 0],
                                    mag_function=lambda r: 1/r**2,
                                    reference_axis='z'):
    """Compute the cartesian magnetic field vectors of a circular field.

    The circular magnetic field is radially symmetric. This function  
    returns the values of the components of a magnetic field given some 
    point(s) x,y,z. The center of the circular magnetic field can be 
    redefined. The axis of symmetry can also be redefined.

    Parameters
    ----------
    x : array_like
        The x values of the points for magnetic field computation.
    y : array_like
        The y values of the points for the magnetic field computation
    z : array_like
        The z values of the points for magnetic field computation.
    center : array_like; optional
        The center of the circular magnetic field function, passed as an
        array ``[x0,y0]``. Default is ``[0,0,0]``
    mag_function : function ``f(r)``; optional
        The value of the magnitude of the vector field at some radius from
        the center. Default is ``f(r) = 1/r**2``.
    reference_axis : string; optional
        The specified axis that the magnetic field curls around. Default is 
        the ``z`` axis.

    Returns
    -------
    Bfield_x : ndarray
        The x component of the magnetic field at the given points. Order
        is perserved.
    Bfield_y : ndarray
        The y component of the magnetic field at the given points. Order
        is perserved.
    Bfield_z : ndarray
        The z component of the magnetic field at the given points. Order
        is perserved.
    """

    # Type check
    x = Robust.valid.validate_float_array(x)
    y = Robust.valid.validate_float_array(y)
    z = Robust.valid.validate_float_array(z)
    center = Robust.valid.validate_float_array(center, shape=(3,))
    mag_function = Robust.valid.validate_function_call(mag_function,
                                                       n_parameters=3)
    reference_axis = Robust.valid.validate_string(
        reference_axis.lower(), length=1, contain_substr=('x', 'y', 'z'))

    # Do a transformation based on the relocation of the center.
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]

    # Calculate the magnetic field based on what the user desired on for
    # the axis of symmetry.
    if (reference_axis == 'x'):
        axis1, axis2 = mff2d.circular_magnetic_field_cart_2d(
            y, z, mag_function=mag_function)
        # The x axis is determined to have zero contribution.
        Bfield_x = 0
        Bfield_y = axis1
        Bfield_z = axis2
    elif (reference_axis == 'y'):
        axis1, axis2 = mff2d.circular_magnetic_field_cart_2d(
            -x, z, mag_function=mag_function)
        # The x axis is determined to have zero contribution. The negatives
        # keep the orientation consistant.
        Bfield_x = axis1 * -1
        Bfield_y = 0
        Bfield_z = axis2
    elif (reference_axis == 'z'):
        axis1, axis2 = mff2d.circular_magnetic_field_cart_2d(
            x, y, mag_function=mag_function)
        # The z axis is determined to have zero contribution.
        Bfield_x = axis1
        Bfield_y = axis2
        Bfield_z = 0
    else:
        raise Robust.InputError('The reference axis specified is not a valid '
                                'axis in the cartesian cordinate system. '
                                'It is not known how this input got past '
                                'input validation. '
                                '    --Kyubey')

    # Return the values of the magnetic field.
    return Bfield_x, Bfield_y, Bfield_z


def hourglass_magnetic_field_cart_3d(x, y, z,
                                     h, k_array, disk_radius, uniform_B0,
                                     center=[0, 0, 0]):
    """Equation for hourglass magnetic fields given by Ewertowshi & Basu 2013.

    This function is the three dimensional version of the equations given by 
    Ewertowshi & Basu 2013. This function assumes, as the paper does, that the
    magnetic field is invariant with respect to phi.

    Parameters
    ----------
    x : array_like
        The input values of the x direction for the equation.
    y : array_like
        The input values of the y direction for the equation. 
    z : array_like
        The input values of the z direction for the equation.
    h : float
        A free parameter as dictated by the paper.
    k_array : array_like
        The list of k coefficient values for the summation in Eq 45.
    disk_radius : float
        The radius of the protoplanetary disk. Relevent for the hourglass
        magnetic field generated by this paper.
    uniform_B0 : float
        The magnitude of the background magnetic field.
    center : array_like; optional
        The center of the hourglass shaped magnetic field function, passed
        as an array ``[r0,phi0,z0]``. Defaults to ``[0,0,0]``

    Returns
    -------
    Bfield_x : ndarray
        The value of the magnetic field in the x-axial direction.
    Bfield_y : ndarray
        The value of the magnetic field in the y-axial direction.
    Bfield_z : ndarray
        The value of the magnetic field in the z-axial direction.
    """

    # Type check
    x = Robust.valid.validate_float_array(x)
    y = Robust.valid.validate_float_array(y)
    z = Robust.valid.validate_float_array(z)
    h = Robust.valid.validate_float_value(h)
    k_array = Robust.valid.validate_float_array(k_array)
    disk_radius = Robust.valid.validate_float_value(
        disk_radius, greater_than=0)
    uniform_B0 = Robust.valid.validate_float_value(uniform_B0)
    center = Robust.valid.validate_float_array(center, shape=(3,))

    # Convert the cartesian cords to cylindrical cords.
    rho, phi, z = _Backend.cst.cartesian_to_cylindrical_3d(x, y, z)

    # Compute the magnetic fields.
    Bfield_rho, Bfield_phi, Bfield_z = hourglass_magnetic_field_cyln_3d(
        rho, phi, z, h, k_array, disk_radius, uniform_B0, center=center)

    # Convert back to the cartesian cords.
    Bfield_x, Bfield_y, Bfield_z = \
        _Backend.cst.cylindrical_to_cartesian_3d(Bfield_rho, phi, Bfield_z)

    return Bfield_x, Bfield_y, Bfield_z


########################################################################
#####            3D Magnetic Field Cylindrical Functions           #####
########################################################################

def hourglass_magnetic_field_cyln_3d(rho, phi, z,
                                     h, k_array, disk_radius, uniform_B0,
                                     center=[0, 0, 0]):
    """Equation for hourglass magnetic fields given by Ewertowshi & Basu 2013.

    This function is the three dimensional version of the equations given by 
    Ewertowshi & Basu 2013. This function assumes, as the paper does, that the
    magnetic field is invariant with respect to phi.

    Parameters
    ----------
    rho : array_like
        The input values of the radial direction for the equation.
    phi : array_like
        The input values of the polar angle for the equation. 
    z : array_like
        The input values of the polar direction for the equation.
    h : float
        A free parameter as dictated by the paper.
    k_array : array_like
        The list of k coefficient values for the summation in Eq 45.
    disk_radius : float
        The radius of the protoplanetary disk. Relevent for the hourglass
        magnetic field generated by this paper.
    uniform_B0 : float
        The magnitude of the background magnetic field.
    center : array_like
        The center of the hourglass shaped magnetic field function, passed
        as an array ``[r0,phi0,z0]``. Defaults to ``[0,0,0]``

    Returns
    -------
    Bfield_rho : ndarray
        The value of the magnetic field in the rho-axial direction.
    Bfield_phi : ndarray
        The value of the magnetic field in the phi-axial direction.
    Bfield_z : ndarray
        The value of the magnetic field in the z-axial direction.
    """

    # Type check the parameters.
    rho = Robust.valid.validate_float_array(rho)
    phi = Robust.valid.validate_float_array(phi)
    z = Robust.valid.validate_float_array(z)
    center = Robust.valid.validate_float_array(center, shape=(3,))
    h = Robust.valid.validate_float_value(h)
    k_array = Robust.valid.validate_float_array(k_array)
    disk_radius = Robust.valid.validate_float_value(disk_radius,
                                                    greater_than=0)
    uniform_B0 = Robust.valid.validate_float_value(uniform_B0)

    # Do a translation based off of the center.
    rho = rho - center[0]
    phi = phi - center[1]
    z = z - center[2]

    # Compute the magnetic fields from Ewertowski and Basu 2013 equations.
    Bfield_rho = _Backend.Ewertowski_Basu_2013.Ewer_Basu__B_r(
        rho, z, h, k_array, disk_radius)
    Bfield_phi = 0
    Bfield_z = _Backend.Ewertowski_Basu_2013.Ewer_Basu__B_z(
        rho, z, h, k_array, disk_radius, uniform_B0)

    return Bfield_rho, Bfield_phi, Bfield_z
