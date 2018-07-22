"""
This file deals with the functions regarding polarization and the 
relationship between the electric and magnetic fields.
"""

import numpy as np

import Robustness as Robust

def electric_to_magnetic(E_i, E_j):
    """Convert a 2D electric field vector set to a magnetic field set.

    This function takes the electric field that would normally be seen
    in a polarization ellipse and converts it to the magnetic field
    vectors. This function returns a perpendicular vector of the 
    magnetic field, perserving the magnitude.

    In two dimensions, there are always two vectors perpendicular to a vector.
    Or one vector with positive and negative magnitude. In three, there are
    infinitely many, so it is much harder to give a good vector.

    Parameters:
    E_i : float or array_like
        The component of the electric field in the i-hat direction.
    E_j : float or array_like
        The component of the electric field in the j-hat direction.

    Returns:
    B_i : ndarray
        The component of the magnetic field in the i-hat direction.
    B_j : ndarray
        The component of the magnetic field in the j-hat direction.
    """

    # Basic type check.
    E_i = np.array(E_i, dtype=float)
    E_j = np.array(E_j, dtype=float)

    # Allow for the decision of which vector to use based on hardcoded
    # values.
    if (True):
        # Use the vector: \vec(u) = -b \i + a \j
        B_i = E_j * -1
        B_j = E_i
        return B_i, B_j
    else:
        # Use the vector: \vec(u) = b \i - a \j
        B_i = E_j
        B_j = E_i * -1
        return B_i, B_j


def magnetic_to_electric(B_i, B_j):
    """Convert a 2D magnetic field vector set to a electric field set.

    This function takes the magnetic field that would normally be seen
    in a polarization ellipse and converts it to the electric field
    vectors. This function returns a perpendicular vector of the 
    magnetic field, perserving the magnitude.

    In two dimensions, there are always two vectors perpendicular to a vector.
    Or one vector with positive and negative magnitude. In three, there are
    infinitely many, so it is much harder to give a good vector.

    Parameters:
    B_i : float or array_like
        The component of the magnetic field in the i-hat direction.
    B_j : float or array_like
        The component of the magnetic field in the j-hat direction.

    Returns:
    E_i : ndarray
        The component of the electric field in the i-hat direction.
    E_j : ndarray
        The component of the electric field in the j-hat direction.
    """
    # Basic type check.
    B_i = np.array(B_i, dtype=float)
    B_j = np.array(B_j, dtype=float)

    # Allow for the decision of which vector to use based on hardcoded
    # values.
    if (True):
        # Use the vector: \vec(u) = -b \i + a \j
        E_i = B_j * -1
        E_j = B_i
        return E_i, E_j
    else:
        # Use the vector: \vec(u) = b \i - a \j
        E_i = B_j
        E_j = B_i * -1
        return E_i, E_j


def Stokes_parameters_from_field(E_i,E_j,
                                 percent_polarized=1,chi=0):
    """Returns Stokes parameters based off non-circularly polarized light.

    This function returns the Stokes parameters based off a given 
    electric field vector. 

    Technically it can handle circularly polarized light, the value that
    must be given is chi, the angle between the semi-major axis of the 
    polarization ellipse and the line segment connecting between two points 
    on the ellipse and the semi-major and semi-minor axes. See note [1].

    Parameters:
    E_i : float or array_like
        The component of the electric field in the i-hat direction.
    E_j : float or array_like
        The component of the electric field in the j-hat direction.
    percent_polarized : float
        The percent of the EM wave that is polarized. It should not be too
        far off the value of 1.
    chi : float or array_like
        The parameter for circularly polarized light.

    Returns:
    I : ndarray
        The first Stokes parameter, equivalent to S_0. The intensity of the 
        light.
    Q : ndarray
        The second Stokes parameter, equivalent to S_1. The x,y polarization
        aspect.
    U : ndarray
        The third Stokes parameter, equivalent to S_2. The a,b (45 deg off set 
        of x,y) polarization aspect.
    V : ndarray
        The fourth Stokes parameter, equivalent to S_3. The circular 
        polarization aspect.

    Notes:
    ------
    [1]  This function's notation is based on the following diagram. 
    https://en.wikipedia.org/wiki/File:Polarisation_ellipse2.svg
    """

    # Basic type checking.
    E_i = np.array(E_i,dtype=float)
    E_j = np.array(E_j,dtype=float)
    percent_polarized = float(percent_polarized)
    chi = np.array(chi)

    if ((E_i.size != E_j.size) or (E_i.shape != E_j.shape)):
        raise Robust.ShapeError('The shapes and lengths of E_i and E_j '
                                'should be equal.'
                                '    --Kyubey')

    if ((chi.size != 1) and (chi.size != E_i.size)):
        raise Robust.ShapeError('The array chi must be broadcastable to E_i'
                                'and E_j. Thus, it must have a size of 1 or '
                                'equal to that of E_i and E_j.'
                                '    --Kyubey')

    # Find the overall intensity of the EM wave.
    intensity = np.hypot(E_i,E_j)
    p = percent_polarized

    # Find the angle between the semi-major axis and the x-axis.
    psi = np.arctan2(E_j,E_i)

    # Plug into Wikipedia equations and solve.
    I = intensity
    Q = I * p * np.cos(2*psi) * np.cos(2*chi)
    U = I * p * np.sin(2*psi) * np.cos(2*chi)
    V = I * p * np.sin(2*chi)

    return I,Q,U,V
        
