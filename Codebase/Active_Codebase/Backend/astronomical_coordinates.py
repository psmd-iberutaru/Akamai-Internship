"""
This file is mostly the calculations required for corrections to the 
astrophysical coordinate system, and any calculations required for such is 
recorded here.
"""

import numpy as np

import Robustness as Robust

def angle_normalization_0_2pi(angle):
    """Automatically normalize angle value(s) to the range of 0-2pi.

    This function relies on modular arithmetic.

    Parameters
    ----------
    angle : array_like
        The angles to be converted

    Returns
    -------
    normalized_angles : ndarray
        The angles after being normalized to be between 0-2pi.
    """

    # Type check.
    angle = Robust.valid.validate_float_array(angle)

    # Attempt to correct all of the angle values. Do an explicit loop in
    # the event of only a single number.
    angle = angle % (2*np.pi)
    normalized_angles = np.where(angle <= 0, angle + 2*np.pi,angle)

    # Vectorize just in case.
    normalized_angles = np.array(normalized_angles,dtype=float)

    return normalized_angles

def auto_ra_wrap_angle(object_ra_value):
    """Automatically calculate the RA wrap value.

    Given an input RA, this function automatically calculates the RA wrap
    angle to likely be used for an Astropy :py:class:`~.SkyCoord` object. 
    
    Parameters
    ----------
    object_ra_value : array_like
        The RA value(s) that will determine the wrap angle(s).

    Returns
    -------
    ra_wrap_angle : ndarray
        The value of the wrap angle that the function best described.

    Notes
    -----
    In this function, it assumes that there are only 4 main quadrants. First,
    the wrap angle is determined to be either at :math:`0` or :math:`\pi` depending on the location of the sightline's RA.
    """

    # Type check.
    object_ra_value = Robust.valid.validate_float_array(object_ra_value)
    object_ra_value = angle_normalization_0_2pi(object_ra_value)

    # Test for the ranges. If it is between pi/2 and 3pi/2, then the wrap 
    # around zero is fine. 
    ra_wrap_angle = np.where(0.5*np.pi <=object_ra_value <= 1.5*np.pi,
                             np.full_like(object_ra_value,2*np.pi),
                             np.full_like(object_ra_value,np.pi))

    return ra_wrap_angle

    