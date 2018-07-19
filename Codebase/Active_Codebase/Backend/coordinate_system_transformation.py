"""
This file is some side functions that allows for the conversion of cordinate
systems.
"""

import numpy as np

########################################################################
##########                From Cartesian to ***               ##########
########################################################################


def cartesian_to_polar_2d(x, y):
    """Convert cartesian points to polar points.

    Convert cartesian coordinate points in 2D to polar coordinate points in 2D.
    This function uses the notation convention of ISO 80000-2:2009 and its 
    related successors.

    Parameters:
    -----------
    x : array_like
        The x values of the points to be transformed.
    y : array_like
        The y values of the points to be transformed.

    Returns:
    --------
    rho : array_like
        The rho (radial) values of the points after transformation.
    phi : array_like
        The phi (angular) values of the points after transformation.    
    """

    # Basic validation
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Convert to polar coordinates.
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    return rho, phi


def cartesian_to_cylindrical_3d(x, y, z):
    """Convert cartesian points to cylindrical points.

    Convert cartesian coordinate points in 3D to cylindrical coordinate points 
    in 3D. This function uses the notation convention of ISO 80000-2:2009 and 
    its related successors.

    Parameters:
    -----------
    x : array_like
        The x values of the points to be transformed.
    y : array_like
        The y values of the points to be transformed.
    z : array_like
        The z values of the points to be transformed.

    Returns:
    --------
    rho : array_like
        The rho (radial) values of the points after transformation.
    phi : array_like
        The phi (angular) values of the points after transformation.    
    z : array_like
        The z (height) values of the points after transformation.
    """

    # Basic validation.
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    # Convert to cylindrical coordinates.
    rho = np.hypot(x, y)
    phi = np.arctan2(y, z)
    z = z

    return rho, phi, z


def cartesian_to_spherical_3d(x, y, z):
    """Convert cartesian points to cylindrical points.

    Convert cartesian coordinate points in 3D to cylindrical coordinate points 
    in 3D. This function uses the notation convention of ISO 80000-2:2009 and 
    its related successors.

    Parameters:
    -----------
    x : array_like
        The x values of the points to be transformed.
    y : array_like
        The y values of the points to be transformed.
    z : array_like
        The z values of the points to be transformed.

    Returns:
    --------
    r : array_like
        The rho (radial) values of the points after transformation.
    theta : array_like
        The theta (azimuthal angle) values of the points after the 
        transformation.    
    phi : array_like
        The phi (polar angle) values of the points after the transformation.
    """

    # Basic validation.
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    # Convert to spherical coordinates.
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    return r, theta, phi


########################################################################
##########                  From Polar to ***                 ##########
########################################################################

def polar_to_cartesian_2d(rho, phi):
    """Convert polar points to cartesian points.

    Convert polar coordinate points in 2D to cartesian coordinate points in 2D.
    This function uses the notation convention of ISO 80000-2:2009 and its 
    related successors.

    Parameters:
    -----------
    rho : array_like
        The rho values of the points to be transformed.
    phi : array_like
        The phi values of the points to be transformed.

    Returns:
    --------
    x : array_like
        The x values of the points after transformation.
    y : array_like
        The y values of the points after transformation.
    """

    # Basic type checking
    rho = np.array(rho, dtype=float)
    phi = np.array(phi, dtype=float)

    # Convert
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


########################################################################
##########               From Cylindrical to ***              ##########
########################################################################

def cylindrical_to_cartesian_3d(rho, phi, z):
    """Convert cylindrical points to cartesian points.

    Convert cylindrical coordinate points in 3D to cartesian coordinate points 
    in 3D. This function uses the notation convention of ISO 80000-2:2009 and  
    its related successors.

    Parameters:
    -----------
    rho : array_like
        The rho values of the points to be transformed.
    phi : array_like
        The phi values of the points to be transformed.
    z : array_like
        The z values of the points to be transformed.

    Returns:
    --------
    x : array_like
        The x values of the points after transformation.
    y : array_like
        The y values of the points after transformation.
    z : array_like
        The z values of the points after transformation.
    """

    # Basic type checking
    rho = np.array(rho, dtype=float)
    phi = np.array(phi, dtype=float)
    z = np.array(z, dtype=float)

    # Convert
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = z

    return x, y, z


def cylindrical_to_spherical_3d(rho, phi, z):
    """Convert cylindrical points to spherical points.

    Convert cylindrical coordinate points in 3D to spherical coordinate points 
    in 3D. This function uses the notation convention of ISO 80000-2:2009 and  
    its related successors.

    Parameters:
    -----------
    rho : array_like
        The rho values of the points to be transformed.
    phi : array_like
        The phi values of the points to be transformed.
    z : array_like
        The z values of the points to be transformed.

    Returns:
    --------
    r : array_like
        The rho (radial) values of the points after transformation.
    theta : array_like
        The theta (azimuthal angle) values of the points after the 
        transformation.    
    phi : array_like
        The phi (polar angle) values of the points after the transformation.
    """

    # Basic type checking
    rho = np.array(rho, dtype=float)
    phi = np.array(phi, dtype=float)
    z = np.array(z, dtype=float)

    # Convert
    r = np.hypot(rho, z)
    theta = np.arccos(z/r)
    phi = phi

    return r, theta, phi


########################################################################
##########                From Spherical to ***               ##########
########################################################################

def spherical_to_cartesian_3d(r, theta, phi):
    """Convert spherical points to cartesian points.

    Convert spherical coordinate points in 3D to cartesian coordinate points 
    in 3D. This function uses the notation convention of ISO 80000-2:2009 and  
    its related successors.

    Parameters:
    -----------
    r : array_like
        The r values of the points to be transformed.
    theta : array_like
        The theta values of the points to be transformed.
    phi : array_like
        The phi values of the points to be transformed.

    Returns:
    --------
    x : array_like
        The x values of the points after transformation.
    y : array_like
        The y values of the points after transformation.
    z : array_like
        The z values of the points after transformation.
    """

    # Basic type checking
    r = np.array(r, dtype=float)
    theta = np.array(theta, dtype=float)
    phi = np.array(phi, dtype=float)

    # Convert
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def spherical_to_cylindrical_3d(r, theta, phi):
    """Convert cylindrical points to cartesian points.

    Convert cylindrical coordinate points in 3D to cartesian coordinate points 
    in 3D. This function uses the notation convention of ISO 80000-2:2009 and  
    its related successors.

    Parameters:
    -----------
    r : array_like
        The r values of the points to be transformed.
    theta : array_like
        The theta values of the points to be transformed.
    phi : array_like
        The phi values of the points to be transformed.

    Returns:
    --------
    rho : array_like
        The rho values of the points after transformation.
    phi : array_like
        The phi (angular) values of the points after transformation.
    z : array_like
        The z values of the points after transformation.
    """

    # Basic type checking
    r = np.array(r, dtype=float)
    theta = np.array(theta, dtype=float)
    phi = np.array(phi, dtype=float)

    # Convert
    rho = r * np.sin(theta)
    phi = phi
    z = r * np.cos(theta)

    return rho, phi, z
