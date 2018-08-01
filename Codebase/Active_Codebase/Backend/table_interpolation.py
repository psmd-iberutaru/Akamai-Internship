"""
This file contains an entire list of functions that would interpolate
a table of values into a numerical approximation of a function. Although
it cannot be an analytical function, these functions attempt to return an
function which would mimic such analytical functions by extrapolating tables. 
"""

import numpy as np
import scipy as sp
import scipy.interpolate as sp_inter

import Robustness as Robust


def funt_interpolate_scalar_table(x_array, y_array, z_array, ans_array,
                                  interp_method):
    """Generate functional interpolation of a scalar table.

    This function takes a table of x,y,z points which correspond to some scalar
    answer and attempts to interpolate to generate a function which would 
    allow for the computation of the scalar at any arbitrary x,y,z point. 

    Parallel array representation of the able is assumed.

    Parameters
    ----------
    x_array : array_like
        The x values of the scalar table.
    y_array : array_like
        The y values of the scalar table.
    z_array : array_like
        The z values of the scalar table.
    ans_array : array_like
        The scalar answers of the table.

    Returns
    -------
    interpolated_scalar_function : function
        The numerical approximation to the generalized function.
    """

    # Basic type checking
    x_array = np.ravel(np.array(x_array, dtype=float))
    y_array = np.ravel(np.array(y_array, dtype=float))
    z_array = np.ravel(np.array(z_array, dtype=float))
    ans_array = np.ravel(np.array(ans_array, dtype=float))

    # Ensure that all arrays are the same length.
    if not (x_array.size == y_array.size == z_array.size == ans_array.size):
        raise Robust.ShapeError('The length of each column of the table'
                                'should be the same. Parallel arrays for '
                                'the table\'s values are assumed.'
                                '    --Kyubey')

    # Begin the creation of the function.
    def interpolated_scalar_function(x, y, z):
        # Basic type check.
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        z = np.array(z, dtype=float)

        # Usage of scipy's functionality.
        extrap_ans = sp_inter.griddata((x_array, y_array, z_array), ans_array,
                                       (x, y, z),
                                       method=interp_method, fill_value=0)

        return extrap_ans

    # Return the interpolated function.
    return interpolated_scalar_function


def funt_interpolate_vector_table(x_array, y_array, z_array,
                                  x_ans_array,y_ans_array,z_ans_array,
                                  interp_method):
    """Generate functional interpolation of a scalar table.

    This function takes a table of x,y,z points which correspond to some scalar
    answer and attempts to interpolate to generate a function which would 
    allow for the computation of the scalar at any arbitrary x,y,z point. 

    Parallel array representation of the able is assumed.

    Parameters
    ----------
    x_array : array_like
        The x values of the scalar table.
    y_array : array_like
        The y values of the scalar table.
    z_array : array_like
        The z values of the scalar table.
    x_ans_array : array_like
        The x component of the answer vector.
    y_ans_array : array_like
        The y component of the answer vector.
    z_ans_array : array_like
        The z component of the answer vector.

    Returns
    -------
    interpolated_vector_function : function
        The numerical approximation to the generalized function.
    """

    # Basic type checking
    x_array = np.ravel(np.array(x_array, dtype=float))
    y_array = np.ravel(np.array(y_array, dtype=float))
    z_array = np.ravel(np.array(z_array, dtype=float))
    x_ans_array = np.ravel(np.array(x_ans_array,dtype=float))
    y_ans_array = np.ravel(np.array(y_ans_array,dtype=float))
    z_ans_array = np.ravel(np.array(z_ans_array,dtype=float))

    # Ensure that all arrays are the same length.
    if not (x_array.size == y_array.size == z_array.size ==
            x_ans_array.size == y_ans_array.size == z_ans_array.size):
        raise Robust.ShapeError('The length of each column of the table'
                                'should be the same. Parallel arrays for '
                                'the table\'s values are assumed.'
                                '    --Kyubey')

    # Begin the creation of the function.
    def interpolated_vector_function(x, y, z):
        # Basic type check.
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        z = np.array(z, dtype=float)

        # Usage of scipy's functionality. Split along all three dimensions in
        # rather brute force sort of way.
        extrap_x_ans = sp_inter.griddata((x_array, y_array, z_array),
                                         x_ans_array, (x, y, z),
                                         method=interp_method, fill_value=0)
        extrap_y_ans = sp_inter.griddata((x_array, y_array, z_array),
                                         y_ans_array, (x, y, z),
                                         method=interp_method, fill_value=0)
        extrap_z_ans = sp_inter.griddata((x_array, y_array, z_array),
                                         z_ans_array, (x, y, z),
                                         method=interp_method, fill_value=0)

        extrap_vect_ans = np.array([extrap_x_ans,extrap_y_ans,extrap_z_ans],
                                   dtype=float)

        return extrap_vect_ans

    # Return the interpolated function.
    return interpolated_vector_function
