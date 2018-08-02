"""
This file contains the methods needed to convert from a non-standard or 
accepted data type into a more usable datatype by this module.
"""

import numpy as np

import Robustness as Robust
import Backend as _Backend

class InterpolationTable():
    """A class representing either a scalar or vector table.

    If a lookup table is to be used instead of a function for the model
    observing method, it is required to standardize the information given
    by the user's lookup table, as is the purpose of this class.    

    Arguments
    ---------
    x_values : array_like
        The values of the x points for the table. Array must be parallel
        with y_values and z_values along with the scalar/vector answer.
    y_values : array_like
        The values of the x points for the table. Array must be parallel 
        with x_values and z_values along with the scalar/vector answer.
    z_values : array_like
        The values of the x points for the table. Array must be parallel 
        with x_values and y_values along with the scalar/vector answer.
    classification : string
        The classification of this table, either as a scalar lookup table 
        or a vector lookup table. Should be one of:

            - 'scalar'   A scalar based lookup table.
            - 'vector'   A vector based lookup table.

    scalar_ans : array_like, {for | classification == 'scalar'}
        The scalar answers to the (x,y,z) point given by the input values. 
        Must be parallel with x_values, y_values, and z_values. Ignored if 
        classification == 'vector'.
    x_vector_ans : array_like, {for | classification == 'vector'}
        The x component of the answer vector that exists at the point 
        (x,y,z) given by the input values. Must be parallel with x_values, 
        y_values, and z_values along with other components. Ignored if 
        classification == 'scalar'
    y_vector_ans : array_like, {for | classification == 'vector'}
        The y component of the answer vector that exists at the point 
        (x,y,z) given by the input values. Must be parallel with x_values, 
        y_values, and z_values along with other components. Ignored if 
        classification == 'scalar'
    z_vector_ans : array_like, {for | classification == 'vector'}
        The z component of the answer vector that exists at the point 
        (x,y,z) given by the input values. Must be parallel with x_values, 
        y_values, and z_values along with other components. Ignored if 
        classification == 'scalar'

    Methods
    -------
    numerical_function() : function {returns | function}
        Returns a function which is an interface to a numerical approximation
        interpolation of the data points given in the lookup table.
        Automatically detects if it is a scalar function or vector function.
        
    """
    def __init__(self,x_values,y_values,z_values,classification,
                 scalar_ans=None,
                 x_vector_ans=None,y_vector_ans=None,z_vector_ans=None):
        """A class representing either a scalar or vector table.

        If a lookup table is to be used instead of a function for the model
        observing method, it is required to standardize the information given
        by the user's lookup table, as is the purpose of this class.

        Arguments
        ---------
        x_values : array_like
            The values of the x points for the table. Array must be parallel
            with y_values and z_values along with the scalar/vector answer.
        y_values : array_like
            The values of the x points for the table. Array must be parallel 
            with x_values and z_values along with the scalar/vector answer.
        z_values : array_like
            The values of the x points for the table. Array must be parallel 
            with x_values and y_values along with the scalar/vector answer.
        classification : string
            The classification of this table, either as a scalar lookup table 
            or a vector lookup table. Should be one of:

                - 'scalar'   A scalar based lookup table.
                - 'vector'   A vector based lookup table.

        scalar_ans : array_like, {for | classification == 'scalar'}
            The scalar answers to the (x,y,z) point given by the input values. 
            Must be parallel with x_values, y_values, and z_values. Ignored if 
            classification == 'vector'.
        x_vector_ans : array_like, {for | classification == 'vector'}
            The x component of the answer vector that exists at the point 
            (x,y,z) given by the input values. Must be parallel with x_values, 
            y_values, and z_values along with other components. Ignored if 
            classification == 'scalar'
        y_vector_ans : array_like, {for | classification == 'vector'}
            The y component of the answer vector that exists at the point 
            (x,y,z) given by the input values. Must be parallel with x_values, 
            y_values, and z_values along with other components. Ignored if 
            classification == 'scalar'
        z_vector_ans : array_like, {for | classification == 'vector'}
            The z component of the answer vector that exists at the point 
            (x,y,z) given by the input values. Must be parallel with x_values, 
            y_values, and z_values along with other components. Ignored if 
            classification == 'scalar'
        """

        # Type check.
        x_values = Robust.valid.validate_float_array(x_values)
        y_values = Robust.valid.validate_float_array(y_values)
        z_values = Robust.valid.validate_float_array(z_values)
        classification = Robust.valid.validate_string(classification).lower()
        # Decide on the type before type checking.
        if (classification == 'scalar'):
            if (scalar_ans is not None):
                scalar_ans = Robust.valid.validate_float_array(scalar_ans)
            else:
                raise TypeError('Scalar answer array must be provided if '
                                'table classification is set to scalar.'
                                '    --Kyubey')
        elif (classification == 'vector'):
            if (x_vector_ans is not None):
                x_vector_ans = Robust.valid.validate_float_array(x_vector_ans)
            else:
                raise TypeError('The x component of the vector answer array '
                                'must be provided if table classification is '
                                'set to vector.'
                                '    --Kyubey')
            if (y_vector_ans is not None):
                y_vector_ans = Robust.valid.validate_float_array(y_vector_ans)
            else:
                raise TypeError('The y component of the vector answer array '
                                'must be provided if table classification is '
                                'set to vector.'
                                '    --Kyubey')
            if (z_vector_ans is not None):
                z_vector_ans = Robust.valid.validate_float_array(z_vector_ans)
            else:
                raise TypeError('The z component of the vector answer array '
                                'must be provided if table classification is '
                                'set to vector.'
                                '    --Kyubey')
        else:
            raise Robust.InputError('Table classification must be one of the '
                                    'following: \n'
                                    'scalar, vector \n'
                                    'It is currently: < {table_cls} >'
                                    '    --Kyubey'
                                    .format(table_cls=classification))

        # Assign variables. Depending on the actual 
        self.x_values = x_values
        self.y_values = y_values
        self.z_values = z_values
        self.classification = classification
        self.scalar_ans = scalar_ans
        self.x_vector_ans = x_vector_ans
        self.y_vector_ans = y_vector_ans
        self.z_vector_ans = z_vector_ans

    
    def numerical_function(self,interp_method='linear'):
        """Generate a numerical function from the lookup table.

        This function creates a functional interface of the data from a lookup
        table. It interpolates values that are not in the table to return 
        what Scipy thinks is the best value.

        Parameters
        ----------
        interp_method : string, optional
            The method of interpolation to be used. Must be one of the 
            following strings:

                - 'nearest'
                - 'linear', default
                - 'cubic'
        
        Returns
        -------
        numeric_function : function
            The numerical interpolation function that attempts to best
            replicate the table.
        """
        # Detect if the table provided is a scalar or vector based lookup
        # table. Return a numerical function implementation based on such.
        numeric_function = None
        if (self.classification == 'scalar'):
            numeric_function = \
                _Backend.tbint.funt_interpolate_scalar_table(
                    self.x_values,self.y_values,self.z_values,
                    self.scalar_ans,
                    interp_method)
        elif (self.classification == 'vector'):
            numeric_function = \
                _Backend.tbint.funt_interpolate_vector_table(
                    self.x_values,self.y_values,self.z_values,
                    self.x_vector_ans,self.y_vector_ans,self.z_vector_ans,
                    interp_method)
        else:
            raise Robust.InputError('Table classification must be one of the '
                                    'following: \n'
                                    'scalar, vector \n'
                                    'It is currently: < {table_cls} >'
                                    '    --Kyubey'
                                    .format(table_cls=self.classification))
        
        if (numeric_function is None):
            raise RuntimeError('The creation of a numerical function for '
                               'this given table did not complete correctly. '
                               '    --Kyubey')
        else:
            return numeric_function