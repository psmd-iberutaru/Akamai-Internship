import numpy as np
import scipy as sp

from Robustness.exception import *


def validate_boolean_value(boolean_value_input):
    """
    The purpose of this function is to validate that a boolean value 
    input is valid. 
    """

    # Type check. If it is not a boolean value, attempt to change it
    # into one.
    if not isinstance(boolean_value_input, bool):
        try:
            boolean_value_input = bool(boolean_value_input)
        except:
            raise TypeError('Input boolean value is not transformable into a '
                            'boolean value.    --Kyubey')

    return bool(boolean_value_input)


def validate_boolean_array(boolean_array_input,
                           shape=None, size=None,
                           deep_validate=False):
    """
    The purpose of this function is to validate that the boolean 
    array is valid. The shape and size of the array can be optionally 
    tested.

    deep_validate instructs the program to loop over every element 
    array and validate it in turn.
    """
    # Type check. If it is not an array, attempt to change it to one.
    if not isinstance(boolean_array_input, (bool, np.ndarray)):
        try:
            boolean_array_input = np.array(boolean_array_input, dtype=bool)
        except:
            raise TypeError('Input boolean array is not transformable into a '
                            'boolean array.    --Kyubey')
    elif (boolean_array_input.dtype != bool):
        try:
            boolean_array_input = np.array(boolean_array_input, dtype=bool)
        except:
            raise TypeError('Input boolean array is not transformable into a '
                            'boolean array.    --Kyubey')

    # Check the optional conditions of shape and size.
    if (shape is not None):
        # Type check optional condition inputs.
        shape = validate_tuple(shape)
        if (boolean_array_input.shape != shape):
            raise ShapeError('Input boolean array is not the correct shape. '
                             'Expected: {expt}  Actual: {act} '
                             '    --Kyubey'
                             .format(expt=shape, act=boolean_array_input.shape))

    if (size is not None):
        # Type check optional condition inputs.
        size = validate_int_value(size, greater_than=0)
        if (boolean_array_input.size != size):
            raise ShapeError('Input boolean array is not the correct size. '
                             'Expected: {expt}  Actual: {act} '
                             ' --Kyubey'
                             .format(expt=size, act=boolean_array_input.size))

    # Check if the user desired a deep validation check, warn about
    # time. First type check.
    deep_validate = validate_boolean_value(deep_validate)
    if (deep_validate):
        # Warn about time.
        kyubey_warning(TimeWarning, ('Deep validate detected for boolean '
                                     'array validation. This may take longer.'
                                     '    --Kyubey'))

        # Enable value function to loop over all elements of an array.
        vect_validate_boolean_value = np.vectorize(validate_boolean_value,)
        boolean_array_input = vect_validate_boolean_value(boolean_array_input)

    return np.array(boolean_array_input, dtype=bool)


def validate_int_value(int_value_input,
                       non_zero=None, greater_than=None, less_than=None):
    """
    The purpose of this function is to validate that a int value is valid. 
    The value, its range (either greater than or less than a number) may 
    also be tested. This function will bark if the value is greater than 
    less_than or less than greater_than. It can also be tested if it is 
    non-zero (true for non-zero, false for zero passes).
    """
    # Type check. If it is not a int value, attempt to change it into one.
    if not isinstance(int_value_input, int):
        try:
            int_value_input = int(int_value_input)
        except:
            raise TypeError('Input int value is not transformable into a '
                            'int value.    --Kyubey')

    # Test the optional conditions.
    if (non_zero is not None):
        non_zero = validate_boolean_value(non_zero)
        if (non_zero):
            if (int_value_input == 0):
                raise ValueError('Input int value is zero, non_zero flag is '
                                 'set to true.    --Kyubey')
        elif not (non_zero):
            if (int_value_input != 0):
                raise ValueError('Input int value is non-zero, non_zero flag '
                                 'is set to false.    --Kyubey')
    if (greater_than is not None):
        # Type check the optional test inputs.
        greater_than = validate_float_value(greater_than)
        if (int_value_input < greater_than):
            raise ValueError('Input int value is less than the stipulated '
                             'value. '
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=greater_than, act=int_value_input))
    if (less_than is not None):
        # Type check the optional test inputs.
        less_than = validate_float_value(less_than)
        if (int_value_input > less_than):
            raise ValueError('Input int value is greater than the stipulated '
                             'value. '
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=less_than, act=int_value_input))

    return int(int_value_input)


def validate_int_array(int_array_input,
                       shape=None, size=None,
                       deep_validate=False,
                       # Deep/value validate parameters.
                       non_zero=None, greater_than=None, less_than=None):
    """
    The purpose of this function is to validate that the integer array is 
    valid. The shape and size of the array can be optionally tested.

    deep_validate instructs the program to loop over every element array and
    validate it in turn.
    """
    # Type check. If it is not an array, attempt to change it to one.
    if not isinstance(int_array_input, (int, np.ndarray)):
        try:
            int_array_input = np.array(int_array_input, dtype=int)
        except:
            raise TypeError('Input integer array is not transformable into a '
                            'integer array.    --Kyubey')
    elif (int_array_input.dtype != int):
        try:
            int_array_input = np.array(int_array_input, dtype=int)
        except:
            raise TypeError('Input integer array is not transformable into a '
                            'integer array.    --Kyubey')

    # Check the optional conditions of shape and size.
    if (shape is not None):
        # Type check optional condition inputs.
        shape = validate_tuple(shape)
        if (int_array_input.shape != shape):
            raise ShapeError('Input integer array is not the correct shape.'
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=shape, act=int_array_input.shape))

    if (size is not None):
        # Type check optional condition inputs.
        size = validate_int_value(size, greater_than=0)
        if (int_array_input.size != size):
            raise ShapeError('Input integer array is not the correct size. '
                             'Expected: {expt}  Actual: {act}'
                             ' --Kyubey'
                             .format(expt=size, act=int_array_input.size))

    # Check if the user desired a deep validation check, warn about time. First
    # type check.
    deep_validate = validate_boolean_value(deep_validate)
    if (deep_validate):
        # Warn about time.
        kyubey_warning(TimeWarning, ('Deep validate detected for integer '
                                     'array validation. This may take longer.'
                                     '    --Kyubey'))

        # Enable value function to loop over all elements of an array.
        vect_validate_int_value = np.vectorize(validate_int_value)
        int_array_input = vect_validate_int_value(int_array_input,
                                                  non_zero=non_zero,
                                                  greater_than=greater_than,
                                                  less_than=less_than)

    return np.array(int_array_input, dtype=int)


def validate_float_value(float_value_input,
                         greater_than=None, less_than=None):
    """
    The purpose of this function is to validate that a float value is valid. 
    The value, its range (either greater than or less than a number) may 
    also be tested. This function will bark if the value is greater than 
    less_than or less than greater_than.
    """
    # Type check. If it is not a float value, attempt to change it into one.
    if not isinstance(float_value_input, float):
        try:
            float_value_input = float(float_value_input)
        except:
            raise TypeError('Input float value is not transformable into a '
                            'float value.    -Kyubey')

    # Test the optional conditions.
    if (greater_than is not None):
        # Type check the optional test inputs.
        greater_than = validate_float_value(greater_than)
        if (float_value_input < greater_than):
            raise ValueError('Input float value is less than the stipulated '
                             'value. '
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=greater_than, act=float_value_input))
    if (less_than is not None):
        less_than = validate_float_value(less_than)
        if (float_value_input > less_than):
            raise ValueError('Input int value is greater than the stipulated '
                             'value. '
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=less_than, act=float_value_input))

    return float(float_value_input)


def validate_float_array(float_array_input,
                         shape=None, size=None,
                         deep_validate=False,
                         # Deep/value validate parameters.
                         greater_than=None, less_than=None):
    """
    The purpose of this function is to validate that the float array is 
    valid. The shape and size of the array can be optionally tested.

    deep_validate instructs the program to loop over every element array and
    validate it in turn.
    """
    # Type check. If it is not an array, attempt to change it to one.
    if not isinstance(float_array_input, (float, np.ndarray)):
        try:
            float_array_input = np.array(float_array_input, dtype=float)
        except:
            raise TypeError('Input float array is not transformable into a '
                            'float array.    --Kyubey')
    elif (float_array_input.dtype != float):
        try:
            float_array_input = np.array(float_array_input, dtype=float)
        except:
            raise TypeError('Input float array is not transformable into a '
                            'float array.    --Kyubey')

    # Check the optional conditions of shape and size.
    if (shape is not None):
        # Type check optional condition inputs.
        shape = validate_tuple(shape)
        if (float_array_input.shape != shape):
            raise ShapeError('Input float array is not the correct shape. '
                             'Expected: {expt}  Actual: {act}.'
                             '    --Kyubey'
                             .format(expt=shape, act=float_array_input.shape))

    if (size is not None):
        # Type check optional condition inputs.
        size = validate_int_value(size, greater_than=0)
        if (float_array_input.size != size):
            raise ShapeError('Input float array is not the correct size. '
                             'Expected: {expt}  Actual: {act}.'
                             ' --Kyubey'
                             .format(expt=size, act=float_array_input.size))

    # Check if the user desired a deep validation check, warn about time. First
    # type check.
    deep_validate = validate_boolean_value(deep_validate)
    if (deep_validate):
        # Warn about time.
        kyubey_warning(TimeWarning, ('Deep validate detected for float '
                                     'array validation. This may take longer.'
                                     '    --Kyubey'))

        # Enable value function to loop over all elements of an array.
        vect_validate_float_value = np.vectorize(validate_float_value)
        float_array_input = vect_validate_float_value(float_array_input,
                                                      greater_than=None,
                                                      less_than=None)

    return np.array(float_array_input, dtype=float)


def validate_list(input_list,
                  length=None):
    """
    The purpose of this function is to validate that a list is valid.
    """

    # Type check. If it not a list, attempt to change it into one.
    if not isinstance(input_list, list):
        try:
            input_list = list(input_list)
        except:
            raise TypeError('Input list cannot be turned into a list.'
                            '    --Kyubey')

    # Check optional conditions if provided.
    if (length is not None):
        # Type check optional input.
        length = validate_int_value(length, greater_than=0)
        if (len(input_list) != length):
            raise ShapeError('The input list is not the correct length.'
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=length, act=len(input_list)))

    return list(input_list)


def validate_tuple(input_tuple,
                   length=None):
    """
    The purpose of this function is to validate that a tuple is valid.
    """

    # Type check. If it not a tuple, attempt to change it into one.
    if not isinstance(input_tuple, tuple):
        try:
            input_tuple = tuple(input_tuple)
        except:
            raise TypeError('Input tuple cannot be turned into a tuple.'
                            '    --Kyubey')

    # Check optional conditions if provided.
    if (length is not None):
        # Type check optional input.
        length = validate_int_value(length, greater_than=0)
        if (len(input_tuple) != length):
            raise ShapeError('The input tuple is not the correct length.'
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=length, act=len(input_tuple)))

    return tuple(input_tuple)


def validate_string(input_string, length=None, contain_substr=None):
    """
    The purpose of this function is to determine if a string is valid. The 
    length of the string can also be tested. If the string contains a substring 
    can also be tested. Due to oddities of strings, this function may return 
    back false errors.
    """
    # Type check. If not string, attempt to turn it into one.
    if not isinstance(input_string, str):
        try:
            input_string = str(input_string)
        except:
            raise TypeError('Input string cannot be turned into a string.'
                            '    --Kyubey')

    # Check for optional conditions.
    if (length is not None):
        # Type check.
        length = validate_int_value(length, greater_than=0)
        if (len(input_string) != length):
            raise ShapeError('Input string is not the correct length.'
                             'Expected: {expt}  Actual: {act}'
                             '    --Kyubey'
                             .format(expt=length, act=len(input_string)))
    if (contain_substr is not None):
        contain_substr = validate_string(contain_substr)
        if not (contain_substr in input_string):
            raise ValueError('Input string does not contain given substring.'
                             '    --Kyubey')

    return str(input_string)
