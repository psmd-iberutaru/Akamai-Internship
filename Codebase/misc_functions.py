import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import scipy.signal as sp_sig
import sympy as sy
import matplotlib.pyplot as plt
import copy
import inspect

from Robustness.exception import *
import Robustness.validation as valid


def merge_two_dicts(x, y):
    z = copy.deepcopy(x)   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return copy.deepcopy(z)


def generate_noise(input_array, noise_domain, distribution='uniform',
                   center=None, std_dev=None,  # Normal distribution terms.
                   debug=False):
    """
    Takes a set of 'perfect' datapoints and scatters them based on some 
    randomly generated noise. The generated noise can be distributed in a number
    of ways.

    Input:
    input_array = array of datapoints to be scattered from the original value
    noise_domain = (2,) shaped array of the upper and lower bounds of scattering
    distribution = method of random number distribution
                    - 'uniform'
                    - 'gaussian'
    debug = Debug mode

    Output:
    output_array = scattered 'noisy' datapoints
    """
    # Type check.
    input_array = valid.validate_float_array(
        input_array, size=len(input_array))
    noise_domain = valid.validate_float_array(noise_domain, shape=(2,), size=2)
    distribution = valid.validate_string(distribution)

    # Initial conditions
    n_datapoints = len(input_array)

    # Ensure the lower bound of the noise domain is the first element.
    if (noise_domain[0] < noise_domain[-1]):
        # This is correct behavior.
        pass
    elif (noise_domain[0] > noise_domain[-1]):
        # Warn and change, the array seems to be reversed.
        noise_domain = np.flip(noise_domain, axis=0)
    elif (noise_domain[0] == noise_domain[-1]):
        raise ValueError('Noise domain range is detected to be zero. There is '
                         'no functional use of this function.    --Kyubey')

    # Check for distribution method, generate noise array from method.
    if (distribution == 'uniform'):
        if (debug):
            print('Noise distribution set to "uniform".')
        noise_array = np.random.uniform(noise_domain[0], noise_domain[1],
                                        size=n_datapoints)
    elif ((distribution == 'gaussian') or (distribution == 'normal')):
        if (debug):
            print('Noise distribution set to "gaussian".')
            kyubey_warning(OutputWarning, ('Noise domain is ignored under '
                                           'gaussian distribution.    --Kyubey'))
        # Type check center and standard deviation.
        if (std_dev is None):
            raise InputError('Noise distribution is set to gaussian, there is '
                             'no standard deviation input.')
        else:
            # Standard deviation cannot be negative
            center = valid.validate_float_value(center)
            std_dev = valid.validate_float_value(std_dev, greater_than=0)
            noise_array = np.random.normal(center, std_dev, size=n_datapoints)

    # Noise array plus standard values.
    return input_array + noise_array


def generate_function_envelope(x_values, functions, parameters):
    """
    Generate a function (x,y points) based on the maximum value of a list of
    functions, given their parameters. This creates an envelope function around
    the list of functions.

    Input:
    x_values = x input values
    functions = list of functions to be used, the first entry of each function
        is assumed to be the main input value of the function.
    parameters = list of tuples or dictionaries of the parameters to be used, 
        parallel array to functions, it must lineup with the function 
        definition or be a dictionary of inputs.

    Output:
    y_values = y output values
    """
    # Initial values.
    y_values = []

    # Type check, the only initial type checking that can be done is for
    # x_values.
    x_values = valid.validate_float_array(x_values)
    parameters = list(parameters)

    # Number of functions.
    total_functions = len(functions)
    # Test if the parameters array is parallel.
    if (len(parameters) != total_functions):
        raise InputError('The number of parameter lists is not equal to '
                         'the total number of functions. '
                         'Expected: {expt}  Actual: {act} '
                         '    --Kyubey'
                         .format(expt=total_functions, act=len(parameters)))

    # Obtain values of each y_output per function.
    for functiondex in range(total_functions):
        # Attempt to get a function signature.
        try:
            function_signature = inspect.signature(functions[functiondex])
        except Exception:
            raise InputError('Cannot get function signature from function '
                             'number {funt_num}. Ensure the input is correct.'
                             '    --Kyubey'
                             .format(funt_num=functiondex + 1))

        # Obtain number of arguments and paramerers for this function. Assume
        # the first is the main input.
        n_arguments = len(function_signature.parameters) - 1
        n_parameters = len(parameters[functiondex])

        # Check that the current list of parameters is of correct size.
        if (n_parameters != n_arguments):
            raise InputError('Not enough parameters for function {funt_num}.'
                             'Expected: {expt}  Actual: {act}.'
                             '    --Kyubey'
                             .format(expt=n_arguments, act=n_parameters))

        # Check if the user provided a dictionary or parallel tuples assume
        # that it is.
        is_dictionary_parameters = True
        if isinstance(parameters[functiondex], dict):
            is_dictionary_parameters = True

            # Get the name of the first (assumped to be x-inputs) term of the
            # function.
            x_input_name = list(function_signature.parameters.keys())[0]
            # Create a dictionary entry and insert at the beginning of the list.
            x_input_dict = {str(x_input_name): x_values}

            # For backwards compatability:
            try:
                parameters[functiondex] = copy.deepcopy(
                    {**x_input_dict, **parameters[functiondex]})
            except Exception:
                parameters[functiondex] = \
                    merge_two_dicts(x_input_dict, parameters[functiondex])

        elif isinstance(parameters[functiondex], (list, tuple)):
            is_dictionary_parameters = False

            # Input the first element, the x-values, just as the first element.
            parameters[functiondex] = list(parameters[functiondex])
            parameters[functiondex] = (x_values,) + parameters[functiondex]
        else:
            # Try and adapt the input into one of the two accepted types.
            try:
                parameters[functiondex] = dict(parameters[functiondex])
            except TypeError:
                try:
                    parameters[functiondex] = list(parameters[functiondex])
                except Exception:
                    raise TypeError('The parameters for function {funt_num} '
                                    'is not and cannot be turned into the '
                                    'accepted input types.'
                                    '    --Kyubey'
                                    .format(funt_num=functiondex + 1))
            else:
                raise InputError('The parameter input for function {funt_num} '
                                 'is unworkable. Please enter it as a '
                                 'dictionary or tuple of parameters.'
                                 '    --Kyubey'
                                 .format(funt_num=functiondex+1))

        # Begin execution of the function. Expect the raising of errors.
        try:
            if (is_dictionary_parameters):
                # Output the function given the parameters of the same index.
                # Use argument slicing based on dictionaries.
                y_values.append(
                    functions[functiondex](**parameters[functiondex]))
            else:
                # Output the function given the parameters of the same index.
                # Use argument slicing based on aligned tuples or lists.
                y_values.append(
                    functions[functiondex](*parameters[functiondex]))
        except Exception:
            print('Error occurred on function {funt_num} '
                  '( functiondex = {functdex}.'
                  '    --Kyubey'
                  .format(funt_num=functiondex+1, functdex=functiondex))
            # Re-raise the error.
            raise

    # Extract only the highest values of y_points.
    y_values = np.amax(y_values, axis=0)

    return np.array(y_values, dtype=float)


def Stokes_parameter_polarization_angle(Q,U):
    """
    This function returns an angle of polarization in radians based on the 
    values of two stoke parameters. The angle is signed.
    """

    # Type check
    Q = valid.validate_float_array(Q)
    U = valid.validate_float_value(U)

    # Based off of Wikipedia and computational testing
    angle = 0.5*np.arctan2(U,Q)

    return angle


def user_equation_parse(user_eq_input,variables):
    """
    This function returns a functional form of a user's input expression. 
    Only standard python math functions are to be used, and nothing else. 
    The functional form will be in return f(x), for the user inputs some string
    for f(x).

    Variables is a string tuple that contains the list of variables expected in
    the equation parse.
    """

    # Find the number of variables expected, and map to required inputs.
    try:
        variables = valid.validate_tuple(variables)
        n_variables = len(variables)
    except Exception:
        print('Double check input variable stipulations:   {input}'
              .format(input=str(variables)))
        raise

    # Type check.
    try:
        user_eq_input = valid.validate_string(user_eq_input)
    except TypeError:
        try:
            # Test to see if the user input a function instead for whatever 
            # reason.
            user_eq_input = valid.validate_function_call(
                user_eq_input,n_parameters=n_variables)

            # If it hits here, the user has input their own function. This 
            # could be dangerous, warn the user.
            valid.kyubey_warning(DangerWarning,
                                 ('It has been detected that an input string '
                                  'for equation parsing is actually a '
                                  'function with the following name: '
                                  '{funt_name}. If this is correct, continue'
                                  'with prompt.'
                                  '    --Kyubey'),
                                 halt_input=True)
            
            # This is chancy, and should be avoided.
            return user_eq_input

        except Exception:
            # It seems it is not a function either. Raise the user once more.
            raise InputError('The string input cannot be turned into a '
                             'parseable function call.'
                             '    --Kyubey')
    
    # Else, try sympy methods or base methods.
    try:
        # The string should be valid in equation form now. Define some symbols.
        sy_variables = sy.symbols(variables)
        sy_variables = sy.utilities.flatten(sy_variables)
        # Attempt to convert the string function input into a lambda equation.
        function = sy.utilities.lambdify(sy_variables,eval(user_eq_input))
        
    except Exception:
        # It does not seem like it can be done with Sympy. Try with base
        # functionality, but, also be very cautious.
        variable_string = ''
        for variabledex in variables:
            variable_string += variabledex + ','
        # Knock off the extra ','
        variable_string = copy.deepcopy(variable_string[:-1])

        # Define the execute function line.
        eval_string = 'lambda ' + variable_string + ' : ' + user_eq_input

        # Warn the user before executing the execution of the string just in
        # case.
        kyubey_warning(DangerWarning,('The following string is going to be'
                                      'passed through the "eval" function. '
                                      'Is this a safe to pass this string? \n'
                                      '< {eval_str} > \n'
                                      '    --Kyubey'
                                      .format(eval_str=eval_string)),
                                      halt_input=True)
        # If the user is very sure.
        function = eval(eval_string)

    return function