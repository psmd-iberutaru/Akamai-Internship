"""
The purpose of this function is that inputs are parsed correctly.
"""
import copy

import numpy as np
import scipy as sp
import sympy as sy

import Robustness as Robust

def user_equation_parse(user_eq_input,variables):
    """Convert input implicit equation into a function.

    This function returns a functional form of a user's input expression. 
    Only standard python math functions are to be used, and nothing else. 
    The functional form will be in return f(x), for the user inputs some string
    for f(x). Variables is a string tuple that contains the list of variables 
    expected in the equation parse.

    Parameters:
    -----------
    user_eq_input : string or function
        This is the wanted string or function to be converted. If it is a 
        function, there is simple verification before passing it back.
    variables : tuple of strings
        This is the symbols within the equation for parsing as the input (e.g. 
        the ``x`` and ``y`` in ``f(x,y)``)

    Returns:
    --------
    function : function
        A callable function that executes the mathematical expression given in
        the string. The order of parameters from variables are kept.

    Raises:
    -------
    DangerWarning : Warning
        This is used because the eval() function is used in this code.
    TerminateError : BaseException
        This is done if the verification of the continuation of the program
        fails.

    Notes:
    ------
    This function does use the eval function and excessive precautions are
    used. 
    """

    # Find the number of variables expected, and map to required inputs.
    try:
        variables = Robust.valid.validate_tuple(variables)
        n_variables = len(variables)
    except Exception:
        print('Double check input variable stipulations:   {input}'
              .format(input=str(variables)))
        raise

    # Type check.
    try:
        user_eq_input = Robust.valid.validate_string(user_eq_input)
    except TypeError:
        try:
            # Test to see if the user input a function instead for whatever 
            # reason.
            user_eq_input = Robust.valid.validate_function_call(
                user_eq_input,n_parameters=n_variables)

            # If it hits here, the user has input their own function. This 
            # could be dangerous, warn the user.
            Robust.valid.kyubey_warning(Robust.DangerWarning,
                                        ('It has been detected that an input '
                                         'string for equation parsing is '
                                         'actually a function with the '
                                         'following name: '
                                         '{funt_name}. If this is correct, '
                                         'continue with prompt.'
                                         '    --Kyubey'),
                                        input_halt=True)
            
            # This is chancy, and should be avoided.
            return user_eq_input

        except Exception:
            # It seems it is not a function either. Raise the user once more.
            raise Robust.InputError('The string input cannot be turned into a '
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
        Robust.kyubey_warning(Robust.DangerWarning,
                              ('The following string is going to be passed '
                               'through the "eval" function. Is it safe to '
                               'pass this string? \n '
                               '< {eval_str} > \n '
                               '    --Kyubey'
                               .format(eval_str=eval_string)),
                               input_halt=True)
        # If the user is very sure.
        function = eval(eval_string)

    return function