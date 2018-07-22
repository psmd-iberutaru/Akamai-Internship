import warnings
import random


########################################################################
# Errors
########################################################################

class TerminateError(BaseException):
    """
    A very serious error that should override the try, except functions
    that are written.
    """

    def __init__(self, message):
        self.message = message


class InputError(Exception):
    """
    An error to be used when the input from a user is incorrect.
    """

    def __init__(self, message):
        self.message = message


class OutputError(Exception):
    """
    An error to be used when the output is grossly unexpected.
    """

    def __init__(self, message):
        self.message = message


class ShapeError(Exception):
    """
    An error to be used when the dimensions or sizes of an array is incorrect.
    """

    def __init__(self, message):
        self.message = message


class AstronomyError(Exception):
    """
    An error to be used if some of the programs executed are trying to 
    do something that is nonsensical in the context of astronomy.
    """

    def __init__(self, message):
        self.message = message


########################################################################
# Warnings
########################################################################

class TimeWarning(ResourceWarning):
    """
    A warning to be used when some computation or flag might take a long time
    to compute. 
    """

    def __init__(self, message):
        self.message = message


class DangerWarning(Warning):
    """
    A warning to be used when some input or output is dangerous for the 
    system or program itself and may be disastrous with unexpected inputs.
    """

    def __init__(self, message):
        self.message = message


class InputWarning(Warning):
    """
    A warning to be used when the values of some item is incorrect, but is 
    fixed within the program.
    """

    def __init__(self, message):
        self.message = message


class OutputWarning(Warning):
    """
    A warning to be used when the values of an output may not use all of the 
    inputs given, or that it might become unexpected because of bugs.
    """

    def __init__(self, message):
        self.message = message


# Begin the main warning function.
def kyubey_warning(warn_class, message,
                   stacklevel=2, input_halt=False):
    """
    General warning for the Robustness module/function package. If the warning
    is serious enough (like a DangerWarning), then force the user to ensure
    the continuation of the program.
    """
    # Warn user.
    warnings.warn(message, category=warn_class, stacklevel=stacklevel)
    # If a halt is desired.
    if (input_halt):
        # Employ random interger returning value to verification.
        validation_number = int(random.randint(0, 9999))
        # Print message.
        print('')
        print('A halt input has been issued by the program. Your input is '
              'required for the program to continue. Please enter in the '
              'following interger value:  < {valid_int} >'
              .format(valid_int=str(validation_number)))
        # Check for the correct validation number.
        user_answer = int(input('Input above integer number:  '))
        print('', end='\n\n')
        # Validate for correct answer.
        if (validation_number == user_answer):
            return None
        else:
            raise TerminateError('The warning verification process has failed. '
                                 'The incorrect value has been inputted. '
                                 'Terminating program to prevent damage. '
                                 '    --Kyubey')
