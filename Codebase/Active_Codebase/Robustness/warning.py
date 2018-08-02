import random
import warnings

from Robustness.exception import *

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


class PhysicalityWarning(Warning):
    """
    A warning to be used when the current program is doing something that does
    not make sense in real life.
    """

    def __init__(self, message):
        self.message = message


class PhysicsWarning(PhysicalityWarning):
    """
    A warning to be used when the current program is doing something a bit
    risky or something that would not make normal sense in physics terms.
    """

    def __init__(self, message):
        self.message = message


class AstronomyWarning(PhysicalityWarning):
    """
    A warning to be used when the current program is doing something a bit
    risky or something that would not make normal sense in astronomical terms.
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

    Parameters
    ----------
    warn_class : :py:class:`warning` object
        The warning which to submit.
    message : string
        The warning message.
    stacklevel : int
        The stack level call that the warning goes back to.
    input_halt : bool; optional
        If the warning requires user input to continue, this is true. Defaults
        to false.

    Raises
    ------
    :py:exc:`~.TerminateError` in the event of the input halt failing.

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
