import warnings


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


def kyubey_warning(warn_class, message, stacklevel=2):
    """
    General warning for the Robustness module/function package.
    """
    warnings.warn(message, category=warn_class, stacklevel=stacklevel)


class TimeWarning(ResourceWarning):
    """
    A warning to be used when some computation or flag might take a long time
    to compute. 
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

    def __init__(self,message):
        self.message = message
