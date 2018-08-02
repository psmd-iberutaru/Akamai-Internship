import warnings
import random


########################################################################
# Errors
########################################################################

class TerminateError(BaseException):
    """
    A very serious error that should override the try-except blocks
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
