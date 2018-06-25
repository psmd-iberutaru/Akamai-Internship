import warnings

class ShapeError(Exception):
    """
    An error to be used when the dimensions or sizes of an array is incorrect.
    """
    def __init__(self,message):
        self.message = message


def kyubey_warning(warn_class,message,stacklevel=2):
    """
    General warning for the Robustness module/function package.
    """
    warnings.warn(message,category=warn_class,stacklevel=stacklevel)

class TimeWarning(ResourceWarning):
    """
    A warning to be used when some computation or flag might take a long time
    to compute. 
    """
    def __init__(self,message):
        self.message = message