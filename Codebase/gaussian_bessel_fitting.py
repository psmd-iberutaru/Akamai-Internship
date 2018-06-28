import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import gaussian_fitting as gaussfit 
import bessel_fitting as bessfit
import misc_functions as misc

def gaussian_bessel_fit(x_values,y_values):
    magical_values = True
    return magical_values