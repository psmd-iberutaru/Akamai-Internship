import copy

import numpy as np
import scipy as sp
import scipy.special as sp_spcl
import matplotlib.pyplot as plt

from Robustness.exception import *
import Robustness.validation as valid
import hourglass_magnetic_field as hgm
import circular_magnetic_field as crm
import misc_functions as misc


def combine_hourglass_circular(percent_hg_to_cr,
                               # Parameters for hourglass
                               hg_center, h, k_array, disk_radius, uniform_B0,
                               # Parameters for circular function
                               cr_center, cir_function):
    """
    This function provides a functional form for a linearly combined hourglass
    and circular magnetic field, given the parameters for both, along with
    the percent combination of the two functions.
    """

    def return_combine_function(x, y):
        # Cordinate shift the values based on center.
        hg_x_value = x - hg_center[0]
        hg_y_value = y - hg_center[1]

        cr_x_value = x - cr_center[0]
        cr_y_value = y - cr_center[1]

        # Determine the strength of the magnetic field of the hourglass model.
        hg_B_x = hgm.Ewer_Basu__B_r(hg_x_value, hg_y_value, 
                                    h, k_array, disk_radius)
        hg_B_y = hgm.Ewer_Basu__B_z(hg_x_value, hg_y_value, 
                                    h, k_array, disk_radius, uniform_B0)

        # The circular magnetic field should overlap perpendicular to the
        # hourglass, such that axis are, for hourglass and circular:
        # z -> x,y. We shall use z -> x.
        cr_B_x, cr_B_y= crm.circular_magnetic_field_cart(
            cr_x_value, cr_y_value, cir_function)

        # Apply the percentage ratios.
        net_B_x = percent_hg_to_cr * hg_B_x + (1-percent_hg_to_cr) * cr_B_x
        net_B_y = percent_hg_to_cr * hg_B_y + (1-percent_hg_to_cr) * cr_B_y

        return net_B_x,net_B_y

    # The function has been defined above according to the user specifications.
    # A docstring just in the event that they forget what is what.
    hybrid_hg_cr_function = return_combine_function
    hybrid_hg_cr_function.__doc__ = \
        """
        Hourglass and circular function magnetic field function.

        Hourglass contribution: {hg_percent_str} 
            Hourglass function parameters:
        --------------------------------------
            center      =  {hg_center_str}
            h           =  {h_str}
            k_array     =  {k_array_str}
            disk_radius =  {disk_radius_str}
            uniform_B0  =  {uniform_B0_str}
        
        Circular contribution: {cr_percent_str}
            Circular function parameters:
        --------------------------------------
            center      =  {cr_center_str}
            cir_funct   =  {cir_function_str}
        """.format(
            # Hourglass things.
            hg_percent_str = str(percent_hg_to_cr),
            hg_center_str = str(hg_center),
            h_str = str(h),
            k_array_str = str(k_array),
            disk_radius_str = str(disk_radius),
            uniform_B0_str = str(uniform_B0),
            # Circular things.
            cr_percent_str = str(1 - percent_hg_to_cr),
            cr_center_str = str(cr_center),
            cir_function_str = str(cir_function.__name__)
        )

    # Finally, return the function to the user for their own usage.
    return hybrid_hg_cr_function