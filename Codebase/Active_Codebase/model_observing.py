"""
Model observing, this module is built to simulate actual observing. The 
object is known, and given sight parameters, the data is given. In particular,
these functions actually give the values of terms derived from the object
model also provided.
"""
import copy

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy as ap
import astropy.units as ap_u
import astropy.coordinates as ap_coord

import Robustness as Robust
import Backend as _Backend

import data_systematization as d_systize


class Sightline():
    """
    This is a sightline. It contains the information for a given sightline
    through space. The sightline is always given by the RA and DEC values. 

    The notation for the accepted values of RA and DEC is found in the 
    Astropy module's :py:class:`~.astropy.coordinates.SkyCoord` class.

    Attributes
    ----------
    self.coordinates : Astropy :py:class:`~.astropy.coordinates.SkyCoord` object.
        This is the sky coordinates of the sightline.

    Methods
    -------
    sightline_parameters() : function (returns | ndarray,ndarray)
        This method returns back both the sightline's center and slopes for
        an actual geometrical representation of the line. Converting from 
        the equatorial coordinate system to the cartesian coordinate system.

    """

    def __init__(self, right_ascension, declination,
                 SkyCoord_object=None, ra_wrap_angle=2*np.pi):
        """Initialization of a sightline.

        The creates the sightline's main parameters, the defineing elements
        of the sightline is the location that it is throughout space. This
        is a specific wrapper around :py:class:`~.astropy.coordinates.SkyCoord`.

        Arguments
        ---------
        right_ascension : string
            The right ascension value for the sightline. This term must be 
            formatted in the Astropy :py:class:`~.astropy.coordinates.SkyCoord` format: ``00h00m00.00s``.
            For the values of the seconds are decimal and may extend to any 
            precision.
        declination : string
            The declination value for the sightline. This term must be 
            formatted in the Astropy :py:class:`~.astropy.coordinates.SkyCoord` format: ``±00d00m00.00s``.
            For the values of the seconds are decimal and may extend to any 
            precision.
        Skycord_object : :py:class:`~.astropy.coordinates.SkyCoord` object; optional
            It may be easier to also just pass an Astropy 
            :py:class:`~.astropy.coordinates.SkyCoord` object in
            general. The other strings are ignored if it is successful.

        Parameters
        ----------
        ra_wrap_angle : float; optional
            This angle, in radians, specifies where the RA values should wrap.
            Wrapping is considered to be very bad and should be avoided. 
            Defaults to 0/2pi wrapping (i.e ra_wrap_angle = 2pi)
        """
        # Type check.
        if (isinstance(SkyCoord_object, ap_coord.SkyCoord)):
            sky_coordinates = SkyCoord_object
        else:
            # Type check for RA and dec before conversion
            right_ascension = Robust.valid.validate_string(right_ascension)
            declination = Robust.valid.validate_string(declination)
            # Convert the strings to sky cords.
            sky_coordinates = ap_coord.SkyCoord(right_ascension,
                                                declination,
                                                frame='icrs')
        ra_wrap_angle = Robust.valid.validate_float_value(ra_wrap_angle)
        ra_wrap_angle = ra_wrap_angle * ap_u.rad

        # Define the member arguments.
        self.coordinates = sky_coordinates
        self._ra_wrap_angle = ra_wrap_angle

    def sightline_parameters(self):
        """ This function returns the sightline linear parameters.

        The sightline is by definition always parallel to the x-axis 
        of the object to be observed. The plane of the sky is the yz-plane
        of the object. This function returns first the central defining
        point, then the deltas for the equation.

        Returns
        -------
        sightline_center : ndarray
            This returns a cartsian point based on the approximation 
            that, if the x-axis and the r-axis are the same of cartesian 
            and spherical cordinates, then so too are the yz-plane and the 
            theta-phi plane. 
        sightline_slopes : ndarray
            This returns the slopes of the cartesian point values given 
            by the center. Because of the approximation from above, it is 
            always [1,0,0].

        Notes
        -----
        The coordinates of the sightline in relation to the object are as
        follows:

        - The x-axis of the object is equal to the r-axis of the telescope. Both pointing away from the telescope, deeper into space.
        - The y-axis of the object is equal to the RA-axis/phi-axis of the 
        telescope, westward (as y increases, RA decreases)
        - The z-axis of the object is equal to the DEC-axis of the telescope. It is also equal to the negative of the theta-axis 
        when it is centered on theta = pi/2. Points north-south of the 
        telescope.

        """

        # Work in radians.
        ra_radians, dec_radians = self._radianize_coordinates()

        sightline_center = np.array([0, ra_radians, dec_radians])
        sightline_slopes = np.array([1, 0, 0])

        return sightline_center, sightline_slopes

    def _radianize_coordinates(self):
        """This method returns the RA and DEC in radians.

        This method converts the RA and DEC coordinate measurements into
        radians for better accounting.

        Returns
        -------
        ra_radians : float
            The RA coordinate in radians.
        dec_radians : float
            The DEC coordinate in radians.
        """

        # Change the wrapping location if necessary. Astropy requires a unit.
        self.coordinates.ra.wrap_angle = self._ra_wrap_angle

        ra_radians = float(self.coordinates.ra.hour * (np.pi / 12))
        dec_radians = float(self.coordinates.dec.radian)

        return ra_radians, dec_radians


class ProtostarModel():
    """
    This is an object that represents a model of an object in space. It
    contains all the required functions and parameters associated with 
    one of the objects that would be observed for polarimetry data.

    Attributes
    ----------
    self.coordinates : Astropy :py:class:`~.astropy.coordinates.SkyCoord` object
        This is the coordinates of the object that this class defines.
    self.cloud_model : function
        This is an implicit function (or a numerical approximation thereof) of 
        the shape of the protostar cloud. 
    self.magnetic_field : function
        This is an implicit function (or a numerical approximation thereof) of 
        the shape of the magnetic field.
    self.density_model : function
        This is an implicit function (or a numerical approximation thereof) of
        the shape of the density model of the cloud.
    self.polarization_model : function
        This is an implicit function (or a numerical approximation thereof) of 
        the polarization model of the cloud.
    """

    def __init__(self, coordinates, cloud_model, magnetic_field_model,
                 density_model=None, polarization_model=None,
                 ra_wrap_angle=2*np.pi, zeros_guess_count=100):
        """Object form of a model object to be observed.

        This is the object representation of an object in the sky. The 
        required terms are present.

        Arguments
        ---------
        coordinates : Astropy :py:class:`~.astropy.coordinates.SkyCoord` object
            These are the coordinates of the observation object. It is up
            to the user to put as complete information as possible.
        cloud_model : function or string,
            An implicit equation of the cloud. The origin of this equation 
            must also be the coordinate specified by self.coordinates. Must
            be cartesian in the form ``f(x,y,z) = 0``, for the function or
            string is ``f(x,y,z)``. The x-axis is always aligned with a 
            telescope as it is the same as a telescope's r-axis.
        magnetic_field_model : function or :py:class:`~.InterpolationTable`
            A function that, given a single point in cartesian space, will 
            return the value of the magnitude of the magnetic field's three 
            orthogonal vectors in xyz-space. If an interpolation table is 
            given, a numerical approximation function will be used instead.
        density_model : function or string, or :py:class:`~.InterpolationTable`; optional
            A function that, given a point in cartesian space, will return
            a value pertaining to the density of the gas/dust within at that
            point. Defaults to uniform. If an interpolation table is 
            given, a numerical approximation function will be used instead.
        polarization_model: function, string, float or :py:class:`~.InterpolationTable`; optional
            This is the percent of polarization of the light. Either given as 
            a function (or string representing a function) ``f(x,y,z)``, or 
            as a constant float value. Default is uniform value of 1. If an 
            interpolation table is given, a numerical approximation function 
            will be used instead.

        Parameters
        ----------
        ra_wrap_angle : float; optional
            This angle, in radians, specifies where the RA values should wrap.
            Wrapping is considered to be very bad and should be avoided. 
            Defaults to 0/2pi wrapping (i.e ra_wrap_angle = 2pi)
        zeros_guess_count : int; optional
            This value stipulates how many spread out test points there should 
            be when finding sightline intersection points. A higher number 
            should be used for complex shapes. Defaults at 100.
        """
        # Initialization of boolean checks.
        # Check if the user input a interpolated data table instead of a
        # function. The integration method must change if so.
        input_interpolated_tables = False

        # Type check
        if (not isinstance(coordinates, ap_coord.SkyCoord)):
            raise TypeError('The input for coordinates must be an Astropy '
                            'SkyCord object.'
                            '    --Kyubey')

        if (callable(cloud_model)):
            cloud_model = \
                Robust.valid.validate_function_call(cloud_model,
                                                    n_parameters=3)
        elif (isinstance(cloud_model, str)):
            cloud_model = \
                Robust.inparse.user_equation_parse(cloud_model,
                                                   ('x', 'y', 'z'))
        else:
            raise TypeError('The input for the cloud equation must either '
                            'be a callable function or a string that can '
                            'be converted into an implicit callable function.'
                            '    --Kyubey')

        # Test magnetic field model.
        if (callable(magnetic_field_model)):
            magnetic_field_model = \
                Robust.valid.validate_function_call(magnetic_field_model,
                                                    n_parameters=3)
        elif (isinstance(magnetic_field_model, d_systize.InterpolationTable)):
            # The user has inputted an interpolation table, record such.
            input_interpolated_tables = True
            if (magnetic_field_model.classification == 'vector'):
                magnetic_field_model = \
                    magnetic_field_model.numerical_function()
            else:
                raise TypeError('The magnetic field lookup table must be a '
                                'vector based table. It is currently a '
                                '< {tb} > based table.'
                                '    --Kyubey'
                                .format(tb=magnetic_field_model.classification))

        # Test density model.
        if (callable(density_model)):
            density_model = \
                Robust.valid.validate_function_call(density_model,
                                                    n_parameters=3)
        elif (isinstance(density_model, str)):
            density_model = \
                Robust.inparse.user_equation_parse(density_model,
                                                   ('x', 'y', 'z'))
        elif (isinstance(density_model, d_systize.InterpolationTable)):
            # The user has inputted an interpolation table, record such.
            input_interpolated_tables = True
            if (density_model.classification == 'scalar'):
                density_model = density_model.numerical_function()
            else:
                raise TypeError('The density model lookup table must be a '
                                'scalar based table. It is currently a '
                                '< {tb} > based table.'
                                '    --Kyubey'
                                .format(tb=density_model.classification))
        elif (density_model is None):
            # The user likely did not input a density model, the default
            # is uniform distribution.
            def uniform_density_function(x, y, z): return np.ones_like(x)
            density_model = uniform_density_function
        else:
            raise TypeError('The input for the density equation must either '
                            'be a callable function or a string that can '
                            'be converted into an implicit callable function.'
                            '    --Kyubey')
        # Test polarization model factor
        if (callable(polarization_model)):
            polarization_model = \
                Robust.valid.validate_function_call(polarization_model,
                                                    n_parameters=3)
        elif (isinstance(polarization_model, str)):
            polarization_model = \
                Robust.inparse.user_equation_parse(polarization_model,
                                                   ('x', 'y', 'z'))
        elif (isinstance(polarization_model, (float, int))):
            percent_polarized = float(copy.deepcopy(polarization_model))
            # The user desires a constant value for the percent polarization.

            def constant_function(x, y, z):
                return np.full_like(x, percent_polarized)
            polarization_model = constant_function
        elif (isinstance(polarization_model, d_systize.InterpolationTable)):
            # The user has inputted an interpolation table, record such.
            input_interpolated_tables = True
            if (polarization_model.classification == 'scalar'):
                polarization_model = polarization_model.numerical_function()
            else:
                raise TypeError('The polarization model lookup table must be '
                                'a scalar based table. It is currently a '
                                '< {tb} > based table.'
                                '    --Kyubey'
                                .format(tb=polarization_model.classification))
        elif (polarization_model is None):
            # The user likely did not input a density model, the default
            # is uniform total distribution.
            def uniform_polarization_function(x, y, z): return np.ones_like(x)
            polarization_model = uniform_polarization_function
        else:
            raise TypeError('The input for the polarization model must either '
                            'be a callable function, a string that can '
                            'be converted into an implicit callable function,'
                            'or a constant float/int value.'
                            '    --Kyubey')

        ra_wrap_angle = Robust.valid.validate_float_value(ra_wrap_angle)
        ra_wrap_angle = ra_wrap_angle * ap_u.rad

        zeros_guess_count = Robust.valid.validate_int_value(zeros_guess_count,
                                                            greater_than=0)

        # All models equations must be offset by the coordinates. This
        # transformation assumes the flat approximation of the astronomical
        # sky.
        coordinates.ra.wrap_angle = ra_wrap_angle
        ra_radians = float(coordinates.ra.hour * (np.pi / 12))
        dec_radians = float(coordinates.dec.radian)
        # Translate the cloud model function.
        def translate_cloud_model(x, y, z):
            return cloud_model(x, y - ra_radians, z-dec_radians)
        # Translate the magnetic field function.
        def translate_magnetic_field(x, y, z):
            return magnetic_field_model(x, y - ra_radians, z-dec_radians)
        # Translate the density model function.
        def translate_density_model(x, y, z):
            return density_model(x, y - ra_radians, z-dec_radians)
        # Translate the polarization model function.
        def translate_polarization_model(x, y, z):
            return polarization_model(x, y - ra_radians, z-dec_radians)

        self.coordinates = coordinates
        self.cloud_model = translate_cloud_model
        self.magnetic_field = translate_magnetic_field
        self.density_model = translate_density_model
        self.polarization_model = translate_polarization_model
        self._ra_wrap_angle = ra_wrap_angle
        self._interpolated_tables = input_interpolated_tables
        self._zeros_guess_count = zeros_guess_count

    def _radianize_coordinates(self):
        """This method returns the RA and DEC in radians.

        This method converts the RA and DEC coordinate measurements into
        radians for better accounting.

        Returns
        --------
        ra_radians : float
            The RA coordinate in radians.
        dec_radians : float
            The DEC coordinate in radians.
        """

        # Change the wrapping location if necessary. Astropy requires a unit.
        self.coordinates.ra.wrap_angle = self._ra_wrap_angle

        ra_radians = float(self.coordinates.ra.hour * (np.pi / 12))
        dec_radians = float(self.coordinates.dec.radian)

        return ra_radians, dec_radians


class ObservingRun():
    """Execute a mock observing run of an object. 

    This class is the main model observations of an object. Taking a 
    central sightline and the field of view, it then gives back a set of 
    plots, similar to those that an observer would see after data reduction.

    The class itself does the computation in its methods, returning back 
    a heatmap/contour object plot from the observing depending on the method.

    Attributes
    ----------
    self.observe_target : :py:class:`ProtostarModel` object
        The model target for simulated observing. Conceptually, the object
        that the telescope observes.
    self.sightline : :py:class:`Sightline` object
        The primary sightline that is used for the model observing, 
        conceptually where the telescope is aimed at.
    self.field_of_view : float
        The field of view value of the observation, given as the length
        of the observing chart.

    Methods
    -------
    Stokes_parameter_contours() : function {returns | ndarray,ndarray}
        Compute the value of Stoke parameters at random sightlines from the
        primary sightline and plot them. Returns the values that was used 
        to plot.


    """

    def __init__(self, observe_target, sightline, field_of_view):
        """Doing an observing run.

        Create an observing run object, compiling the primary sightline and
        the field of view. 

        Arguments
        ---------
        observe_target : :py:class:`ProtostarModel` object
            This is the object to be observed. 
        sightline : Sightline object
            This is the primary sightline, in essence, where the telescope
            is pointing in this simulation.
        field_of_view : float
            The width of the sky segment that is being observed. Must be in
            radians. Applies to both RA and DEC evenly for a square image. 
            Seen range is `` (RA,DEC) ± field_of_view/2 ``.
        """

        # Basic type checking
        if (not isinstance(observe_target, ProtostarModel)):
            raise TypeError('The observed target must be a ProtostarModel '
                            'class object.'
                            '    --Kyubey')
        if (not isinstance(sightline, Sightline)):
            raise TypeError('The sightline must be a Sightline class object.'
                            '    --Kyubey')
        field_of_view = Robust.valid.validate_float_value(field_of_view,
                                                          greater_than=0)

        # Check if both objects have the same RA wraping angle. If not, then
        # it is highly likely that the mapping will be incorrect.
        if (observe_target._ra_wrap_angle != sightline._ra_wrap_angle):
            Robust.kyubey_warning(Robust.AstronomyWarning,
                                  ('The RA wrapping angle for both objects '
                                   'are different. This may result in '
                                   'improper mapping during computations.'))

        # Check if the object is actually within the field of view.
        obs_target_ra_radians, obs_target_dec_radians = \
            observe_target._radianize_coordinates()

        sightline_ra_radians, sightline_dec_radians = \
            sightline._radianize_coordinates()

        if (((sightline_ra_radians - field_of_view/2)
             <= obs_target_ra_radians <=
             (sightline_ra_radians + field_of_view/2)) and
            ((sightline_dec_radians - field_of_view/2)
             <= obs_target_dec_radians <=
             (sightline_dec_radians + field_of_view/2))):
            # If at this stage, it should be fine.
            pass
        else:
            raise Robust.AstronomyError('Object is not within the sightline '
                                        'and field of view. Please revise. '
                                        '    --Kyubey')

        # Assign and create.
        self.target = observe_target
        self.sightline = sightline
        self.offset = field_of_view/2

    def Stokes_parameter_contours(self,
                                  plot_parameters=True, n_axial_samples=25):
        """This function produces a contour plot of the stoke values.

        This function generates a large number of random sightlines to 
        traceout contour information of the of the fields. From 
        there, is creates and returns a contour plot.

        The values of the intensity, I, the two polarization values, Q,U, and
        the polarization intensity, hypt(Q,U) is plotted.

        Parameters
        ----------
        plot_parameters : bool; optional
            A boolean value to specify if the user wanted the parameters to be
            plotted.
        n_axial_samples : int; optional
            The number of points along one RA or DEC axis to be sampled. The
            resulting sample is a mesh n**2 between the bounds. Default is 25.

        Returns
        -------
        ra_dec_array : tuple(ndarray)
            This is a tuple of the values of the RA and DEC of the random
            sightlines (arranged in parallel arrays).
        stokes_parameters : tuple(ndarray)
            This is a tuple of ndarrays of the stoke parameters calculated by 
            the random sightlines.
        """
        # Type check
        n_axial_samples = Robust.valid.validate_int_value(n_axial_samples,
                                                          greater_than=0)

        # Make a plotting background.
        fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,
                                                       figsize=(15, 2), dpi=100,
                                                       sharex=True, sharey=True)

        # Extract Stokes parameter data.
        stokes_parameters, ra_dec_array, _ = \
            self._Stoke_parameters(n_axial_samples)

        # Decompose the stokes parameters into I,Q,U,V along with the angle
        # of polarization.
        I, Q, U, V = stokes_parameters
        polar_I = np.hypot(Q, U)
        angle = _Backend.efp.angle_from_Stokes_parameters(Q, U)

        # Double check if the user actually wanted them plotted.
        if (plot_parameters):
            # Arrange the values into plottable values. The x-axis is RA, and
            # the y-axis is DEC.
            x_axis_plot = ra_dec_array[0]
            y_axis_plot = ra_dec_array[1]

            # Color maps, each gets their own special-ish kind of color map
            # depending on the plot.
            intensity_maps = mpl.cm.get_cmap('inferno')
            seismic_map = mpl.cm.get_cmap('seismic')
            Q_polarization_map = \
                _Backend.pltcust.zeroedColorMap(seismic_map, Q.min(), Q.max())
            U_polarization_map = \
                _Backend.pltcust.zeroedColorMap(seismic_map, U.min(), U.max())
            PuOr_map = mpl.cm.get_cmap('PuOr')
            angle_map = \
                _Backend.pltcust.zeroedColorMap(PuOr_map,
                                                angle.min(), angle.max())

            # Extrapolate and plot a contour based on irregularly spaced data.
            ax1_o = ax1.tricontourf(x_axis_plot, y_axis_plot, I, 50,
                                    cmap=intensity_maps)
            ax2_o = ax2.tricontourf(x_axis_plot, y_axis_plot, polar_I, 50,
                                    cmap=intensity_maps)
            ax3_o = ax3.tricontourf(x_axis_plot, y_axis_plot, Q, 50,
                                    cmap=Q_polarization_map)
            ax4_o = ax4.tricontourf(x_axis_plot, y_axis_plot, U, 50,
                                    cmap=U_polarization_map)
            ax5_o = ax5.tricontourf(x_axis_plot, y_axis_plot, angle, 50,
                                    cmap=angle_map)

            # Assign titles.
            ax1.set_title('Total Intensity')
            ax2.set_title('Polar Intensity')
            ax3.set_title('Q Values')
            ax4.set_title('U Values')
            ax5.set_title('Angle')

            # Assign a color bar legends
            fig1.colorbar(ax1_o, ax=ax1)
            fig1.colorbar(ax2_o, ax=ax2)
            fig1.colorbar(ax3_o, ax=ax3)
            fig1.colorbar(ax4_o, ax=ax4)
            fig1.colorbar(ax5_o, ax=ax5)

            plt.show()

        # Just in case they want to play with the data.
        return ra_dec_array, stokes_parameters

    def _compute_integrated_intensity(self, sightline):
        """Computes the total strength of the light/E-field.

        Given a sightline independent of the primary one, this function 
        computes the integrated value of the magnitude of the E-field. It
        is assumed that the magnitude of the E-field is directly related to 
        energy given by the Poynting vector.

        Parameters
        ----------
        sightline : :py:class:`Sightline` object
            The sightline through which the intensity will be calculated 
            through, using the density function.

        Returns
        -------
        integrated_intensity : float
            The total integrated intensity.
        polarized_integrated_intensity : float
            The total integrated intensity from polarization contribution,
            given by the polarization model function.
        error : float
            The error of the integrated intensity.
        """

        # Basic type checking.
        if (not isinstance(sightline, Sightline)):
            raise TypeError('The sightline must be a sightline object.'
                            '    --Kyubey')

        # Extract information about the target. The coefficient is rather
        # arbitrary.
        box_width = 10 * self.offset

        # Extract sightline information
        sightline_center, sightline_slopes = sightline.sightline_parameters()

        # If the Protostar model contains an interpolation table instead of
        # a normal function. Assume the usage of a Simpson's integration.
        if (self.target._interpolated_tables):
            integral_method = 'simpsons'
        else:
            integral_method = 'scipy'

        # Integration function with a polarization dependence, as the amount of
        # polarization influences. The polarization model must be sqrt(f(x))
        # because the user expects a I_p = I_t * p, while the most efficient
        # method of implementation (modifying the E-fields), produces a
        # relationship of I_p = I_t * p**2.
        def polarization_intensity(x, y, z):
            total = (self.target.density_model(x, y, z)
                     * np.sqrt(np.abs(self.target.polarization_model(x, y, z))))
            return total

        # Integrate over the density function.
        integrated_intensity, int_error = _Backend.cli.cloud_line_integral(
            self.target.density_model, self.target.cloud_model,
            sightline_center, box_width,
            view_line_deltas=sightline_slopes,
            n_guesses=self.target._zeros_guess_count,
            integral_method=integral_method)

        # Also find out the total polarized intensity.
        polarized_integrated_intensity, pol_error = \
            _Backend.cli.cloud_line_integral(
                polarization_intensity, self.target.cloud_model,
                sightline_center, box_width,
                view_line_deltas=sightline_slopes,
                n_guesses=self.target._zeros_guess_count,
                integral_method=integral_method)

        # Error propagates in q.uadrature
        error = np.hypot(int_error, pol_error)

        # Return
        return integrated_intensity, polarized_integrated_intensity, error

    def _compute_integrated_magnetic_field(self, sightline):
        """Computes total magnetic field vectors over a sightline.

        Given a sightline independent of the primary one, compute the
        integrated values of the magnetic field vectors. The values given
        is of little importance because of their computation of an improper summation, but the angles are most important. Nonetheless, magnitude
        is preserved.

        Parameters
        ----------
        sightline : :py:class:`Sightline` object
            The sightline through which the magnetic fields will be calculated
            through.

        Returns
        -------
        Bfield_x_integrated : float
            The total value of all x-axial magnetic field vectors added 
            together through the sightline and object cloud.
        Bfield_y_integrated : float
            The total value of all y-axial magnetic field vectors added 
            together through the sightline and object cloud.
        Bfield_z_integrated : float
            The total value of all z-axial magnetic field vectors added 
            together through the sightline and object cloud.
        errors : ndarray
            A collection of error values, parallel to the float value 
            collection above.
        """

        # Basic type checking.
        if (not isinstance(sightline, Sightline)):
            raise TypeError('The sightline must be a sightline object.'
                            '    --Kyubey')

        # Extract information about the target. The coefficient is rather
        # arbitrary.
        box_width = 10 * self.offset

        # If the Protostar model contains an interpolation table instead of
        # a normal function. Assume the usage of a Simpson's integration.
        if (self.target._interpolated_tables):
            integral_method = 'simpsons'
        else:
            integral_method = 'scipy'

        # Define custom functions such that integrating over a vector function
        # is instead an integration over the three independent dimensions.
        def target_cloud_Bfield_x(x, y, z):
            return self.target.magnetic_field(x, y, z)[0]

        def target_cloud_Bfield_y(x, y, z):
            return self.target.magnetic_field(x, y, z)[1]

        def target_cloud_Bfield_z(x, y, z):
            return self.target.magnetic_field(x, y, z)[2]

        # Extract sightline information
        sightline_center, sightline_slopes = sightline.sightline_parameters()

        # Begin computation.
        Bfield_x_integrated, error_x = _Backend.cli.cloud_line_integral(
            target_cloud_Bfield_x, self.target.cloud_model,
            sightline_center, box_width,
            view_line_deltas=sightline_slopes,
            n_guesses=self.target._zeros_guess_count,
            integral_method=integral_method)
        Bfield_y_integrated, error_y = _Backend.cli.cloud_line_integral(
            target_cloud_Bfield_y, self.target.cloud_model,
            sightline_center, box_width,
            view_line_deltas=sightline_slopes,
            n_guesses=self.target._zeros_guess_count,
            integral_method=integral_method)
        Bfield_z_integrated, error_z = _Backend.cli.cloud_line_integral(
            target_cloud_Bfield_z, self.target.cloud_model,
            sightline_center, box_width,
            view_line_deltas=sightline_slopes,
            n_guesses=self.target._zeros_guess_count,
            integral_method=integral_method)

        error = np.array([error_x, error_y, error_z], dtype=float)

        return (Bfield_x_integrated,
                Bfield_y_integrated,
                Bfield_z_integrated,
                error)

    def _Stoke_parameters(self, n_axial_samples):
        """Return the stoke parameters for a large range of random sightlines.

        This function computes an entire slew of Stokes parameters by 
        generating random sightlines within the field of view of the primary
        sightline. This function is the precursor for all of the contour plots.

        Parameters
        ----------
        n_axial_samples : int
            The number of points along one RA or DEC axis to be sampled. The
            resulting sample is a mesh n**2 between the bounds.

        Returns
        -------
        stokes_parameters : ndarray
            This is the array of all four Stoke parameters over all of the 
            random sightlines.
        ra_dec_array : ndarray
            This is the array of all of the random sightline's RA and DEC 
            values.
        sightline_list : ndarray
            This is an array containing all of the sightline's SkyCord objects,
            just in case for whatever need.
        """
        # Type checking.
        n_axial_samples = Robust.valid.validate_int_value(n_axial_samples,
                                                          greater_than=0)

        # Work in radians for the target's center.
        target_ra, target_dec = self.target._radianize_coordinates()

        # Use the primary sightline's wrap angle as it is the most likely to
        # accurate.
        ra_wrap_angle_value = self.sightline._ra_wrap_angle.value

        # Create a large list of sightlines.
        ra_range = np.linspace(target_ra - self.offset,
                               target_ra + self.offset,
                               n_axial_samples)
        dec_range = np.linspace(target_dec - self.offset,
                                target_dec + self.offset,
                                n_axial_samples)

        # Establish a mesh grid, the flatten to 1D arrays of points.
        ra_mesh, dec_mesh = np.meshgrid(ra_range, dec_range)
        ra_array = np.ravel(ra_mesh)
        dec_array = np.ravel(dec_mesh)

        # Compile the sightlines in a list.
        sightline_list = []
        for radex, decdex in zip(ra_array, dec_array):
            temp_skycoord = ap_coord.SkyCoord(radex, decdex,
                                              frame='icrs', unit='rad')
            sightline_list.append(Sightline(None, None, temp_skycoord,
                                            ra_wrap_angle=ra_wrap_angle_value))

        # It is best if it is not vectored like other numpy operations.
        # Because it deals with specific classes and whatnot.
        intensity_array = []
        polarized_intensity = []
        Bfield_x_array = []
        Bfield_y_array = []
        Bfield_z_array = []
        error_array = []
        for sightlinedex in sightline_list:
            temp_intensity, temp_polarized_intensity, intensity_error = \
                self._compute_integrated_intensity(sightlinedex)
            Bfield_x, Bfield_y, Bfield_z, Bfield_error = \
                self._compute_integrated_magnetic_field(sightlinedex)
            # Append
            intensity_array.append(temp_intensity)
            polarized_intensity.append(temp_polarized_intensity)
            Bfield_x_array.append(Bfield_x)
            Bfield_y_array.append(Bfield_y)
            Bfield_z_array.append(Bfield_z)
            # Combine errors in quadrature.
            error_array.append(np.hypot(intensity_error, Bfield_error))

        # Vectorize.
        intensity_array = np.array(intensity_array, dtype=float)
        polarized_intensity = np.array(polarized_intensity, dtype=float)
        Bfield_x_array = np.array(Bfield_x_array, dtype=float)
        Bfield_y_array = np.array(Bfield_y_array, dtype=float)
        Bfield_z_array = np.array(Bfield_z_array, dtype=float)
        error_array = np.array(error_array, dtype=float)

        # The value of Bfeild_x_array is of non-issue because of the desire
        # for the Stokes parameters and the orientation of the coordinate
        # system.
        del Bfield_x_array, Bfield_x

        # Convert the magnetic fields to electric fields. Because the strength
        # of the magnetic field is independent of the strength of the E field
        # through the virtue of the reflecting dust grains, scale by intensity.
        Efield_y_array_norm, Efield_z_array_norm = \
            _Backend.efp.magnetic_to_electric(Bfield_y_array, Bfield_z_array,
                                              normalize=True)
        Efield_y_array = Efield_y_array_norm * intensity_array
        Efield_z_array = Efield_z_array_norm * intensity_array

        # Polarized light.
        Efield_y_array_polar = Efield_y_array_norm * polarized_intensity
        Efield_z_array_polar = Efield_z_array_norm * polarized_intensity

        # Get all of the Stokes parameters.
        I, Q, U, V = \
            _Backend.efp.Stokes_parameters_from_field(
                Efield_y_array_polar, Efield_z_array_polar)

        # Total intensity is actually from the Efield of the total light, not
        # just polarized light.
        I = Efield_y_array**2 + Efield_z_array**2

        # Return all of the parameters as this is a hidden function. The front
        # end contour functions take care of producing only the one the user
        # wants. Also, return back the sightline locations.
        stokes_parameters = (I, Q, U, V)
        ra_dec_array = (ra_array, dec_array)
        return stokes_parameters, ra_dec_array, sightline_list
