"""
Model observing, this module is built to simulate actual observing. The 
object is known, and given sight parameters, the data is given. In particular,
these functions actually give the values of terms derived from the object
model also provided.
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy as ay
import astropy.coordinates as ay_coord

import Robustness as Robust
import Backend


class Sightline():
    """
    This is a sightline. It contains the information for a given sightline
    through space. The sightline is always given by the RA and DEC values. 

    The notation for the accepted values of RA and DEC is found in the 
    Astropy module's SkyCord class.

    Attributes:
    -----------
    self.coordinates : Astropy SkyCord object.
        This is the sky coordinates of the sightline.

    Methods:
    --------
    sightline_parameters() : ndarray,ndarray
        This method returns back both the sightline's center and slopes for
        an actual geometrical representation of the line. Converting from 
        the equatorial coordinate system to the cartesian coordinate system.

    """

    def __init__(self, right_ascension, declination,
                 Skycord_object=None):
        """Initialization of a sightline.

        The creates the sightline's main parameters, the defineing elements
        of the sightline is the location that it is throughout space. This
        is a specific wrapper around SkyCord.

        Arguments:
        ----------
        right_ascension : string
            The right ascension value for the sightline. This term must be 
            formatted in the Astropy SkyCord format: `` '00h00m00.00s' ``. For
            the values of the seconds are decimal and may extend to any 
            precision.
        declination : string
            The declination value for the sightline. This term must be 
            formatted in the Astropy SkyCord format: `` '±00d00m00.00s' ``. For
            the values of the seconds are decimal and may extend to any 
            precision.
        Skycord_object : SkyCord object, optional
            It may be easier to also just pass an Astropy Skycord object in
            general. The other strings are ignored if it is successful.
        """

        if (Skycord_object is not None):
            if (isinstance(Skycord_object, ay_coord.SkyCoord)):
                sky_coordinates = Skycord_object
        else:
            # Type check
            right_ascension = Robust.valid.validate_string(right_ascension)
            declination = Robust.valid.validate_string(declination)
            # Convert the strings to sky cords.
            sky_coordinates = ay_coord.SkyCoord(right_ascension,
                                                declination,
                                                frame='icrs')

        # Define the member arguments.
        self.coordinates = sky_coordinates

    def sightline_parameters(self):
        """ This function returns the sightline linear parameters.

        The sightline is by definition always parallel to the x-axis 
        of the object to be observed. The plane of the sky is the yz-plane
        of the object. This function returns first the central defining
        point, then the deltas for the equation.

        Parameters:
        -----------
        None

        Returns:
        --------
        sightline_center : ndarray
            This returns a cartsian point based on the approximation 
            that, if the x-axis and the r-axis are the same of cartesian 
            and spherical cordinates, then so too are the yz-plane and the 
            theta-phi plane. 
        sightline_slopes : ndarray
            This returns the slopes of the cartesian point values given 
            by the center. Because of the approximation from above, it is 
            always [1,0,0].

        Notes:
        ------
        The coordinates of the sightline in relation to the object are as
        follows:

        - The x-axis of the object is equal to the r-axis of the telescope. Both pointing away from the telescope, deeper into space.
        - The y-axis of the object is equal to the RA-axis/phi-axis of the 
        telescope, eastward.
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

        Returns:
        --------
        ra_radians : float
            The RA coordinate in radians.
        dec_radians : float
            The DEC coordinate in radians.
        """

        ra_radians = float(self.coordinates.ra.hour * (np.pi / 12))
        dec_radians = float(self.coordinates.dec.radian)

        return ra_radians, dec_radians


class ProtostarModel():
    """
    This is an object that represents a model of an object in space. It
    contains all the required functions and parameters associated with 
    one of the objects that would be observed for polarimetry data.

    Attributes:
    -----------
    self.coordinates : Astropy SkyCord object
        This is the coordinates of the object that this class defines.
    self.cloud_equation : function
        This is an implicit function of the shape of the protostar cloud. 
    self.magnetic_field : function
    """

    def __init__(self, coordinates, cloud_equation, magnetic_field_equation):
        """Object form of a model object to be observed.

        This is the object representation of an object in the sky. The 
        required terms are present.

        Arguments:
        -----------
        coordinates : Astropy SkyCord object
            These are the coordinates of the observation object. It is up
            to the user to put as complete information as possible.
        cloud_equation : function or string
            An implicit equation of the cloud. The origin of this equation 
            must also be the coordinate specified by self.coordinates. Must
            be cartesian in the form ``f(x,y,z) = 0``, for the function or
            string is ``f(x,y,z)``. The x-axis is always aligned with a 
            telescope as it is the same as a telescope's r-axis.
        magnetic_field_equation : function
            A function that, given a single point in catesian space, will 
            return the value of the magnitude of the magnetic field's three 
            orthogonal vectors in xyz-space.
        """
        # Type check
        if (not isinstance(coordinates, ay_coord.SkyCoord)):
            raise TypeError('The input for coordinates must be an Astropy '
                            'SkyCord object.'
                            '    --Kyubey')

        if (callable(cloud_equation)):
            cloud_equation = \
                Robust.valid.validate_function_call(cloud_equation,
                                                    n_parameters=3)
        elif (isinstance(cloud_equation, str)):
            cloud_equation = \
                Robust.inparse.user_equation_parse(cloud_equation,
                                                   ('x', 'y', 'z'))
        else:
            raise TypeError('The input for the cloud equation must either '
                            'be a callable function or a string that can '
                            'be converted into an implicit callable function.'
                            '    --Kyubey')

        magnetic_field_equation = \
            Robust.valid.validate_function_call(magnetic_field_equation,
                                                n_parameters=3)

        self.coordinates = coordinates
        self.cloud_equation = cloud_equation
        self.magnetic_field = magnetic_field_equation

    def _radianize_coordinates(self):
        """This method returns the RA and DEC in radians.

        This method converts the RA and DEC coordinate measurements into
        radians for better accounting.

        Returns:
        --------
        ra_radians : float
            The RA coordinate in radians.
        dec_radians : float
            The DEC coordinate in radians.
        """

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
    """

    def __init__(self, observe_target, sightline, field_of_view):
        """Doing an observing run.

        Create an observing run object, compiling the primary sightline and
        the field of view. 

        Attributes:
        -----------
        observe_target : ProtostarModel object
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
                            'class.'
                            '    --Kyubey')
        if (not isinstance(sightline, Sightline)):
            raise TypeError('The sightline must be a Sightline class.'
                            '    --Kyubey')
        field_of_view = Robust.valid.validate_float_value(field_of_view,
                                                          greater_than=0)

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
             (sightline_dec_radians + field_of_view)/2)):
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
                                  n_samples=100):
        """This function produces a contour plot of the stoke values.

        This function generates a large number of random sightlines to 
        traceout contour information of the of the fields. From 
        there, is creates and returns a contour plot.

        The values of the intensity, I, the two polarization values, Q,U, and
        the polarization intensity, hypt(Q,U) is plotted.

        Parameters:
        -----------
        n_samples : int, optional
            The number of random points that will be sampled to build the 
            contour function. Defaults to 100.

        Returns:
        figure : Matplotlib Figure object
            The figure plot.
        """
        # Type check
        n_samples = Robust.valid.validate_int_value(n_samples, greater_than=0)

        # Make a plotting background.
        fig1, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2,
                                                      figsize=(5, 4), dpi=100,
                                                      sharex=True, sharey=True)

        # Extract Stokes parameter data.
        stokes_parameters, ra_dec_array, _ = self._Stoke_parameters(n_samples)

        # Decompose the stokes parameters into I,Q,U,V
        I, Q, U, V = stokes_parameters
        hypt = np.hypot(Q, U)

        # Arrange the values into plottable values. The x-axis is RA, and the
        # y-axis is DEC.
        plotting_x_axis = ra_dec_array[0]
        plotting_y_axis = ra_dec_array[1]

        # Extrapolate and plot a contour based on irregularly spaced data.
        ax1_o = ax1.tricontourf(plotting_x_axis, plotting_y_axis, I, 50)
        ax2_o = ax2.tricontourf(plotting_x_axis, plotting_y_axis, Q, 50)
        ax3_o = ax3.tricontourf(plotting_x_axis, plotting_y_axis, U, 50)
        ax4_o = ax4.tricontourf(plotting_x_axis, plotting_y_axis, hypt, 50)

        # Assign titles.
        ax1.set_title('Intensity')
        ax2.set_title('Q Values')
        ax3.set_title('U Values')
        ax4.set_title('hypot(Q,U)')

        # Assign a color bar legends
        fig1.colorbar(ax1_o, ax=ax1)
        fig1.colorbar(ax2_o, ax=ax2)
        fig1.colorbar(ax3_o, ax=ax3)
        fig1.colorbar(ax4_o, ax=ax4)

        plt.show()

        return None

    def _compute_integrated_magnetic_field(self, sightline):
        """Computes total magnetic field vectors over a sightline.

        Given a sightline independent of the primary one, compute the
        integrated values of the magnetic field vectors. The values given
        is of little importance because of their computation of an improper summation, but the angles are most important. Nonetheless, magnitude
        is preserved.

        Parameters:
        -----------
        sightline : Sightline object
            The sightline through which the magnetic fields will be calculated
            through.

        Returns:
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
        Bfield_x_integrated, error_x = Backend.cli.cloud_line_integral(
            target_cloud_Bfield_x, self.target.cloud_equation,
            sightline_center, box_width, view_line_deltas=sightline_slopes)
        Bfield_y_integrated, error_y = Backend.cli.cloud_line_integral(
            target_cloud_Bfield_y, self.target.cloud_equation,
            sightline_center, box_width, view_line_deltas=sightline_slopes)
        Bfield_z_integrated, error_z = Backend.cli.cloud_line_integral(
            target_cloud_Bfield_z, self.target.cloud_equation,
            sightline_center, box_width, view_line_deltas=sightline_slopes)

        error = np.array([error_x, error_y, error_z], dtype=float)

        return (Bfield_x_integrated,
                Bfield_y_integrated,
                Bfield_z_integrated,
                error)

    def _Stoke_parameters(self, n_samples):
        """Return the stoke parameters for a large range of random sightlines.

        This function computes an entire slew of Stokes parameters by 
        generating random sightlines within the field of view of the primary
        sightline. This function is the precursor for all of the contour plots.

        Parameters:
        -----------
        n_samples : int
            The number of random points that will be sampled to build the 
            contour function.

        Returns:
        --------
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
        n_samples = Robust.valid.validate_int_value(n_samples, greater_than=0)

        # Work in radians for the target's center.
        target_ra, target_dec = self.target._radianize_coordinates()

        # Create a large list of sightlines.
        ra_array = np.random.uniform(target_ra - self.offset,
                                     target_ra + self.offset,
                                     n_samples)
        dec_array = np.random.uniform(target_dec - self.offset,
                                      target_dec + self.offset,
                                      n_samples)

        # Compile the sightlines in a list.
        sightline_list = []
        for radex, decdex in zip(ra_array, dec_array):
            temp_skycoord = ay_coord.SkyCoord(radex, decdex,
                                              frame='icrs', unit='rad')
            sightline_list.append(Sightline(None, None, temp_skycoord))

        # It is best if it is not vectored like other numpy operations.
        # Because it deals with specific classes and whatnot.
        Bfield_x_array = []
        Bfield_y_array = []
        Bfield_z_array = []
        error_array = []
        for sightlinedex in sightline_list:
            Bfield_x, Bfield_y, Bfield_z, error = \
                self._compute_integrated_magnetic_field(sightlinedex)
            # Append
            Bfield_x_array.append(Bfield_x)
            Bfield_y_array.append(Bfield_y)
            Bfield_z_array.append(Bfield_z)
            error_array.append(error)

        # Vectorize.
        Bfield_x_array = np.array(Bfield_x_array, dtype=float)
        Bfield_y_array = np.array(Bfield_y_array, dtype=float)
        Bfield_z_array = np.array(Bfield_z_array, dtype=float)
        error_array = np.array(error, dtype=float)

        # The value of Bfeild_x_array is of non-issue because of the desire
        # for the Stokes parameters and the orientation of the coordinate
        # system.
        del Bfield_x_array, Bfield_x

        # Convert the magnetic fields to electric fields.
        Efield_y_array, Efield_z_array = \
            Backend.efp.magnetic_to_electric(Bfield_y_array, Bfield_z_array)

        # Get all of the Stokes parameters.
        I, Q, U, V = Backend.efp.Stokes_parameters_from_field(Efield_y_array,
                                                              Efield_z_array)

        # Return all of the parameters as this is a hidden function. The front
        # end contour functions take care of producing only the one the user
        # wants. Also, return back the sightline locations.
        stokes_parameters = (I, Q, U, V)
        ra_dec_array = (ra_array, dec_array)
        return stokes_parameters, ra_dec_array, sightline_list
