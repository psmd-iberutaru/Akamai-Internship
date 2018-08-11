Example
=======

This is the example from the :doc:`~.quickstart` guide function. 

The combined code block of that guide is below.

.. code:: python

    import numpy as np
    import astropy as ap
    import astropy.coordinates as ap_coord

    import model_observing as m_obs

    # Making the coordinate input, should be an Astropy SkyCoord class.
    sky_coordinates = ap_coord.SkyCoord('00h00m00.00s','00d00m00.00s',frame='icrs')

    # Making a cloud function, a sphere in this case. Note that the units
    # are in angular space, and thus the unit of circle is radians.
    def cloud_equation(x,y,z):
        radius = 0.01
        return x**2 + y**2 + z**2 - radius**2

    # Making a magnetic field that is uniform in one direction. Consider a 
    # field that is always 0i + 1j + 2k.
    def magnetic_field(x,y,z):
        Bfield_x = np.zeros_like(x)
        Bfield_y = np.ones_like(y)
        Bfield_z = np.full_like(z,2)

        return Bfield_x,Bfield_y,Bfield_z

    # Making a density function of a 1/r**2 profile.
    def density_function(x,y,z):
        density = 1/np.dot([x,y,z],[x,y,z])

        # The above line is a faster implementation of the following.
        # density = 1/(x**2 + y**2 + z**2)

        return density

    # Making a polarization function of a 1/r**2 profile.
    def polarization_function(x,y,z):
        polarization = 1/np.dot([x,y,z],[x,y,z])

        # The above line is a faster implementation of the following.
        # polarization = 1/(x**2 + y**2 + z**2)

        return polarization

    # Create the protostar class.
    protostar = m_obs.ProtostarModel(sky_coordinates, 
                                     cloud_equation, 
                                     magnetic_field,
                                     density_function, 
                                     polarization_function)


    # Creating the Sightline class using the SkyCoord class.
    sightline = m_obs.Sightline('00h00m00', '00d00m00')


    # Define the field of view of the observation, in radians as the total 
    # length of the observation square.
    field_of_view = 0.015

    observing_run = m_obs.ObservingRun(protostar,sightline,field_of_view)

    # Decide on the resolution of the data observed, this sets the number of
    # data points on one axis.
    axis_resolution = 30

    results = observing_run.Stokes_parameter_contours(
        n_axial_samples=axis_resolution)


