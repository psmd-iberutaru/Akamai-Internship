Quickstart
==========

This is a brief quickstart introduction to this module/code. This highlights the main purposes of the code, providing examples and other references.

It is assumed that the program/code has been installed along with its inherent dependencies. If not, then consult the main page for more information.

The core of this module is the :py:class:`~.ObservingRun` object. This object, as the name implies, is the class that acts as the observation run of the simulation. 

In order to make the :py:class:`~.ObservingRun` object, we first need two other objects: :py:class:`~.ProtostarModel` and :py:class:`~.Sightline` objects.

.. _ProtostarModel_Creation:

ProtostarModel Creation
-----------------------

First, let us make the :py:class:`~.ProtostarModel` object. First off there are five things that we need to provide in order to make such an object (although two are optional).

Please note that the coordinate system for the equations and functions within this section is where the origin of the object is the origin of the coordinate system, with the x-axis on the light of sight, positive going from the observing telescope, to the origin of the object, deeper into space, and thus is equivalent to the r-axis from Earth. The y-z plane is assumed to be analogous to the RA-DEC plane of the sky.

The five things that are required for the :py:class:`~.ProtostarModel` is a :py:class:`~.astropy.coordinates.SkyCoord`, a cloud function, a magnetic field function, a density function, and a polarization function. 

Coordinates
+++++++++++

The first parameter is the coordinates of the object in the sky. This is supposed to simulate actual observations, and thus a real sky coordinate is the accepted term. Consider the example parameters below for an object that is at the spring equinox (i.e. a RA of 00h00m00.00s and a DEC of 00d00m00.00s).

.. code:: python

    import astropy as ap
    import astropy.coordinates as ap_coord
    
    # Making the coordinate input, should be an Astropy SkyCoord class.
    sky_coordinates = ap_coord.SkyCoord('00h00m00.00s','00d00m00.00s',
                                        frame='icrs')

Cloud Equation
++++++++++++++

The cloud equation must be a function that takes in three dimensions and is the expression of the implicit functional form. In a general sense, the function is programed as :math:`f(x,y,z) \to \text{ return}`, where the expression
:math:`f(x,y,z)` is from the mathematical implicit equation :math:`f(x,y,z) = 0`.

.. code:: python

    # Making a cloud function, a sphere in this case. Note that the units
    # are in angular space, and thus the unit of circle is radians.
    def cloud_equation(x,y,z):
        radius = 0.01
        return x**2 + y**2 + z**2 - radius**2

Magnetic Field
++++++++++++++

The magnetic field is a little bit more complex, this function is not a scalar function like the density or polarization functions. It is a vector function and thus must return three vectors in cartesian space, still with respect to the origin of the protostar model. Consider the following field that is uniform in one direction. More complex field geometries can be found in :py:mod:`~.magnetic_field_functions_3d` and can be called from there. 

Please note the use of Numpy's structures. Although it is possible to return only integers for the components, using these structures is required if table interpolation is used, and thus usage is suggested for compatibility purposes.

.. code:: python

    import numpy as np

    # Making a magnetic field that is uniform in one direction. Consider a 
    # field that is always 0i + 1j + 2k.
    def magnetic_field(x,y,z):
        Bfield_x = np.zeros_like(x)
        Bfield_y = np.ones_like(y)
        Bfield_z = np.full_like(z,2)

        return Bfield_x,Bfield_y,Bfield_z

Density Function
++++++++++++++++

The density function is a scalar function in three dimensional space. As such, it should return a density value given a three dimensional input (:math:`f(x,y,z) = d`). Consider the example function of a density profile that drops off as a function of :math:`\frac{1}{r^2}`.

.. code:: python

    import numpy as np

    # Making a density function of a 1/r**2 profile.
    def density_function(x,y,z):
        density = 1/np.dot([x,y,z],[x,y,z])

        # The above line is a faster implementation of the following.
        # density = 1/np.dot(x**2 + y**2 + z**2)

        return density

Polarization Function
+++++++++++++++++++++

The polarization function, like the density function is a scalar function in three dimensional space. As such, it should return a value which corresponds to the level of polarization of the light at that given location in space (:math:`f(x,y,z) = p`). Consider the example function of a polarization profile that drops off as a function of :math:`r^2`.

.. code:: python

    import numpy as np

    # Making a polarization function of a 1/r**2 profile.
    def polarization_function(x,y,z):
        polarization = np.sqrt(np.dot([x,y,z],[x,y,z]))

        # The above line is a faster implementation of the following.
        # polarization = np.sqrt(x**2 + y**2 + z**2)

        return polarization

Creating the Class
++++++++++++++++++

From these parameters, a :py:class:`~.ProtostarModel` by the following line.

.. code:: python

    import model_observing as m_obs

    # Create the protostar class.
    protostar = m_obs.ProtostarModel(sky_coordinates, 
                                     cloud_equation, 
                                     magnetic_field,
                                     density_function, 
                                     polarization_function)

Or, all in one block of code.

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

    # Making a density function of a 1/r**2 profile.
    def density_function(x,y,z):
        density = np.sqrt(np.dot([x,y,z],[x,y,z]))

        # The above line is a faster implementation of the following.
        # density = np.sqrt(x**2 + y**2 + z**2)

        return density

    # Making a polarization function of a 1/r**2 profile.
    def polarization_function(x,y,z):
        polarization = np.sqrt(np.dot([x,y,z],[x,y,z]))

        # The above line is a faster implementation of the following.
        # polarization = np.sqrt(x**2 + y**2 + z**2)

        return polarization

    # Create the protostar class.
    protostar = m_obs.ProtostarModel(sky_coordinates, 
                                     cloud_equation, 
                                     magnetic_field,
                                     density_function, 
                                     polarization_function)


Sightline Creation
------------------

When the object is made, it would also be helpful to actually look at the object (simulated observations). Thus, there is a :py:class:`~.Sightline` class. The purpose of this class is to simulate the telescope's pointing location. 

The :py:class:`~.Sightline` class takes in two strings for the RA and DEC of the object. They should be in the following format:

    - RA: ##h##m##.##s (i.e 12h05m10.00s)
    - DEC: ##d##m##.##s (i.e 06d18m10.25s)

Note that the seconds may have more decimals for an accuracy greater than the hundredths place.

If we would like to observe the object that was created in the previous step in :ref:`_ProtostarModel_Creation`, it is best to also point to the object. Therefore, we expect to point to the same location in the sky.

The code below generates a :py:class:`~.Sightline` class just to do that.

.. code:: python
    
    import astropy as ap
    import astropy.coordinates as ap_coord

    import model_observing as m_obs

    # RA of 00h00m00.00s and a DEC of 00d00m00.00s
    sightline = m_obs.Sightline('00h00m00.00s','00d00m00.00s')

Note that the class can also accept a :py:class:`~.astropy.coordinates.SkyCoord` class object. This functionality is done to improve the compatibility with other RA-DEC notations. This alternative method of creating a Sightline is demonstrated below. Because the SkyCoord object contains all of the information needed, the strings that the user would have normally input is ignored in favor for the SkyCoord object.

.. code:: python

    import astropy as ap
    import astropy.coordinates as ap_coord

    import model_observing as m_obs
    
    # Making the SkyCoord class with a RA of 00h00m00.00s and a 
    # DEC of 00d00m00.00s
    sky_coordinates = ap_coord.SkyCoord('00h00m00.00s','00d00m00.00s', 
                                        frame='icrs')

    # Creating the Sightline class using the SkyCoord class.
    sightline = m_obs.Sightline(None, None, SkyCoord_object=sky_coordinates)


ObservingRun Creation
---------------------

When both the :py:class:`~.ProtostarModel` object and the :py:class:`~.Sightline` object is created, an :py:class:`~.ObservingRun` object can be made using both of them.

An :py:class:`~.ObservingRun` object is an object that simulates the act of doing an observing run with a telescope (as the name implies). Through its member functions, the class allows for the computation of different observation run styles.

To make an ObservingRun object, it can be made as follows.

.. code:: python

    import model_observing as m_obs

    # Define the field of view of the observation, in radians as the total 
    # length of the observation square.
    field_of_view = 0.015

    observing_run = m_obs.ObservingRun(protostar,sightline,field_of_view)

From the :py:class:`~.ObservingRun` object, the following observations can be completed from it.

Model Stokes Parameters
+++++++++++++++++++++++

Modeling Stokes parameters in plots is the primary function (as of current) of this repository codebase. It can be normally called by executing the member function :py:meth:`~.ObservingRun.Stokes_parameter_contours`. The most basic execution of this method is as follows.

.. code:: python

    import model_observing as m_obs
    
    # Decide on the resolution of the data observed, this sets the number of
    # data points on one axis.
    axis_resolution = 30

    results = observing_run.Stokes_parameter_contours(
        n_axial_samples=axis_resolution)

The value of the returned function is a list of the RA-DEC values for the sampled points along with the Stokes parameters from a sightline at those given RA-DEC values. Refer to the method documentation (see :py:meth:`~.ObservingRun.Stokes_parameter_contours`) for more information.

It should be the case that the plots of the object desired is created with this method. The method itself should already plot the information for you in heat-map based contours.

Warning: If the values of the intensity is highly similar across the field of view, it may be the case that the colorbar readjustments fatally fails. To prevent this, it is suggested to choose one's field of view such that at least one sightline misses the object. 
