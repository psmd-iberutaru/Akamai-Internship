.. Akamai Polarization Modeling documentation master file, created by
   sphinx-quickstart on Wed Aug  1 09:58:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Akamai Polarization Modeling's documentation!
========================================================

This is the documentation of a simulation program that simulates observations of magnetic morphologies of star-forming regions. Modeling arbitrary and custom 3D cloud shape geometries, integrating along sightlines over user-inputted magnetic fields, density profiles, and polarization functions.

Overall, this codebase would return the user back the values of `Stokes parameters <https://en.wikipedia.org/wiki/Stokes_parameters>`_ for the given object that they inputted. It will also, by default, plot the data to contour maps in the form of heat maps that visually describe the object that the user input into the simulation.

Written apart of a `Akamai Workforce Initiative <https://akamaihawaii.org/>`_ Summer internship. 

Written in `Python 3.6 <https://www.python.org/>`_, this package also relies on the following external Python packages:

- `Numpy <http://www.numpy.org/>`_
- `Scipy <https://www.scipy.org/>`_
- `Sympy <http://www.sympy.org/en/index.html>`_
- `Matplotlib <https://matplotlib.org/>`_
- `Astropy <http://www.astropy.org/>`_

However, it is suggested to use a standard `Anaconda Distribution <https://www.anaconda.com/download/>`_ instead of manually installing the packages in the event of conflicts.



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
