Quickstart
==========

This is a brief quickstart introduction to this module/code. This highlights the main purposes of the code, providing examples and other references.

It is assumed that the program/code has been installed along with its inherent dependencies. If not, then consult the main page for more information.

The core of this module is the :py:class:`~.ObservingRun` object. This object, as the name implies, is the class that acts as the observation run of the simulation. 

In order to make the :py:class:`~.ObservingRun` object, we first need two other objects: :py:class:`~.ProtostarModel` and :py:class:`~.Sightline` objects.

First, let us make the :py:class:`~.ProtostarModel` object. First off there are five things that we need to provide in order to make such an object (although two are optional).
