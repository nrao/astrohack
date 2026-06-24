Astrohack Installation
~~~~~~~~~~~~~~~~~~~~~~

Installation under Anaconda
###########################

When installing Astrohack in an `Anaconda
<https://docs.conda.io/projects/conda/en/latest/>`_ environment it is
recommended to start with a fresh environment, Preferably under
python3.13, as it is the most recent, and also fastest, version of
python supported by astrohack. A fresh environment is recommended as
to avoid conflicting dependencies with other packages. To create such
an environment:

.. code-block:: sh
		
   $ conda create --name astrohack python=3.12 --no-default-packages
   $ conda activate astrohack

Astrohack reads MeasurementSets and CASA calibration tables through
`casacoretables <https://github.com/nrao/casacoretables>`_, a self-contained
build of casacore's table system that is installed automatically as a
dependency on both Linux and macOS. No separate ``python-casacore`` install is
required (this used to be a manual step on macOS, and is no longer needed).

Astrohack is not yet available for download directly from conda-forge,
therefore we suggest to install astrohack by using pip:

.. code-block:: sh

   $ pip install astrohack

It is also possible to install astrohack from source by downloading
the `source code
<https://github.com/nrao/astrohack/archive/refs/heads/astrohack-dev.zip>`_
directly from github. With the zip extracted you can then navigate to
the root directory and make a local pip installation:

.. code-block:: sh
		
   $ cd <Astrohack_root_dir>
   $ pip install -e .
		

Installation inside CASA
########################

Astrohack can now be installed inside CASA! this is possible due to a new
python package (`casacoretables <https://pypi.org/project/casacoretables/>`_)
that reimplements the access to CASA tables so that they can be accessed
inside a CASA environment without conflicts with CASA.

For installation inside CASA to work it is required that the CASA version
being used is based on python 3.12 (CASA version >= 6.7). The installation can then be done with a simple pip call:

.. code-block:: sh

   CASA <#>: pip install astrohack


Installation or execution problems
##################################

If the user encounters any issues during installation and/or execution
of astrohack they should leave an issue here on github or write an
e-mail to Victor de Souza at NRAO.
