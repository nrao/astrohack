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

Source code installation
########################

If you would like or need to be following the latest developments of astrohack, it is also possible to install astrohack from source by downloading
the `source code
<https://github.com/nrao/astrohack/archive/refs/heads/astrohack-dev.zip>`_
directly from github or using ``git clone``.

.. code-block:: sh

   $ cd <your/preferred/installation/location>
   $ git clone git@github.com:nrao/astrohack.git
   $ cd astrohack


With the zip extracted or the cloned repository via git you can then navigate to
astrohack's root directory and make a local editable (-e) pip installation:

.. code-block:: sh
		
   $ cd <Astrohack_root_dir>
   $ pip install -e .

Updating a local git installation
---------------------------------

To update a local git installation it is necessary to use git.
The default installation follows the ``astrohack-dev`` branch, i.e. the main development branch of astrohack, which unless you are working with active development is the branch that should be followed. The updating process is the following:

.. code-block:: sh

   $ cd <Astrohack_root_dir>
   $ git pull

If you are required to follow a specific development branch, there is one extra step:

.. code-block:: sh

   $ cd <Astrohack_root_dir>
   $ git switch <current-development-branch-name>
   $ git pull

Installation inside CASA
########################

Astrohack can now be installed inside CASA! this is possible due to a new
python package (`casacoretables <https://pypi.org/project/casacoretables/>`_)
that reimplements the access to CASA tables so that they can be accessed
inside a CASA environment without conflicts with CASA.

For installation inside CASA to work it is required that the CASA version
being used is based on python 3.12 (CASA version >= 6.7). The installation can then be done as is described above, but remembering to execute the pip commands inside casa, regular pip install:

.. code-block:: sh

   $ casa
   CASA <1>: pip install astrohack

Source code editable install:

.. code-block:: sh

   $ cd <Astrohack_root_dir>
   $ casa
   CASA <1> pip install -e .

Running CASA + Astrohack @ NRAO
###############################

The distributed CASA versions available in workstations at NRAO (e.g. casa-pipeline) do not allow for the installations of packages, as they are located in remote machines to which the user has no write access, making it impossible to install astrohack directly in them. Currently there are two workarounds:

- Install a local version of casa and then follow the above instructions (Fastest).
- Create an alias on your .bashrc or .profile to a version maintained by me on lustre (very slow...):

.. code-block:: sh

   alias casa-astrohack="/lustre/aoc/projects/ngvla/vdesouza/casa-astrohack/casa-6.7.5-18-py3.12.el8/bin/casa"

In the near future (fall) I will work with IT services to provide a distributed version of casa + astrohack to be available at all sites.


Installation or execution problems
##################################

If the user encounters any issues during installation and/or execution
of astrohack they should leave an issue here on github or write an
e-mail to Victor de Souza at NRAO.
