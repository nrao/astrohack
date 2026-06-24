Beam cut data reduction pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Astrohack provides two executable scripts for the data reduction of beam cuts.
The first one is the CASA beam cut calibration pipeline, intended to be called from within CASA to perform the calibration of the beam cut data using CASA's calibration capabilities.
The second one is the beam cut reduction pipeline that uses AstroHACK's functions and plotting capabilities to produce beam cuts from calibrated MSes.
The following work flows assumes that astrohack has been installed within CASA >=6.7.

Calibration pipeline
####################

After pip installation a python executable called :code:`beamcut-calibration` is available in CASA's path, callable with the :code:`!` marker, e.g.:

.. code-block:: Ipython

   CASA: !beamcut-calibration -h

Which would produce the following output:

.. code-block::

    ####################################################################################################
    ###  Welcome to CASA beam cut calibration pipeline                                               ###
    ####################################################################################################

    usage: beamcut-calibration [-h] [-r ROOT_NAME] [-f BEAMCUT_FIELD] [-o] [-y] [-q QUACK_NCHAN] filename refant

    Beam cut CASA calibration pipeline

    positional arguments:
      filename              Path to the input MS/ASDM file
      refant                Reference antenna for calibration

    options:
      -h, --help            show this help message and exit
      -r ROOT_NAME, --root-name ROOT_NAME
                            Root name for the calibration tables, default is filename without extension
      -f BEAMCUT_FIELD, --beamcut-field BEAMCUT_FIELD
                            Field Id or name of the beam cut data (default is to determine it from data)
      -o, --overwrite       Overwrite existing calibration files
      -y, --assume-yes      Assume yes on proceed.
      -q QUACK_NCHAN, --quack-nchan QUACK_NCHAN
                            Number of channels to quack at the edge of the spectral window (default is 4)

The only required arguments to run the Calibration pipeline are the name of the data on which to work, be it an ASDM or an MS (The pipeline can detect which type it is), e.g.:

.. code-block:: ipython

    CASA: !beamcut-calibration beamcuts_otf_U_002.61188.83799753472.ms ea11

After a few moments the pipeline, before proceeding with the calibration, will produce a summary of the inputs, a list of the calibration tables that will be created, the selected calibration scans, the spectral windows and channels to be used in calibration as well as the field that contains the beamcut data, e.g.:

.. code-block::

    ####################################################################################################
    ###  Welcome to CASA beam cut calibration pipeline                                               ###
    ####################################################################################################

    CASA calibration parameters:
        filename              => beamcuts_otf_U_002.61188.83799753472.ms
        refant                => ea11
        root_name             => beamcut-u-band-cal-01
        beamcut_field         => 3
        overwrite             => True
        assume_yes            => False
        quack_nchan           => 4
        is_asdm               => False
        delay_caltable        => beamcut-u-band-cal-01.delay.cal
        bandpass_caltable     => beamcut-u-band-cal-01.bandpass.cal
        gain_caltable         => beamcut-u-band-cal-01.gain.cal
        msname                => beamcuts_otf_U_002.61188.83799753472.ms
        calibration_scans     => 2,5,10,15
        beamcut_scans         => 8,13
        quacked_spw_selection => 0~7:4~60

    Proceed? <(Y)es/(N)o>:

The check before proceeding can be suppressed by adding the :code:`-y` option to the call, e.g.:

.. code-block:: ipython

    CASA: !beamcut-calibration beamcuts_otf_U_002.61188.83799753472.ms ea11 -y

Beam cut Reduction pipeline
###########################

After the beam cut data has been calibrated the user can then use AstroHACK's beam cut data reduction pipeline (:code:`beamcut-reduction-pipeline`).
This pipeline goes through the steps of `extracting the pointing data from the ms <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_pointing/index.html>`_, `extracting the visibilities and matching them to the pointing data <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_holog/index.html>`_, `grouping the visibilities into beam cuts <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/beamcut/index.html>`_ and finally producing the beam cut plots (Phase, Amplitude, Pointing and Power attenuation) and the beam cut report.
The pipeline options can be seen with the used of the :code:`-h` option, e.g.:

.. code-block:: ipython

    CASA: !beamcut-reduction-pipeline -h

By default the pipeline will run in parallel by using 4 cores, each with 6GB of memory dedicated to them, this can be changes by passing the :code:`-n` option to control the number of cores and :code:`-m` to control the dedicated memory per core. As with the calibration pipeline the pipeline will produce a summary, with a check (that can be suppressed with the :code:`-y` option), before proceeding, e.g.:

.. code-block::

    ####################################################################################################
    ###  Welcome to AstroHACK BeamCut Pipeline                                                       ###
    ####################################################################################################

    Beam cut reduction parameters:
        ms_name                  => beamcuts_otf_U_002.61188.83799753472.ms
        root_name                => u-band
        spectral_window          => [0]
        antenna                  => ['ea28']
        ncores                   => 1
        memory_per_core          => 10GB
        overwrite                => True
        data_column              => CORRECTED_DATA
        assume_yes               => False
        starting_stage           => extract_pointing
        dpi                      => 300
        plot_array_configuration => False
        plot_pointing            => False
        exclude_bad_antennas     => None
        parallel                 => False
        point_name               => u-band.point.zarr
        holog_name               => u-band.holog.zarr
        beamcut_name             => u-band.beamcut.zarr
        exports_name             => u-band.exports

    Proceed? <(Y)es/(N)o>:

When all is done, there will be a few astrohack data files, a `.point.zarr` containing pointing information, a `.holog.zarr` containing the pointing matched visibilities and a `.beamcut.zarr` containing the processed beam cuts. There will be also a directory with a name terminated in `.exports` containing the beam cut reports and corresponding plots for each antenna.

For more details on each of the steps and how to interact with astrohack's beam cut files please refer to our `beam cut tutorial <https://astrohack.readthedocs.io/en/stable/tutorials/beamcut_tutorial.html>`_.