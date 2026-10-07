Antenna position Correction pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Astrohack provides an executable script for obtaining antenna position corrections, which is installed by pip somewhere in the PATH. This script has 5 main stages:

#. ASDM import to ms (if data set has not yet been imported to an MS).

#. A CASA pre-locit stage where visibilities are channel averaged and phase solutions are computed.

#. A locit stage where phase solutions are extracted from a gain table and then processed to obtained antenna position corrections.

#. An export stage where data products such as plots and reports are created.

#. The creation of a report grouping all the data products.

This pipeline has been written under the assumption that the user will be running it in CASA or in an environment that provides the casatasks, casaplotms and casatools modules. The instructions below assume that the pipeline is being run inside CASA.

Pipeline interface
==================

The pipeline has been written with a simple command line interface that expects two mandatory arguments from the user, the name of the dataset to be processed (be it an MS or an ASDM) and a reference antenna for the calibration stage, several execution customization options are also available a simple help can be accessed with the ``-h`` flag:

.. code-block::

    CASA <1>: !baseline-reduction-pipeline -h

    #####################################################################################################################################
    ###  Welcome to the AstroHACK baseline pipeline for the VLA                                                                       ###
    #####################################################################################################################################

    usage: baseline-reduction-pipeline [-h] [-r ROOT_NAME] [-f FRINGEFIT_SOURCE] [--scans_to_flag SCANS_TO_FLAG] [-i INTENT] [-s SPW]
                                       [-a ANTENNA] [-e ELEVATION_LIMIT] [-p {both,L,R}] [-c {simple,difference}] [-k]
                                       [-l DELAY_LIMITS] [-d DPI] [-o] [--starting-stage {calibration,locit,exports,report}]
                                       [--reimport-asdm] [-y]
                                       filename refant

    CASA baseline pipeline

    positional arguments:
      filename              Path to the input MS/ASDM file
      refant                Reference antenna for calibration

    options:
      -h, --help            show this help message and exit
      -r ROOT_NAME, --root-name ROOT_NAME
                            Root name for the calibration tables, default is filename without extension
      -f FRINGEFIT_SOURCE, --fringefit_source FRINGEFIT_SOURCE
                            Fringe fit source, default is 0319+415
      --scans_to_flag SCANS_TO_FLAG
                            Comma separated list of scans to flag, default is None
      -i INTENT, --intent INTENT
                            Intent for pointing observations.
      -s SPW, --spw SPW     Select SPWs for locit processing, for a list use comma separated values with no spaces, e.g.: '0,1,2', default is all
      -a ANTENNA, --antenna ANTENNA
                            Select antennas for which to produce antenna position corrections, for a list use comma separated values with no spaces, e.g.: 'ea01,ea02', default is all
      -e ELEVATION_LIMIT, --elevation-limit ELEVATION_LIMIT
                            Lowest elevation of data for consideration in degrees, default is 10.0
      -p {both,L,R}, --polarization {both,L,R}
                            Which polarization hands to be used for locit processing, default is both
      -c {simple,difference}, --combination {simple,difference}
                            How to combine different spws for locit processing, default is simple
      -k, --fit_kterm       Fit antennas K term (i.e. Offset between azimuth and elevation axes)
      -l DELAY_LIMITS, --delay_limits DELAY_LIMITS
                            Delay limits for delay plots, values must be given between quotes("), default is "-0.1,0.1"
      -d DPI, --dpi DPI     DPI for png figures (default: 300)
      -o, --overwrite       Overwrite existing files (MSes, caltables, locit files, plots)
      --starting-stage {calibration,locit,exports,report}
                            Starting stage in which to start processing (default: calibration).
      --reimport-asdm       Forcefully re-import the asdm file is the ms already exists (default: False)
      -y, --assume-yes      Assume yes on proceed.

Pre-locit stage
===============

With a reference antenna chosen it is now time to run the antenna position correction pipeline.
The first step of the pipeline is to check whether the data is an ASDM or an MS and if it is an ASDM if it needs to be imported into an MS. With an MS in hands the pipeline proceeds to fetching some metadata from it and then prints a summary of what it has found and which parameters it will use for calibration and further data reduction, e.g.:

.. code-block::

    CASA <3>: !baseline-reduction-pipeline short_x.ms ea13 -f 2148+611

    #####################################################################################################################################
    ###  Welcome to the AstroHACK baseline pipeline for the VLA                                                                       ###
    #####################################################################################################################################

    2026-07-20 17:44:09	INFO	msmetadata_cmpt.cc::open	Performing internal consistency checks on short_x.ms...

    Baseline determination parameters:
        filename           => short_x.ms
        refant             => ea13
        root_name          => None
        fringefit_source   => 2148+611
        scans_to_flag      => []
        intent             => CALIBRATE_POINTING#ON_SOURCE
        spw                => all
        antenna            => all
        elevation_limit    => 10.0
        polarization       => both
        combination        => simple
        fit_kterm          => False
        delay_limits       => [-0.1, 0.1]
        dpi                => 300
        overwrite          => False
        starting_stage     => calibration
        reimport_asdm      => False
        assume_yes         => False
        is_asdm            => False
        msname             => short_x.ms
        pointing_only_ms   => short_x.pnt.ms
        freq_averaged_ms   => short_x.avg.ms
        fringefit_caltable => short_x.sbd
        phase_caltable     => short_x.pha.gcal
        antpos_caltable    => short_x.antpos
        locit_name         => short_x.locit.zarr
        position_name      => short_x.position.zarr
        exports_name       => short_x.exports
        report_name        => short_x-report.html
        n_chan             => 64


    Proceed? <(Y)es/(N)o>:


The check before proceeding can be suppressed by adding the ``-y`` option to the call, e.g.:

.. code-block::

    CASA <4>: !baseline-reduction-pipeline short_x.ms ea13 -y

The code will then proceed through the pre-locit steps:

#. Split the data to contain only the pointing scans with ``split``.

#. (Optional) Flag scans provided with the ``--scans-to-flag`` option, default is to do no flagging.

#. Perform a ``fringefit`` over all sources in the pointing only MS to obtain a delay estimate with each spectral window.

#. Apply the fringefit computed delays with ``applycal``.

#. Average all channels in the now phase aligned spectral windows with ``split``.

#. Obtain phase solutions for all sources using ``gaincal(calmode="p")``

#. Apply the phase solutions to the channel averaged ms with ``applycal`` and then plot then with ``plotms`` for user inspection (they are now expected to be clustered around 0).


Locit stage
===========

With the phase gain table obtained in the previous stage the pipeline now goes through the astrohack steps in the locit stage:

#. `extract_locit <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_locit/index.html>`_: extract phase gains from the gain table and stored then in a convenient format for further processing.

#. `locit <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/locit/index.html>`_: Process phases from all spectral windows either combined or through their differences to produce antenna position solutions.

In case of failures or there is a desire to re run the pipeline from a particular stage, the user can then use option ``--starting-stage``.
For more details on the antenna position corrections processing stages there is the more detailed `locit tutorial <https://astrohack.readthedocs.io/en/stable/tutorials/locit_tutorial.html>`_.

Export & report stages
======================

After the astrohack data files are created, the pipeline then proceeds to execute the exporting functions from the associated Python classes:

#. `AstrohackLocitFile.plot_source_positions <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/locit_mds/index.html>`_: Single plot showing the positions in the sky of the sources used for obtaining antenna position corrections.

#. `AstrohackLocitFile.plot_array_configuration <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/locit_mds/index.html>`_: Single plot displaying the array configuration at observation time.

#. `AstrohackPositionFile.export_locit_fit_results <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a single table with all antenna position corrections.

#. `AstrohackPositionFile.export_results_to_parminator <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a parminator file with proposed antenna position corrections to bea applied at the correlator.

#. `AstrohackPositionFile.plot_position_corrections <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a single plot with arbitrarily scaled antenna position corrections over the plot of the array configuration to have a graphical representation of antenna corrections.

#. `AstrohackPositionFile.plot_delays <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a plot per antenna showing the measured delays, the modeled delays and the residual delays.

After creating the astrohack plots the pipeline then proceeds to an extra stage:

#. Produce an antenna position correction calibration table using CASA's ``gencal``.

#. Apply antenna position corrections to the channel averaged MS using ``applycal``.

#. Produce plots of the over time for the raw and baseline corrected data.

After the production of these export products the pipeline then creates a standalone HTML report with all of them that can then be stored or shared without the need to carry any extra data, an example of such a report can be seen `here <../example-baseline-short_x-report.html>`_.


