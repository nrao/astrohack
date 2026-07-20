Beam cut data reduction pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Astrohack provides an executable script for the data reduction of beam cuts, which is installed by pip somewhere in the PATH. This script has 5 main stages:

#. ASDM import to ms (if data set has not yet been imported to an MS).

#. Calibration of the beam cut data using CASA tasks (delay, bandpass and phase).

#. Beam cut processing with astrohack (extract_pointing, extract_holog, beamcut).

#. Data exports generation (plots in amplitude, db, phase, etc).

#. HTML report creation (by combining all plots and text products onto a single standalone HTMl to be shared or stored).

This pipeline has been written under the assumption that the user will be running it in CASA or in an environment that provides the casatasks and casatools modules. The instructions below assume that the pipeline is being run inside CASA.

Pipeline interface
##################

The pipeline has been written with a simple command line interface that expects two mandatory arguments from the user, the name of the dataset to be processed (be it an MS or an ASDM) and a reference antenna for the calibration stage, several execution customization options are also available a simple help can be accessed with the ``-h`` flag:

.. code-block::

    CASA <1>: !beamcut-reduction-pipeline -h

    #####################################################################################################################################
    ###  Welcome to the AstroHACK BeamCut reduction pipeline                                                                          ###
    #####################################################################################################################################

    usage: beamcut-reduction-pipeline [-h] [-r ROOT_NAME] [-q QUACK_NCHAN] [-f BEAMCUT_FIELD] [-s SPW] [-a ANTENNA] [-n NCORES]
                                      [-m MEMORY_PER_CORE] [-o] [-d DATA_COLUMN] [-y]
                                      [--starting-stage {calibration,extract_pointing,extract_holog,beamcut,exports,report}]
                                      [--dpi DPI] [--plot-pointing] [--exclude-bad-antennas EXCLUDE_BAD_ANTENNAS] [--reimport-asdm]
                                      filename refant

    Beam cut reduction pipeline

    positional arguments:
      filename              Path to the input dataset to process.
      refant                Reference antenna for calibration

    options:
      -h, --help            show this help message and exit
      -r ROOT_NAME, --root-name ROOT_NAME
                            Root name for the products of the pipeline, default is ms_name without extension
      -q QUACK_NCHAN, --quack-nchan QUACK_NCHAN
                            Number of channels to quack at the edge of the spectral window (default is 4)
      -f BEAMCUT_FIELD, --beamcut-field BEAMCUT_FIELD
                            Field Id or name of the beam cut data (default is to determine it from data)
      -s SPW, --spw SPW     Select SPWs for which to produce beam cuts, for a list use comma separated values with no spaces, e.g.:
                            '0,1,2', default is all
      -a ANTENNA, --antenna ANTENNA
                            Select antennas for which to produce beam cuts, for a list use comma separated values with no spaces, e.g.:
                            'ea01,ea02', default is all
      -n NCORES, --ncores NCORES
                            Number of cores to use, default is 4
      -m MEMORY_PER_CORE, --memory-per-core MEMORY_PER_CORE
                            Memory per core to use, default is 10GB
      -o, --overwrite       Overwrite existing files if found
      -d DATA_COLUMN, --data-column DATA_COLUMN
                            Data column to be extracted from MS, default is CORRECTED_DATA
      -y, --assume-yes      Assume yes on proceed.
      --starting-stage {calibration,extract_pointing,extract_holog,beamcut,exports,report}
                            Starting stage in which to start processing (default: calibration).
      --dpi DPI             Dots Per Inch for plotting, default is 300
      --plot-pointing       Plot antenna pointing, default is False
      --exclude-bad-antennas EXCLUDE_BAD_ANTENNAS
                            Exclude antennas with bad data, for a list use comma separated values with no spaces, e.g.: 'ea18,ea01',
                            default is None.
      --reimport-asdm       Forcefully re-import the asdm file is the ms already exists (default: False)



Calibration stage
#################

With a reference antenna chosen it is now time to run the beamcut pipeline.
The first step of the pipeline is to check whether the data is an ASDM or an MS and if it is an ASDM if it needs to be imported into an MS. With an MS in hands the pipeline proceeds to fetching some metadata from it and then prints a summary of what it has found and which parameters it will use for calibration and further data reduction, e.g.:

.. code-block::

    CASA <4>: !beamcut-reduction-pipeline X002.ms ea05

    #####################################################################################################################################
    ###  Welcome to the AstroHACK BeamCut reduction pipeline                                                                          ###
    #####################################################################################################################################

    2026-07-20 16:06:28	INFO	msmetadata_cmpt.cc::open	Performing internal consistency checks on X002.ms...
    2026-07-20 16:06:28	INFO	MSMetaData::_computeScanAndSubScanProperties 	Computing scan and subscan properties...

    Beam cut reduction parameters:
        filename              => X002.ms
        refant                => ea05
        root_name             => None
        quack_nchan           => 4
        beamcut_field         => 3
        spw                   => all
        antenna               => all
        ncores                => 4
        memory_per_core       => 10GB
        overwrite             => False
        data_column           => CORRECTED_DATA
        assume_yes            => False
        starting_stage        => calibration
        dpi                   => 300
        plot_pointing         => False
        exclude_bad_antennas  => None
        reimport_asdm         => False
        is_asdm               => False
        msname                => X002.ms
        delay_cal_name        => X002.dcal
        bandpass_cal_name     => X002.bcal
        gain_cal_name         => X002.gcal
        point_name            => X002.point.zarr
        holog_name            => X002.holog.zarr
        beamcut_name          => X002.beamcut.zarr
        exports_name          => X002.exports
        report_name           => X002-report.html
        calibration_scans     => 2,11
        beamcut_scans         => 5,9
        quacked_spw_selection => 0~7:4~60
        parallel              => True


    Proceed? <(Y)es/(N)o>:


The check before proceeding can be suppressed by adding the ``-y`` option to the call, e.g.:

.. code-block::

    CASA <4>: !beamcut-reduction-pipeline X002.ms ea05 -y

The code will then proceed through the calibration steps:

#. Delay calibration with ``gaincal(gaintype="K")``.

#. Bandpass calibration with ``bandpass``.

#. Amplitude and Phase calibration with ``gaincal(calmode="AP")``.

#. Application of all the previously computed calibration tables with ``applycal``.

Beam cut processing
###################

After the beam cut data has been calibrated the pipeline then proceeds to run Astrohack's functions:

#. `extract_pointing <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_pointing/index.html>`_: Extract pointing data from the MS onto a ``.point.zarr`` file that is arranged in a convenient way for further processing.

#. `extract_holog <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_holog/index.html>`_: Identify moving antennas from the pointing data, then extract visibilities from the ms for these antennas and finally match the pointing data to the visibilities.

#. `beamcut <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/beamcut/index.html>`_: Separate the visibility data onto the different beam cuts present in the data, determine the direction of the beam cuts, fit multiple gaussians to the beam cut to try to determine the beam parameters like Primary beam offset and FWHM & first side lobe ratio.

By default the astrohack stages are run in parallel, (ncores =4), this can be changed by explicitly giving a number of cores e.g. ``--ncores 5``. For a serial run, one should use ``--ncores 0`` or ``--ncores 1``. In case of failures or there is a desire to re run the pipeline from a particular stage, the user can then use option ``--starting-stage``.
For more details on the beam cut processing stages there is the more detailed `beamcut tutorial <https://astrohack.readthedocs.io/en/stable/tutorials/beamcut_tutorial.html>`_.

Exports and Report stages
#########################

After the astrohack data files are created, the pipeline then proceeds to execute the exporting functions from the associated Python classes:

#. `AstrohackPointFile.plot_array_configuration <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/point_mds/index.html#plot_array_configuration>`_: Single plot displaying the array configuration at observation time.

#. `AstrohackBeamcutFile.plot_in_amplitude <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/beamcut_mds/index.html>`_: Plots of the beam cuts in Amplitude with an overlay of the multi gaussian fit.

#. `AstrohackBeamcutFile.plot_in_db <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/beamcut_mds/index.html>`_: plots of the beam cuts in amplitude expressed in dBs normalized to the brightest amplitude correlation.

#. `AstrohackBeamcutFile.plot_in_phase <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/beamcut_mds/index.html>`_: Plots of the beam cuts in phase with an overlay of the multi gaussian fit.

#. `AstrohackBeamcutFile.plot_lm_offsets <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/beamcut_mds/index.html>`_: Plots of the lm offsets for the antennas during the beam cut observations.

#. `AstrohackBeamcutFile.export_report <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/beamcut_mds/index.html>`_: Create an ASCII report of the fitted parameters of the multi gaussian fit.

After the production of these export products the pipeline then creates a standalone HTML report with all of them that can then be stored or shared without the need to carry any extra data, an example of such a report can be seen `here <../example-beamcut-u-band-report.html>`_.