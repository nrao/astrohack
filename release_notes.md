# v0.10.0

This release brings a new functionality to astrohack, the ability to
analyze beam cuts. Beam cuts are extremely useful as they are in a
sense a type of very poor resolution holography while but also are
very cheap to obtain in terms of time.

## General

- Documentation has been slightly reorganized.
  - Tutorials are now found in a dedicated section of the read the
    docs pages.
  - Remaining pages are now all found in one directory inside the docs
    root.
    
- A script for cloudflare file uploading has been added to
  astrohack/etc/test_data_upload, it relies on the user setting
  cloudflare account info and tokens as environment variables

- Added a deep checking tool to verify if 2 mdses are equal, it works
  with both regular dict based mdses and newer data tree based mdses.

- Python files containing IO, and in memory data storage functions and
  classes that are user facing have been moved to a sub directory
  called IO.

- Fixed numpy version to <= 2.2 as Numba does not support numpy 2.3 or
  2.4.

- Tutorial notebooks now close their clients at the end of execution.


## Beam cuts

- `beamcut` is a new astohack function to process .holog.zarr files
  that contain beam cut measurements.

- The newly introduced `AstrohackBeamcutFile` class uses Xarray Data
  Trees to store that in memory and uses Xarray interfaces to save it
  to disk.

- `open_beamcut` is a new astrohack function to process .beamcut.zarr
  files created by `beamcut`.

- A beam cut calibration pipeline can be found at
  astrohack/etc/beamcuts.

- A beam cut tutorial has been included and can be found on the new
  tutorial section of astrohack.readthedocs.io.


## Holography

- `extract_holog` now stores scan information in the output
  .holog.zarr files. This change was made to simplify processing of
  beam cuts.

# v0.9.4

This release contains only fixes to the readthedocs pages.

# v0.9.3

This release contains no changes to the code itself, but it contains needed changes to address the change in the URL for the Astrohack repo.

# v0.9.0

This releases brings quite a few major changes, with the biggest
change being the tentative full support for the ngVLA prototype
holography commissioning tests. A summary of the changes can be found
below.

- General:

  - Telescope objects:
  
    - The telescope class has been split up.
    
    - Cassegrain telescopes are now supported by a the
      RingedCassegrain class.
      
    - The ngvla prototype is supported by the NgvlaPrototype class.
    
    - Telescope objects may now be initialized with the
      get_proper_telescope function.
      
  - Changed import scopes:
  
    - Now only user intended functions are available at the package
      level.
    
  - Increased test coverage:
  
    - Now 2D gridding and 1D interpolation functions are tested during
      continuous integration.

    - Opening routines for all AstrohackDataFiles are now tested
      during continuous integration.
    
  - Bug fixes:
  
    - The creation of file names based on input files now uses
      str.removesuffix as .rstrip has behaviors that were leading to
      weirdness in output file names.
      
    - check_if_file_can_be_opened now checks if path exists before
      doing anything.

- Holography:

  - Panel:
  
    - part of the work in processing the apertures done by the
      AntennaSurface class has now be ported to the new telescope
      classes as some of the processing differs between Cassegrain and
      the ngVLA due to their different optics.
      
    - Improved PolygonPanel class has been brought to production, this
      class is used to represent panels that are represented by
      polygons, such as the ngVLA panels.
      
    - Phase to deviation correction for the ngVLA is computed using a
      Quintic Pseudo Spline (QPS) representation of the surface to
      compute the cosine of the surface normal to the direction of
      boresight.
      
    - Aperture plot orientation and panel sample coordinates reference
      frames have been reconciled yeilding cohesive coordinate frames
      between panel points and the points in aperture plots.

   - Extract_holog:

     - Found a bug in the interpolation of pointings onto visibility
       times that allowed for NaNs in the interpolated pointings.

     - Added a new method to interpolate pointings onto visibility
       times using a gaussian convolution, this is also the fallback
       behavior when NaNs are detected in the linear interpolation of
       the pointings.

   - Extract_pointing:

     - extract_pointing now raises a warning when the pointing tables
       have irregular sampling times.
    

# v0.8.0

This release includes the following changes:

- Holography:
  - There is now a observation summary attached to each holography xarray dataset inside a mds (holog.zarr, image.zarr and panel.zarr).
  - This summary can be printed and exported to an ASCII file through a method called .observation_summary of the respective mdses.
  - This allows for the decoupling of automatically defined beam gridding characteristics between DDIs.
  - This change breaks backward compatibility with files from versions before this one.
  - Checking if a file can be opened is now performed at the start of processing and when opening files.
  
- General:
  - Added a Z scale control to the image_compare_tool.

# v0.7.1

This release addresses a bug that disabled the 'AIPS' colormap.

# v0.7.0

This release includes some features for astrohack.

- General
  - Astrohack is now fully Black compliant
  - Astrohack now supports python versions 3.11, 3.12 and 3.13.
  - Astrohack now uses the toolviper set of configurations for testing and code coverage. 
  - Astrohack now uses Numpy V>2.
  - Astrohack is now compatible with the latest releases of Dask.

- Holog:
  - holog now fits Zernike polynomials to the apertures before converting them to Stokes.
  - The Zernike polynomial fitting can be controlled by chosen the highest order of polynomials to be fitted using parameter zernike_n_order.
  - The highest order of Zernike polynomials that can be fitted is N = 10 (66 polynomial coefficients in total).
  - The Zernike polynomial fit can also be used for phase fitting, this new feature can be chosen by using the parameter phase_fit_engine.
  - Zernike polynomial phase fitting should be used with orders 2-4 as higher orders may start fitting structures in the aperture plane that are caused by the panels.
  - It is possible to retrieve the values of the fitted Zernike polynomial coefficients by using the image_mds method export_zernike_fit_results.
  - It is possible to plot the Zernike polynomial model along with the fitting results by using the image_mds method plot_zernike_model.
  - Both phase fitting engines are now almost fully covered by tests.
  - Perturbation phase fitting (classic AIPS like algorithm) now can deal with no square pixels or images.

# v0.6.2

- General:
  - Fixed issues with problematic scape sequences.
  - Moved user documentation pages from the wiki to the readthedocs.
  - Misc improvements to readthedocs page.

- FITS Comparison tool:
  - Fixed bug caused by a typo.
  - Added new types of plot: Reference image plot and unresampled data plot.
  - Added statistics to the headers of all plots.
  - Created a function to extract RMS values from an FITSImage obj: rms_table_from_zarr_datatree.

- Panel:
  - Fixed a bug that prevented old files with no polarization state information from being opened.

# v0.6.1

General Changes:
- Dropped support for Python 3.9 as the latest release of xarray drops support for this python version
- Added support for Python 3.12, this python version was not support by python-casacore for a long time but that is no longer the case.

Panel:
- Introduced a new parameter called exclude_shadows.
  - This parameter activates using a mask which also excludes the shadows caused by the quadrupole holding the secondary.

Image Comparison Tool:
- This release introduces a simple FITS comparison tool to gauge the differences between apertures taken at different epochs and software such as AIPS.

# v0.6.0

This release includes the release of the Cassegrain ray Tracing tool.
More information about this tool can be seen at our readthedocs page:
astrohack.readthedocs.io

# v0.5.11

This release bring a bug fix for locit:
- When a delay fit failed locit was crashing, there is now error trapping around the fitting routines giving out a warning when a fit fails.

# v0.5.10

This is a patch release, that brings a bug fix to HOLOG:

- After the phase fitting process the phase corrections were subtracted from the phase image without wrapping the phase to the - $\pi$ to $\pi$ interval, this has now been fixed by the addition of phase wrapping at the end of the phase fitting process.
- A test has been added to panel to detect images from HOLOG that have been produced with astrohack versions previous to this one to trigger a phase wrapping of the input phase image before computing deviations.


# v0.5.9

This release brings some improvements to LOCIT:
- It is now possible to use 'RL', 'LR', 'XY' and 'YX' to describe the polarization to be used, 'both' is still accepted as well.
- LOCIT no longer crash in the event of missing data, it will produce a warning instead.
- If all data for an antenna DDI combination is missing during processing due to filtering (e.g. low elevation data) LOCIT will now produce a warning.

# v0.5.8

This release contains a few changes to locit, plus a new method for installing astrohack in NRAO machines.
- LOCIT:
  - Delay plots with the fitted delay model now display the delay model residuals.
  - Improved selection of X and Y limits on delay, and sky cover plots.
  - Table that contains position corrections now present station numbers and fit RMS in degrees.
  - Added a method to the position_mds called export_to_parminator to export locit results to a parminator file.
  - Changed formatting of values in the table that contains position corrections.
- Installation under a VENV:
  - Installation script.
  - CASA fringe fit and phase gain script for pre locit reduction.
  - LOCIT execution script.
  - post locit parminator export script
  - installation instructions on the wiki
  - Execution instructions on the wiki

# v0.5.7

This release contains a miscellany of minor changes and a few bug fixes.

- extract_holog: 
  - Corrected the time interpolation of pointing for the case in which the pointing table is  sampled less frequently than the visibilities (pre 2021 VLA).
  - Changed test on needing to fix the pointing table to when there are no mapping antennas rather than using a date.
  - Changed the default of baseline_average_nearest parameter from 'all' to 1.
- locit:
  - Fixed the case where field_ids were taken from the negative phases but assumed to be the same as the positive phases which cause an access error due to array length.
  - Reference antenna is now always marked on the output table with antenna position corrections.
  - The antenna position corrections table is now printed when position_mds.export_locit_fit_results is called.
- Miscellanious:
  - Fixed the formatting of the API documentation in readthedocs for a few tasks where line breaks were causing a formatting breakdown.
  - Updated github workflows to use V4 artefacts rather than V3 artefacts.

# v0.5.6

This release contains a series of minor improvements.

Panel:
- Amplitude cutoff based on noise maximum and threshold on excluding data.
- RMS in screw file is now always quoted in mm as well as the unit chosen.
- Antenna gain is now computed using the expected FWHM of the primary beam.

Extract_holog:
- It is now possible to exclude antennas that have bad data.

Miscelanious:
- Fixed a typo in notebooks referencing apply_mask, a defunct parameter of holog, that caused the notebooks to not complete execution.

# v0.5.5

This release contains the following changes:

- It is now possible to specify a specific amplitude cut off for any antenna and ddi combination.
- Apertures are no longer clipped to allow for the analysis of the noise outside the dish image.
- Aperture amplitude plots now display simple amplitude and noise statistics.

# v0.5.4

* New Feature:

- Added antenna surface RMS estimates before and after screw adjustments to Gain table and screw adjustment files.

# v0.5.3

* Bug fixes

The U and V axes of the aperture FITS files produced by image_mds.export_to_fits 
had the wrong values, this is now fixed.

extract_holog now works with int ddi input


* Improvements
Improved resolution of the display of frequency in the file containing the screw adjustments

Improved header definition of the exported ASCII screw adjustment file

gaussian convolution is now the default gridding method in holog.

Panel fitting and correction have been optimized with execution times being 10-50% faster depending on model.

holog_obs_dict json files are no longer created as hidden files (. file name)

Decreasead overall verbosity of tasks.

* New Features

New method for image_mds file that export phase fitting results: export_phase_fit_results

New method for panel_mds file that export estimated antenna gains at multiple frequencies: export_gains_table

New model for panel fitting, flexible, based on the model of the same name found in AIPS.

inspect_holog_obs_dict now also opens the holog_obs_dict inside of holog_mds files.


* Miscelanious changes

Replaced dependency of graphviper with toolviper

# v0.5.1

- New method of gridding the visibilities onto a beam. This method uses the convolution of the visibilities with a gaussian kernel with a size similar to the primary beam. This is a fast release to attend to VLA validation needs, no documentation on the gaussian convolution gridding is provided at this point.

# v0.5.0

- General
  - Dropped support for python 3.8, prompted by drop of support for 3.8 in graphviper
  - Pixel coordinates now reflect the middle of the pixel rather than its left edge
  - U and V axes are now always in meters in data files rather than in kilo lambda

- Combine
  - Fixed weighted combining of DDIs

- Holog
  - Refactoring for better separation between beam gridding and aperture generation
  - Padding with zeros now produces images that have a size of 2^(ceiling(log2((original_size*padding_factor)))
  - image_mds.plot_apertures can now choose polarization state
  - image_mds.plot_beams plots are now split by polarization state
  - Beam gridding using a gaussian kernel
  - Bug fix: Calling holog with a grid_size and cell_size no longer overwrite these values in the input .holog.zarr file

- Panel
  - Screw adjustments file now contains the method use to fit the panel and if it was a fallback
  - Polarization state selection with I being the default
  
- ALMA Near field support (Alpha)
  - ALMA NF ASDM filler to the .holog.zarr format
  - Selection of ALMA OSF pad in holog to determine distance and defocusing
  - Beam apodization
  - Non-fresnel terms in aperture generation (Not fully correct)
  - Phase near field corrections (Not fully correct)
  - GILDAS\CLIC like phase fitting including astigmatism


# v0.4.3

- Fixed issue with dropping antenna messing with antenna indexing.
- Updated default calculation for grid and cell size.
- Filter out SYSTEM_CONFIGURATION scans.
- Add time smoothing of the visibilities and pointing data.

# v0.4.2

- Fixed missing import of extract_pointing in memory test.

- Polarization selection in Panel now uses xarray tools.

- Panel can now select a polarization state.
  * Previously panel picked the 0th element in the polarization axis of the aperture for doing the work, but this was not robust if the data is not on the stokes order I, Q, U, V.
  * Panel is now allowed to choose which polarization state to pick from the data, for example running on RR or XX when stokes parameters are not available.

- Generalize creation of default file name.
  * When output file names were not given the code created a default name based on the input name. This was repeated code among the different modules and has now been generalized to a single agnostic function.

- Parameter no longer checking fails in Github Actions in MacOS

- Parameter checking validation added to plotting API.
- Fix bug where client didn't load worker logging plugin when logger parameters were not specifically passed to `local_client(...)` function.

- The function `extract_point(...)` now includes an option to drop a given antenna(s). The changes have been propagated into `extract_holog(...)` as well. The new input parameter to `extract_point(...)` is `exclude = []` and takes a single or list of antennas.

  * A warning has been added to `extract_point(...)` to notify the user when the antenna data length ensemble has a fractional error of more than 1%, ie. this should be essentially zero.

  * Dropped antenna info in recorded in the pointing dataset object.
