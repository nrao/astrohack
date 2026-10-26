ALMA Near Field holographies support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current support for ALMA near field holographies is limited.  It is
possible to produce aperture images corrected for the near field
effects, but panel adjustments from them have not been tested.



Extracting the holography data from the ASDM
============================================

ALMA near field holographies are stored in ASDMs. The function
extract_alma_nf_holog reads them with `pyasdm
<https://github.com/casangi/pyasdm>`_, which is installed with
astrohack. It calibrates the total power data against the calibration
subscans, matches it to the pointing data and creates a .holog.zarr
file that contains the data necessary to produce the gridded beam
image and the apertures.

.. code:: python

    from astrohack import extract_alma_nf_holog

    holog_mds = extract_alma_nf_holog(asdm_name='uid___A002_Xad52b4_X3',
                                      holog_name='almanf.holog.zarr',
                                      overwrite=True # depending on your preferences
                                      )

The calibration fits, the length of the calibration cycle and the
time averaging interval can be chosen, and the calibrations can be
plotted to png files; see the `API
<https://astrohack.readthedocs.io/en/stable/api.html>`_.

Producing apertures
===================

In astrohack the function that produces apertures from .holog.zarr
files is called holog. This function grids the beam image from the
data and then Fourier transforms (FT) it to produce the apertures.  In
the case of near field holographies after the first FT subsequent
transformations are applied to correct for the non-fresnel terms
followed by a step of correction of the aperture phase from the NF
effects.

The apertures can be produced and then subsequently plot with a simple
python script. holog uses the tower distance and feed focus offset of
the pad on which the antenna was positioned during the holography
measurement, which defaults to the antenna's station in the ASDM; the
parameter **alma_osf_pad** overrides it. We set to parallel to False as
holog parallelizes holographies per antenna and ALMA nf holographies
contain a single antenna. **to_stokes** must be False: the
polarization axis of near field data holds the signal (I) and the
reference power (R2), which cannot be converted to Stokes parameters.

After the call to holog we are left with an object that contains the
gridded beam images as well as the apertures. This object is reflect
on disk by a .image.zarr file. From this object we can acces the
plotting functions, plot\textunderscore beams and plot\textunderscore
apertures.

.. code:: python

    from astrohack import holog

    image_mds = holog(holog_name='almanf.holog.zarr',
    		  image_name='almanf.image.zarr',
    		  to_stokes=False,
    		  parallel=False,
    		  overwrite=True # denpending on your preferences
    		  )

    # Plotting gridded beam images
    image_mds.plot_beams('almanf_plots', angle_unit='asec', display=False)
    # Plotting aperture images
    image_mds.plot_apertures('almanf_plots', polarization_state=['I'], display=False)

If after a while you want to interact once more with the image_mds
object you can use the open_image function.

.. code:: python

    from astrohack import open_image

    image_mds = open_image('almanf.image.zarr')

For more information on the API of holog and other astrohack functions
can be found `here
<https://astrohack.readthedocs.io/en/stable/api.html>`_. More detailed
instruction on how to use the plotting functions can be found on our
`visualization tutorial
<https://astrohack.readthedocs.io/en/stable/visualization_tutorial.html>`_.

Caveat
======

On uid___A002_Xad52b4_X3 (DV08, 104 GHz) the aperture phase left by
the phase fit corresponds to a surface error of about 20 um rms. A
concentric ripple of about 15 um remains within about a metre of the
dish edge; its origin is not yet understood.
