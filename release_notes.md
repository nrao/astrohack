## Release Notes: Astrohack v0.4.1

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
