=========
Changelog
=========

**Legend**: :fa:`star,text-success` New feature, :fa:`plus-square,text-primary` Improvement, :fa:`bug,text-danger` Bug fix

v0.5.3 (11/25/2023)
*******************

   :fa:`star,text-success` Implement interactive measuring tool for micrographs and tomograms.

   :fa:`star,text-success` Allow multiple sessions when user login mode is enabled.

   :fa:`plus-square,text-primary` Sort classes in increasing order in Class View panel.

   :fa:`bug,text-danger` Fix issues when limiting total number of tasks in slurm scheduler.

v0.5.2 (11/18/2023)
*******************

   :fa:`star,text-success` Add support for PACEtomo tilt-series in streaming Sessions.

   :fa:`plus-square,text-primary` Parallelize reconstruction step during 3D classification for faster speeds.

   :fa:`plus-square,text-primary` Add new options to flip maps in post-processing block.

   :fa:`plus-square,text-primary` Simplify installation instructions and setup process.

   :fa:`bug,text-danger` Fix issue with location of executables for neural network-based particle picking.

   :fa:`bug,text-danger` Fix issue with re-calculation of binned tomograms when reconstruction parameters change.

   :fa:`bug,text-danger` Fix issue with re-calculation of particle coordinates when no particles were found.

   :fa:`bug,text-danger` Correctly display particle size in tomography pre-processing block statistics.

v0.5.1 (11/04/2023)
*******************

   :fa:`star,text-success` Import frame tilt-series data using mdoc files produced by PACEtomo.

   :fa:`plus-square,text-primary` Allow typing iteration number in navigation bar for refinement blocks.

   :fa:`plus-square,text-primary` Show refinement/bundle IDs in ``Per-particle Score`` and ``Exposure Weights`` tabs for refinement blocks.

   :fa:`bug,text-danger` Fix issue with display of tomograms with arbitrary thickness.

   :fa:`bug,text-danger` Fix broken CLI commands and update CLI tutorials.

v0.5.0 (10/26/2023)
*******************

   This was the first release of nextPYP.