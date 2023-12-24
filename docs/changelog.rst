=========
Changelog
=========

**Legend**: :fa:`star,text-success` New feature, :fa:`plus-square,text-primary` Improvement, :fa:`bug,text-danger` Bug fix

v0.5.4 (12/12/2023)
*******************

   :fa:`star,text-success` Add `MotionCor3 <https://github.com/czimaginginstitute/MotionCor3>`_ wrapper for movie frame alignment (requires GPU).

   :fa:`star,text-success` Add `AreTomo2 <https://github.com/czimaginginstitute/AreTomo2>`_ wrapper for tilt-series alignment and reconstruction (requires GPU).

   :fa:`star,text-success` Add `Topaz <https://github.com/tbepler/topaz>`_ wrapper for 2D particle picking and 3D denoising (GPU recommended).

   :fa:`star,text-success` Produce .bild files after each refinement iteration for 3D visualization in Chimera.

   :fa:`plus-square,text-primary` Allow mix-and-matching IMOD and AreTomo2 for tilt-series alignment and tomogram reconstruction.

   :fa:`plus-square,text-primary` Automatically submit jobs to a GPU partition when running tasks that require GPUs.

   :fa:`plus-square,text-primary` Display version number and amount of allocated memory at the beginning of every job.

   :fa:`plus-square,text-primary` Change default memory allocation for launch task to 4GB and add Resources tab to data import blocks.

   :fa:`plus-square,text-primary` Simplify Resources tab for all blocks by hiding unnecessary parameters.

   :fa:`plus-square,text-primary` Allow use of pre-computed results from reference-based refinement in tomography and classification tutorials.

   :fa:`bug,text-danger` Fix OOM error when running constrained refinement using a single thread.

   :fa:`bug,text-danger` Fix error in particle filtering blocks when no particles were left on a given micrograph/tilt-series.

   :fa:`bug,text-danger` Fix issue in tomography sessions when not using .mdoc files to import metadata.

   :fa:`bug,text-danger` Fix bug when exporting sub-tomograms for use in external programs.

   :fa:`bug,text-danger` Fix various other small bugs.

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