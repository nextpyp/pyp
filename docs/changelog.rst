=========
Changelog
=========

**Legend**: :fa:`star,text-success` New feature, :fa:`plus-square,text-primary` Improvement, :fa:`bug,text-danger` Bug fix

v0.6.3 (2/10/2024)
******************

   :fa:`plus-square,text-primary` Allow import of clean particles obtained after 3D classification into pre-processing block.

   :fa:`plus-square,text-primary` Stop saving unnecessary metadata files during constrained refinement.

   :fa:`plus-square,text-primary` Implement particle list picker that was missing from some import blocks.

   :fa:`bug,text-danger` Fix bug in function used to export sessions to .star format.

   :fa:`bug,text-danger` Fix inconsistencies in the determination of parameter changes between consecutive runs.

   :fa:`bug,text-danger` Do not try to launch external programs for sub-tomogram averaging after particle extraction.

   :fa:`bug,text-danger` Fix issue with missing metadata entries during tilt-series re-processing.

   :fa:`bug,text-danger` Correctly discard particles that are too close to gold fiducials (tomography pipeline).

v0.6.2 (2/01/2024)
******************

   :fa:`plus-square,text-primary` Expose additional parameters for frame alignment when using MotionCor3.

   :fa:`plus-square,text-primary` Remove unnecessary tabs from tomography refinement blocks.

   :fa:`plus-square,text-primary` Display slurm job launch information in the logs window.

   :fa:`plus-square,text-primary` Allow users to specify resources for the launch task on the Sessions side.

   :fa:`bug,text-danger` Fix bugs in parameter definitions when running movie frame alignment.

   :fa:`bug,text-danger` Fix bugs in the management of slurm's GRES options when submitting jobs to the scheduler.

   :fa:`bug,text-danger` Fix bug with movie drifts being deleted from the database when tilt-series were re-processed.

v0.6.1 (1/30/2024)
******************

   :fa:`star,text-success` Produce metadata for 3D visualization using `ArtiaX <https://github.com/FrangakisLab/ArtiaX>`_ for all refinement blocks. See the :doc:`user guide<guide/chimerax_artiax>` for details.

   :fa:`star,text-success` Enable dose weighting and magnification correction options during frame alignment and averaging.

   :fa:`star,text-success` Allow specification of SLURM account for all job types to improve portability.

   :fa:`plus-square,text-primary` Expose full set of options when using MotionCor3 for frame alignment.

   :fa:`plus-square,text-primary` Allow specification of GPU resources using Gres option to allow selection of specific types of graphics cards, e.g., gpu:A100:1.

   :fa:`plus-square,text-primary` Add support for multiple date formats when reading metadata from .mdoc files.

   :fa:`plus-square,text-primary` Add support for .gain reference files and automatically resize corresponding .eer movies in data import blocks.

   :fa:`bug,text-danger` Fix issue when handling *.tif files that have a *.tiff extension.

   :fa:`bug,text-danger` Fix issue with multiprocessing library when using NFS mounts as local scratch.

   :fa:`bug,text-danger` Fix bug in single-particle sessions when using unbinned images for 2D classification.

   :fa:`bug,text-danger` Fix bug when picking particles using neural network-based approach on non-square tomograms.

   :fa:`bug,text-danger` Fix bug that prevented GPU jobs from running because the jobs were sent to the CPU queue.

v0.6.0 (1/21/2024)
*******************

   :fa:`star,text-success` Allow use of `MotionCor3 <https://github.com/czimaginginstitute/MotionCor3>`_ for movie frame alignment (GPU required).

   :fa:`star,text-success` Allow use of `AreTomo2 <https://github.com/czimaginginstitute/AreTomo2>`_ for tilt-series alignment and reconstruction (GPU required).

   :fa:`star,text-success` Allow use of `Topaz <https://github.com/tbepler/topaz>`_ for 2D particle picking and 3D denoising (GPU recommended).

   :fa:`star,text-success` Produce .bild files after each refinement iteration for 3D visualization in Chimera/ChimeraX.

   :fa:`star,text-success` Automatic determination of CTF handedness during pre-processing of tilt-series.

   :fa:`plus-square,text-primary` Allow mix-and-match of IMOD and AreTomo2 for tilt-series alignment and tomogram reconstruction.

   :fa:`plus-square,text-primary` Automatically submit jobs to a GPU partition when running tasks that require GPU acceleration.

   :fa:`plus-square,text-primary` Display version number and amount of allocated memory at the beginning of every job.

   :fa:`plus-square,text-primary` Change default memory allocation for launch task to 4GB and add Resources tab to all data import blocks.

   :fa:`plus-square,text-primary` Simplify Resources tab by hiding unnecessary parameters depending on the block type.

   :fa:`plus-square,text-primary` Implement GPU resource management policies for slurm and standalone modes.

   :fa:`plus-square,text-primary` Show per-particle score distribution for all tomography refinement blocks and improve plot layout.

   :fa:`plus-square,text-primary` Allow use of slurm's GRES (generic resource scheduling) when submitting jobs to a cluster.

   :fa:`bug,text-danger` Fix OOM error when running constrained refinement using a single thread.

   :fa:`bug,text-danger` Fix error in particle filtering blocks when no particles are left in a given micrograph/tilt-series.

   :fa:`bug,text-danger` Fix issue in tomography sessions when .mdoc files are not used to import metadata.

   :fa:`bug,text-danger` Fix bug when exporting sub-tomograms for use in external programs.

   :fa:`bug,text-danger` Update systemd script to improve robustness during program restart.

   :fa:`bug,text-danger` Fix issues with cancellation of jobs in standalone mode.

   :fa:`bug,text-danger` Fix discrepancy with gain reference rotation/flips between data import and pre-processing blocks.

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