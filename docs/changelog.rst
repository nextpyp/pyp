=========
Changelog
=========

**Legend**: :fa:`star,text-success` New feature, :fa:`plus-square,text-primary` Improvement, :fa:`bug,text-danger` Bug fix

v0.7.0 (6/24/2024)
******************

   :fa:`star,text-success` Incorporated external program tomoDRGN for continuous variability analysis.

   :fa:`star,text-success` Incorporated external program cryoDRGN and cryoDRGN-ET for continuous variability analysis.

   :fa:`star,text-success` Incorporated external program MemBrain-Seg for segmenting tomograms.

   :fa:`star,text-success` Incorporated external program IsoNet for denoising tomograms.

   :fa:`star,text-success` Incorporated external program cryoCARE for denoising tomograms.

   :fa:`star,text-success` Incorporated CTF estimation using CTFFIND5.

   :fa:`star,text-success` nextPYP can now submit slurm jobs using individual OS user accounts.

   :fa:`star,text-success` Implement ab-initio refinement strategy for tomography pipeline.

   :fa:`star,text-success` Add option to do automasking during refinement.

   :fa:`star,text-success` Molecular pattern mining and particle localization (MiLoPYP).

   :fa:`star,text-success` Start saving refinement metadata in cisTEM's binary format (2x speedup during reference-based refinement).

   :fa:`star,text-success` Add support for beam-tilt refinement and correction (as implemented in cisTEM).

   :fa:`plus-square,text-primary` Allow selection of blocks using a single click in the Run jobs menu.

   :fa:`plus-square,text-primary` Move options to reshape images into squares from Reconstruction to Tilt-series alignment tab.

   :fa:`plus-square,text-primary` Add option to import existing tilt-series alignments from IMOD files (*.xf and *.tlt).

   :fa:`plus-square,text-primary` Add option to save particle stacks for use in external programs.

   :fa:`bug,text-danger` Fix bug in navigation bar for refiement blocks when multiple classes were used.

   :fa:`bug,text-danger` Fix bug that prevented launch task parameters from being used when launching sessions.

   :fa:`bug,text-danger` Fix bug where incorrect binning was applied when picking virions manually.

   :fa:`bug,text-danger` Fix bug where tomogram dimensions and binning were not properly updated.

   :fa:`bug,text-danger` Fix bug that prevented the recalculation of tomograms when using AreTomo if the reconstruction parameters changed.

v0.6.5 (4/6/2024)
******************

   :fa:`plus-square,text-primary` Update format of logger messages to more clearly show the nextPYP version and resources assigned to each job.

   :fa:`plus-square,text-primary` Use same tilt-axis angle convention for aligning tilt-series using IMOD and AreTomo2.

   :fa:`bug,text-danger` Prevent error during tilt-series alignment with AreTomo2 when number of patches = 1.

   :fa:`bug,text-danger` Fix bug in command line interface that ocurred when launching constrained refinement.

   :fa:`bug,text-danger` Fix bug that was causing the server startup routines to be called during the CLI processing of the configuration file.

   :fa:`bug,text-danger` Fix bug that ocurred when retrieving metadata from mdoc files.

   :fa:`bug,text-danger` Fix bug when trying to retrieve tilt-series metadata from failed runs.

   :fa:`bug,text-danger` Fix conflicts with library paths when running external executables.

v0.6.4 (3/24/2024)
******************

   :fa:`plus-square,text-primary` Implement mechanism to isolate logs from failed jobs in the Logs tab.

   :fa:`plus-square,text-primary` Add support for project names with special characters.

   :fa:`plus-square,text-primary` Remove many commonly used parameters from the advanced category.

   :fa:`plus-square,text-primary` Add progress bars during export of metadata to .star format.

   :fa:`plus-square,text-primary` Allow export of particle coordinates from streaming sessions.

   :fa:`plus-square,text-primary` Check that .order files have the same number of entries as images in the tilt-series.

   :fa:`bug,text-danger` Fix bugs when reading metadata from *.mdoc files.

   :fa:`bug,text-danger` Prevent dragging of multiple connections from block outputs in project view.

   :fa:`bug,text-danger` Fix bug when managing GPU resources in standalone mode.

   :fa:`bug,text-danger` Fix bug when using grouping of frames during movie processing.

   :fa:`bug,text-danger` Fix bug in single-particle pipeline during hot pixel removal.

   :fa:`bug,text-danger` Fix bug in Table view that caused content to overlap when resizing columns.

   :fa:`bug,text-danger` Always export metadata in .star format to current project directory (user specified location is no longer supported).

v0.6.3 (3/01/2024)
******************

   :fa:`plus-square,text-primary` Allow import of clean particles obtained after 3D classification into pre-processing block.

   :fa:`plus-square,text-primary` Stop saving unnecessary metadata files during constrained refinement.

   :fa:`plus-square,text-primary` Implement particle list picker that was missing from some import blocks.

   :fa:`plus-square,text-primary` Implement parameter groups in UI to better handle conditional parameters.

   :fa:`plus-square,text-primary` Add links to download tomograms and metadata for ArtiaX plugin.

   :fa:`plus-square,text-primary` Provide more granular information when determining handedness of tilt-series.

   :fa:`plus-square,text-primary` Allow users to control the timeout for deleting the scratch folder of zombie jobs.

   :fa:`plus-square,text-primary` Add new parameter to control size of patches during patch-tracking to prevent tiltxcorr errors.

   :fa:`plus-square,text-primary` Upgrade program versions to MotionCor3 1.1.1 and AreTomo2 1.1.2.

   :fa:`plus-square,text-primary` Allow use of environment variables when specifying the local scratch directory.

   :fa:`bug,text-danger` Hide the export tab from particle filtering blocks for tomography projects.

   :fa:`bug,text-danger` Fix bug that ocurred when skipping frame alignment during movie processing.

   :fa:`bug,text-danger` Fix bug in function used to export sessions to .star format.

   :fa:`bug,text-danger` Fix bug in tomography sessions that ocurred when using size-based particle picking.

   :fa:`bug,text-danger` Fix bug when exporting metadata in star format that saved the files to the incorrect folder.

   :fa:`bug,text-danger` Fix bug when setting number of patches when running AreTomo2.

   :fa:`bug,text-danger` Fix inconsistencies in the determination of parameter changes between consecutive runs.

   :fa:`bug,text-danger` Stop trying to launch external programs for sub-tomogram averaging after particle extraction.

   :fa:`bug,text-danger` Fix issue with missing metadata entries during tilt-series re-processing.

   :fa:`bug,text-danger` Correctly discard particles that are too close to gold fiducials.

   :fa:`bug,text-danger` Fix issue with management of virion selection thresholds that affected geometric particle picking.

   :fa:`bug,text-danger` Fix bug when creating montages that ocurred when particle radius was equal to half the box size.

   :fa:`bug,text-danger` Fix bug when re-running pre-processing after virion selection.

   :fa:`bug,text-danger` Fix bug with links used to download maps for older iterations.

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