=========
Changelog
=========

.. list-table:: **Key**
   :header-rows: 0
   :align: left

   * - :fa:`star` New feature
     - :fa:`plus-square` Improvement
     - :fa:`bug` Bug fix

v0.7.0
------

.. nextpyp:: Released 2/26/2025
   :collapsible: open
   
   :fa:`star` New features
   
   - New blocks to run :doc:`molecular pattern mining and particle localization<guide/milopyp>` (MiLoPYP) as described in `Huang et al., 2024 <https://www.nature.com/articles/s41592-024-02403-6>`_.

   - New block architecture with dedicated training and evaluation blocks facilitates the execution of neural network-based operations.

   - New dedicated suite of blocks for tomography particle picking that is more intuitive and decoupled from other pre-processing operations.

   - 3D particle picking using template-search as implemented in `pytom-match-pick <https://sbc-utrecht.github.io/pytom-match-pick/>`_.

   - 3D size-based particle picking as described in `Jin et al., 2024 <https://doi.org/10.1016/j.yjsbx.2024.100104>`_.

   - 3D segmentation of tomograms using `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_(evaluation only).

   - Tomogram denosing using `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_ (training and evaluation).

   - Tomogram denosing using `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_ (training and evaluation).

   - New blocks to run continuous heterogeneity analysis using `tomoDRGN <https://github.com/bpowell122/tomodrgn>`_.

   - New ab-initio refinement strategy using constrained single-particle tomography to determine structures *de novo*.

   - Beam-tilt refinement and correction for single-particle and tomography pipelines.

   - Add support for multiple GPUs to run computationally intensive jobs.

   - Standalone mode is now also supported in the :doc:`command line interface (CLI)<cli/installation>` pipeline.

   - nextPYP can now submit slurm jobs using individual OS user accounts. This allows users in the same instance of the web server to run jobs using their own linux accounts and permissions.

   - New system of :doc:`cluster templates<reference/templates>` provides more flexibility when submitting jobs to a SLURM cluster.

   - Import tilt-series alignments obtained with external programs in IMOD format (\*.xf and \*.tlt files).

   - Export 3D particle coordinates in IMOD format to use in external programs (sva/\*.spk files).

   - New option to export particle stacks for use in external programs (using this option sacrifices storage savings).

   - Store refinement metadata in cisTEM's new binary format for 2x faster refinement and classification.

   - Automatic density-based shape-masking during 3D refinement.

   - Add option to save `*.mrc` files in 16-bit precision to enable 50% storage savings (enabled by default).

   - Select individual blocks to run from the **Jobs** menu using a single click ("Only" option).

   - New theme for documentation with expanded tutorials, user and configuration guides.

   :fa:`plus-square` Improvements

   - More efficient and robust handling of large single-particle and tomography datasets.

   - Finer control over tilt-series alignment and reconstruction options when using IMOD.

   - Checkbox to "Show advanced options" has global scope now.

   - Move options to reshape images into squares from Reconstruction to Tilt-series alignment tab.

   - Allow users to specify how many times should nextPYP try to relaunch failed SLURM jobs.

   - Improved handling of micrographs/tilt-series that have few or no particles after filtering.

   - Report IMOD's fiducial model residual error during tilt-series alignment.

   - Better handling of tilt-series from rectangular-shaped detectors.

   - Import Relion 5 tomography projects.

   :fa:`bug` Bug fixes
   
   - Fix bug in navigation bar for refinement blocks when multiple classes were used.

   - Fix bug when applying IMOD anisotropic diffusion denoising during refinement.

   - Fix bug that prevented launch task parameters from being used when launching sessions.

   - Fix bug where incorrect binning was applied when picking virions manually.

   - Fix bug where tomogram dimensions and binning were not properly updated.

   - Fix bug that prevented the recalculation of tomograms when using AreTomo if the reconstruction parameters changed.

   - Several other bug fixes and improvements.

v0.6.5
------
.. nextpyp:: Released 4/6/2024
   :collapsible: open

   :fa:`plus-square` Improvements

   - Update format of logger messages to more clearly show the nextPYP version and resources assigned to each job.

   - Use same tilt-axis angle convention for aligning tilt-series using IMOD and AreTomo2.

   :fa:`bug` Bug fixes

   - Prevent error during tilt-series alignment with AreTomo2 when number of patches = 1.

   - Fix bug in command line interface that ocurred when launching constrained refinement.

   - Fix bug that was causing the server startup routines to be called during the CLI processing of the configuration file.

   - Fix bug that ocurred when retrieving metadata from mdoc files.

   - Fix bug when trying to retrieve tilt-series metadata from failed runs.

   - Fix conflicts with library paths when running external executables.

v0.6.4
------
.. nextpyp:: Released 3/24/2024
   :collapsible: open

   :fa:`plus-square` Improvements

   - Implement mechanism to isolate logs from failed jobs in the Logs tab.

   - Add support for project names with special characters.

   - Remove many commonly used parameters from the advanced category.

   - Add progress bars during export of metadata to .star format.

   - Allow export of particle coordinates from streaming sessions.

   - Check that .order files have the same number of entries as images in the tilt-series.

   :fa:`bug` Bug fixes

   - Fix bugs when reading metadata from \*.mdoc files.

   - Prevent dragging of multiple connections from block outputs in project view.

   - Fix bug when managing GPU resources in standalone mode.

   - Fix bug when using grouping of frames during movie processing.

   - Fix bug in single-particle pipeline during hot pixel removal.

   - Fix bug in Table view that caused content to overlap when resizing columns.

   - Always export metadata in .star format to current project directory (user specified location is no longer supported).

v0.6.3
------
.. nextpyp:: Released 3/01/2024
   :collapsible: open

   :fa:`plus-square` Improvements

   - Allow import of clean particles obtained after 3D classification into pre-processing block.

   - Stop saving unnecessary metadata files during constrained refinement.

   - Implement particle list picker that was missing from some import blocks.

   - Implement parameter groups in UI to better handle conditional parameters.

   - Add links to download tomograms and metadata for ArtiaX plugin.

   - Provide more granular information when determining handedness of tilt-series.

   - Allow users to control the timeout for deleting the scratch folder of zombie jobs.

   - Add new parameter to control size of patches during patch-tracking to prevent tiltxcorr errors.

   - Upgrade program versions to MotionCor3 1.1.1 and AreTomo2 1.1.2.

   - Allow use of environment variables when specifying the local scratch directory.

   :fa:`bug` Bug fixes

   - Hide the export tab from particle filtering blocks for tomography projects.

   - Fix bug that ocurred when skipping frame alignment during movie processing.

   - Fix bug in function used to export sessions to .star format.

   - Fix bug in tomography sessions that ocurred when using size-based particle picking.

   - Fix bug when exporting metadata in star format that saved the files to the incorrect folder.

   - Fix bug when setting number of patches when running AreTomo2.

   - Fix inconsistencies in the determination of parameter changes between consecutive runs.

   - Stop trying to launch external programs for sub-tomogram averaging after particle extraction.

   - Fix issue with missing metadata entries during tilt-series re-processing.

   - Correctly discard particles that are too close to gold fiducials.

   - Fix issue with management of virion selection thresholds that affected geometric particle picking.

   - Fix bug when creating montages that ocurred when particle radius was equal to half the box size.

   - Fix bug when re-running pre-processing after virion selection.

   - Fix bug with links used to download maps for older iterations.

v0.6.2
-------
.. nextpyp:: Released 2/01/2024
   :collapsible: open

   :fa:`plus-square` Improvements

   - Expose additional parameters for frame alignment when using MotionCor3.

   - Remove unnecessary tabs from tomography refinement blocks.

   - Display slurm job launch information in the logs window.

   - Allow users to specify resources for the launch task on the Sessions side.

   :fa:`bug` Bug fixes

   - Fix bugs in parameter definitions when running movie frame alignment.

   - Fix bugs in the management of slurm's GRES options when submitting jobs to the scheduler.

   - Fix bug with movie drifts being deleted from the database when tilt-series were re-processed.

v0.6.1
------
.. nextpyp:: Released 1/30/2024
   :collapsible: open

   :fa:`star` New features

   - Produce metadata for 3D visualization using `ArtiaX <https://github.com/FrangakisLab/ArtiaX>`_ for all refinement blocks. See the :doc:`user guide<guide/chimerax_artiax>` for details.

   - Enable dose weighting and magnification correction options during frame alignment and averaging.

   - Allow specification of SLURM account for all job types to improve portability.

   :fa:`plus-square` Improvements

   - Expose full set of options when using MotionCor3 for frame alignment.

   - Allow specification of GPU resources using Gres option to allow selection of specific types of graphics cards, e.g., gpu:A100:1.

   - Add support for multiple date formats when reading metadata from .mdoc files.

   - Add support for .gain reference files and automatically resize corresponding .eer movies in data import blocks.

   :fa:`bug` Bug fixes

   - Fix issue when handling \*.tif files that have a \*.tiff extension.

   - Fix issue with multiprocessing library when using NFS mounts as local scratch.

   - Fix bug in single-particle sessions when using unbinned images for 2D classification.

   - Fix bug when picking particles using neural network-based approach on non-square tomograms.

   - Fix bug that prevented GPU jobs from running because the jobs were sent to the CPU queue.

v0.6.0
------
.. nextpyp:: Released 1/21/2024
   :collapsible: open

   :fa:`star` New features

   - Allow use of `MotionCor3 <https://github.com/czimaginginstitute/MotionCor3>`_ for movie frame alignment (GPU required).

   - Allow use of `AreTomo2 <https://github.com/czimaginginstitute/AreTomo2>`_ for tilt-series alignment and reconstruction (GPU required).

   - Allow use of `Topaz <https://github.com/tbepler/topaz>`_ for 2D particle picking and 3D denoising (GPU recommended).

   - Produce .bild files after each refinement iteration for 3D visualization in Chimera/ChimeraX.

   - Automatic determination of CTF handedness during pre-processing of tilt-series.

   :fa:`plus-square` Improvements

   - Allow mix-and-match of IMOD and AreTomo2 for tilt-series alignment and tomogram reconstruction.

   - Automatically submit jobs to a GPU partition when running tasks that require GPU acceleration.

   - Display version number and amount of allocated memory at the beginning of every job.

   - Change default memory allocation for launch task to 4GB and add Resources tab to all data import blocks.

   - Simplify Resources tab by hiding unnecessary parameters depending on the block type.

   - Implement GPU resource management policies for slurm and standalone modes.

   - Show per-particle score distribution for all tomography refinement blocks and improve plot layout.

   - Allow use of slurm's GRES (generic resource scheduling) when submitting jobs to a cluster.

   :fa:`bug` Bug fixes

   - Fix OOM error when running constrained refinement using a single thread.

   - Fix error in particle filtering blocks when no particles are left in a given micrograph/tilt-series.

   - Fix issue in tomography sessions when .mdoc files are not used to import metadata.

   - Fix bug when exporting sub-tomograms for use in external programs.

   - Update systemd script to improve robustness during program restart.

   - Fix issues with cancellation of jobs in standalone mode.

   - Fix discrepancy with gain reference rotation/flips between data import and pre-processing blocks.

v0.5.3
------
.. nextpyp:: Released 11/25/2023
   :collapsible: open

   :fa:`star` New features

   - Implement interactive measuring tool for micrographs and tomograms.

   - Allow multiple sessions when user login mode is enabled.

   :fa:`plus-square` Improvements

   - Sort classes in increasing order in Class View panel.

   :fa:`bug` Bug fixes

   - Fix issues when limiting total number of tasks in slurm scheduler.

v0.5.2
------
.. nextpyp:: Released 11/18/2023
   :collapsible: open

   :fa:`star` New features

   - Add support for PACEtomo tilt-series in streaming Sessions.

   :fa:`plus-square` Improvements

   - Parallelize reconstruction step during 3D classification for faster speeds.

   - Add new options to flip maps in post-processing block.

   - Simplify installation instructions and setup process.

   :fa:`bug` Bug fixes

   - Fix issue with location of executables for neural network-based particle picking.

   - Fix issue with re-calculation of binned tomograms when reconstruction parameters change.

   - Fix issue with re-calculation of particle coordinates when no particles were found.

   - Correctly display particle size in tomography pre-processing block statistics.

v0.5.1
------
.. nextpyp:: Released 11/04/2023
   :collapsible: open

   :fa:`star` New features

   - Import frame tilt-series data using mdoc files produced by PACEtomo.

   :fa:`plus-square` Improvements

   - Allow typing iteration number in navigation bar for refinement blocks.

   - Show refinement/bundle IDs in ``Per-particle Score`` and ``Exposure Weights`` tabs for refinement blocks.

   :fa:`bug` Bug fixes

   - Fix issue with display of tomograms with arbitrary thickness.

   - Fix broken CLI commands and update CLI tutorials.

v0.5.0
------
.. nextpyp:: Released 10/26/2023
   :collapsible: open

   - This was the first release of nextPYP.