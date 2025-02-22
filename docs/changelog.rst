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
   
   :fa:`star` New blocks to run :doc:`molecular pattern mining and particle localization<guide/milopyp>` (MiLoPYP) as described in `Huang et al., 2024 <https://www.nature.com/articles/s41592-024-02403-6>`_.

   :fa:`star` New block architecture with dedicated training and evaluation blocks facilitates the execution of neural network-based operations.

   :fa:`star` New dedicated suite of blocks for tomography particle picking that is more intuitive and decoupled from other pre-processing operations.

   :fa:`star` 3D template-search particle picking using `pytom-match-pick <https://sbc-utrecht.github.io/pytom-match-pick/>`_.

   :fa:`star` 3D size-based particle picking as described in `Jin et al., 2024 <https://doi.org/10.1016/j.yjsbx.2024.100104>`_.

   :fa:`star` 3D segmentation of tomograms using `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_.

   :fa:`star` Tomogram denosing using `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_ (training and evaluation)).

   :fa:`star` Tomogram denosing using `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_ (training and evaluation).

   :fa:`star` 3D continuous heterogneity analysis using `tomoDRGN <https://github.com/bpowell122/tomodrgn>`_.

   :fa:`star` New ab-initio refinement strategy using constrained single-particle tomography to determine structures de novo.

   :fa:`star` Beam-tilt refinement and correction for single-particle and tomography pipelines.

   :fa:`star` Use multiple GPUs to run computationally intensive jobs.

   :fa:`star` Standalone mode now also supported in command line interface (CLI) pipeline.

   :fa:`star` nextPYP can now submit slurm jobs using individual OS user accounts! This allows users in the same instance of the web server to run jobs using their own linux accounts.

   :fa:`star` New system of *cluster templates* provides more flexibility when submitting jobs to a SLURM cluster.

   :fa:`star` Import tilt-series alignments obtained with external programs in IMOD format (\*.xf and \*.tlt files).

   :fa:`star` Export 3D particle coordinates in IMOD format to use in external programs (sva/\*.spk files).

   :fa:`star` New option to export particle stacks for use in external programs (plenty of storage required!).

   :fa:`star` Store refinement metadata in cisTEM's new binary format for 2x faster refinement and classification.

   :fa:`star` Automatic density-based shape-masking during 3D refinement.

   :fa:`star` Add option to save mrc files in 16-bit precision to enable 50% storage savings (activated by default).

   :fa:`star` Select individual blocks to run from the Jobs menu using a single click ("Only" option).

   :fa:`star` New theme for documentation with expanded tutorials, user and configuration guides.

   :fa:`plus-square` More efficient and robust handling of large single-particle and tomography datasets.

   :fa:`plus-square` Finer control over tilt-series alignment and reconstruction options when using IMOD.

   :fa:`plus-square` Checkbox to "Show advanced options" has global scope now.

   :fa:`plus-square` Move options to reshape images into squares from Reconstruction to Tilt-series alignment tab.

   :fa:`plus-square` Allow users to specify how many times should nextPYP try to relaunch failed SLURM jobs.

   :fa:`plus-square` Improved handling of micrographs/tilt-series that have few or no particles after filtering.

   :fa:`plus-square` Report IMOD's fiducial model residual error during tilt-series alignment.

   :fa:`plus-square` Better handling of tilt-series from rectangular-shaped detectors.

   :fa:`plus-square` Import Relion 5 tomography projects.

   :fa:`bug` Fix bug in navigation bar for refinement blocks when multiple classes were used.

   :fa:`bug` Fix bug when applying IMOD anisotropic diffusion denoising during refinement.

   :fa:`bug` Fix bug that prevented launch task parameters from being used when launching sessions.

   :fa:`bug` Fix bug where incorrect binning was applied when picking virions manually.

   :fa:`bug` Fix bug where tomogram dimensions and binning were not properly updated.

   :fa:`bug` Fix bug that prevented the recalculation of tomograms when using AreTomo if the reconstruction parameters changed.

   :fa:`bug` Several other bug fixes and improvements.

v0.6.5
------
.. nextpyp:: Released 4/6/2024
   :collapsible: open

   :fa:`plus-square` Update format of logger messages to more clearly show the nextPYP version and resources assigned to each job.

   :fa:`plus-square` Use same tilt-axis angle convention for aligning tilt-series using IMOD and AreTomo2.

   :fa:`bug` Prevent error during tilt-series alignment with AreTomo2 when number of patches = 1.

   :fa:`bug` Fix bug in command line interface that ocurred when launching constrained refinement.

   :fa:`bug` Fix bug that was causing the server startup routines to be called during the CLI processing of the configuration file.

   :fa:`bug` Fix bug that ocurred when retrieving metadata from mdoc files.

   :fa:`bug` Fix bug when trying to retrieve tilt-series metadata from failed runs.

   :fa:`bug` Fix conflicts with library paths when running external executables.

v0.6.4
------
.. nextpyp:: Released 3/24/2024
   :collapsible: open

   :fa:`plus-square` Implement mechanism to isolate logs from failed jobs in the Logs tab.

   :fa:`plus-square` Add support for project names with special characters.

   :fa:`plus-square` Remove many commonly used parameters from the advanced category.

   :fa:`plus-square` Add progress bars during export of metadata to .star format.

   :fa:`plus-square` Allow export of particle coordinates from streaming sessions.

   :fa:`plus-square` Check that .order files have the same number of entries as images in the tilt-series.

   :fa:`bug` Fix bugs when reading metadata from \*.mdoc files.

   :fa:`bug` Prevent dragging of multiple connections from block outputs in project view.

   :fa:`bug` Fix bug when managing GPU resources in standalone mode.

   :fa:`bug` Fix bug when using grouping of frames during movie processing.

   :fa:`bug` Fix bug in single-particle pipeline during hot pixel removal.

   :fa:`bug` Fix bug in Table view that caused content to overlap when resizing columns.

   :fa:`bug` Always export metadata in .star format to current project directory (user specified location is no longer supported).

v0.6.3
------
.. nextpyp:: Released 3/01/2024
   :collapsible: open

   :fa:`plus-square` Allow import of clean particles obtained after 3D classification into pre-processing block.

   :fa:`plus-square` Stop saving unnecessary metadata files during constrained refinement.

   :fa:`plus-square` Implement particle list picker that was missing from some import blocks.

   :fa:`plus-square` Implement parameter groups in UI to better handle conditional parameters.

   :fa:`plus-square` Add links to download tomograms and metadata for ArtiaX plugin.

   :fa:`plus-square` Provide more granular information when determining handedness of tilt-series.

   :fa:`plus-square` Allow users to control the timeout for deleting the scratch folder of zombie jobs.

   :fa:`plus-square` Add new parameter to control size of patches during patch-tracking to prevent tiltxcorr errors.

   :fa:`plus-square` Upgrade program versions to MotionCor3 1.1.1 and AreTomo2 1.1.2.

   :fa:`plus-square` Allow use of environment variables when specifying the local scratch directory.

   :fa:`bug` Hide the export tab from particle filtering blocks for tomography projects.

   :fa:`bug` Fix bug that ocurred when skipping frame alignment during movie processing.

   :fa:`bug` Fix bug in function used to export sessions to .star format.

   :fa:`bug` Fix bug in tomography sessions that ocurred when using size-based particle picking.

   :fa:`bug` Fix bug when exporting metadata in star format that saved the files to the incorrect folder.

   :fa:`bug` Fix bug when setting number of patches when running AreTomo2.

   :fa:`bug` Fix inconsistencies in the determination of parameter changes between consecutive runs.

   :fa:`bug` Stop trying to launch external programs for sub-tomogram averaging after particle extraction.

   :fa:`bug` Fix issue with missing metadata entries during tilt-series re-processing.

   :fa:`bug` Correctly discard particles that are too close to gold fiducials.

   :fa:`bug` Fix issue with management of virion selection thresholds that affected geometric particle picking.

   :fa:`bug` Fix bug when creating montages that ocurred when particle radius was equal to half the box size.

   :fa:`bug` Fix bug when re-running pre-processing after virion selection.

   :fa:`bug` Fix bug with links used to download maps for older iterations.

v0.6.2
-------
.. nextpyp:: Released 2/01/2024
   :collapsible: open

   :fa:`plus-square` Expose additional parameters for frame alignment when using MotionCor3.

   :fa:`plus-square` Remove unnecessary tabs from tomography refinement blocks.

   :fa:`plus-square` Display slurm job launch information in the logs window.

   :fa:`plus-square` Allow users to specify resources for the launch task on the Sessions side.

   :fa:`bug` Fix bugs in parameter definitions when running movie frame alignment.

   :fa:`bug` Fix bugs in the management of slurm's GRES options when submitting jobs to the scheduler.

   :fa:`bug` Fix bug with movie drifts being deleted from the database when tilt-series were re-processed.

v0.6.1
------
.. nextpyp:: Released 1/30/2024
   :collapsible: open

   :fa:`star` Produce metadata for 3D visualization using `ArtiaX <https://github.com/FrangakisLab/ArtiaX>`_ for all refinement blocks. See the :doc:`user guide<guide/chimerax_artiax>` for details.

   :fa:`star` Enable dose weighting and magnification correction options during frame alignment and averaging.

   :fa:`star` Allow specification of SLURM account for all job types to improve portability.

   :fa:`plus-square` Expose full set of options when using MotionCor3 for frame alignment.

   :fa:`plus-square` Allow specification of GPU resources using Gres option to allow selection of specific types of graphics cards, e.g., gpu:A100:1.

   :fa:`plus-square` Add support for multiple date formats when reading metadata from .mdoc files.

   :fa:`plus-square` Add support for .gain reference files and automatically resize corresponding .eer movies in data import blocks.

   :fa:`bug` Fix issue when handling \*.tif files that have a \*.tiff extension.

   :fa:`bug` Fix issue with multiprocessing library when using NFS mounts as local scratch.

   :fa:`bug` Fix bug in single-particle sessions when using unbinned images for 2D classification.

   :fa:`bug` Fix bug when picking particles using neural network-based approach on non-square tomograms.

   :fa:`bug` Fix bug that prevented GPU jobs from running because the jobs were sent to the CPU queue.

v0.6.0
------
.. nextpyp:: Released 1/21/2024
   :collapsible: open

   :fa:`star` Allow use of `MotionCor3 <https://github.com/czimaginginstitute/MotionCor3>`_ for movie frame alignment (GPU required).

   :fa:`star` Allow use of `AreTomo2 <https://github.com/czimaginginstitute/AreTomo2>`_ for tilt-series alignment and reconstruction (GPU required).

   :fa:`star` Allow use of `Topaz <https://github.com/tbepler/topaz>`_ for 2D particle picking and 3D denoising (GPU recommended).

   :fa:`star` Produce .bild files after each refinement iteration for 3D visualization in Chimera/ChimeraX.

   :fa:`star` Automatic determination of CTF handedness during pre-processing of tilt-series.

   :fa:`plus-square` Allow mix-and-match of IMOD and AreTomo2 for tilt-series alignment and tomogram reconstruction.

   :fa:`plus-square` Automatically submit jobs to a GPU partition when running tasks that require GPU acceleration.

   :fa:`plus-square` Display version number and amount of allocated memory at the beginning of every job.

   :fa:`plus-square` Change default memory allocation for launch task to 4GB and add Resources tab to all data import blocks.

   :fa:`plus-square` Simplify Resources tab by hiding unnecessary parameters depending on the block type.

   :fa:`plus-square` Implement GPU resource management policies for slurm and standalone modes.

   :fa:`plus-square` Show per-particle score distribution for all tomography refinement blocks and improve plot layout.

   :fa:`plus-square` Allow use of slurm's GRES (generic resource scheduling) when submitting jobs to a cluster.

   :fa:`bug` Fix OOM error when running constrained refinement using a single thread.

   :fa:`bug` Fix error in particle filtering blocks when no particles are left in a given micrograph/tilt-series.

   :fa:`bug` Fix issue in tomography sessions when .mdoc files are not used to import metadata.

   :fa:`bug` Fix bug when exporting sub-tomograms for use in external programs.

   :fa:`bug` Update systemd script to improve robustness during program restart.

   :fa:`bug` Fix issues with cancellation of jobs in standalone mode.

   :fa:`bug` Fix discrepancy with gain reference rotation/flips between data import and pre-processing blocks.

v0.5.3
------
.. nextpyp: Released 11/25/2023
   :collapsible: open

   :fa:`star` Implement interactive measuring tool for micrographs and tomograms.

   :fa:`star` Allow multiple sessions when user login mode is enabled.

   :fa:`plus-square` Sort classes in increasing order in Class View panel.

   :fa:`bug` Fix issues when limiting total number of tasks in slurm scheduler.

v0.5.2
------
.. nextpyp:: Released 11/18/2023
   :collapsible: open

   :fa:`star` Add support for PACEtomo tilt-series in streaming Sessions.

   :fa:`plus-square` Parallelize reconstruction step during 3D classification for faster speeds.

   :fa:`plus-square` Add new options to flip maps in post-processing block.

   :fa:`plus-square` Simplify installation instructions and setup process.

   :fa:`bug` Fix issue with location of executables for neural network-based particle picking.

   :fa:`bug` Fix issue with re-calculation of binned tomograms when reconstruction parameters change.

   :fa:`bug` Fix issue with re-calculation of particle coordinates when no particles were found.

   :fa:`bug` Correctly display particle size in tomography pre-processing block statistics.

v0.5.1
------
.. nextpyp:: Released 11/04/2023
   :collapsible: open

   :fa:`star` Import frame tilt-series data using mdoc files produced by PACEtomo.

   :fa:`plus-square` Allow typing iteration number in navigation bar for refinement blocks.

   :fa:`plus-square` Show refinement/bundle IDs in ``Per-particle Score`` and ``Exposure Weights`` tabs for refinement blocks.

   :fa:`bug` Fix issue with display of tomograms with arbitrary thickness.

   :fa:`bug` Fix broken CLI commands and update CLI tutorials.

v0.5.0
------
.. nextpyp:: Released 10/26/2023
   :collapsible: open

   This was the first release of nextPYP.