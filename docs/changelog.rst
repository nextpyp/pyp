=========
Changelog
=========

v0.7.1
------

.. nextpyp:: Released TBA
   :collapsible: open
   
   :fa:`plus-square` **Improvements**

   - Installer now shows progress bars when downloading files. If you have an older version of wget that doesn't support this option, progress bars will not be shown.

   - Update documentation, improve installation instructions, fix broken links, and start opening external links in new tabs.

   - Improve efficiency when copying files from network storage to local scratch by removing unnecessary transfers.

   - Use the ``-slurm_verbose`` option in the CLI to control whether sbatch commands are displayed in standard output.
   
   - Incorporate `AreTomo3 <https://github.com/czimaginginstitute/AreTomo3p>`_ routines for tilt-series alignment and tomogram reconstruction.
   
   - Always use mean image value to fill values when transforming stacks to prevent high contrast artifacts in tomograms.
   
   - Expose options for IMOD's findbeads3d command to give users more control over the detection of beads when erasing gold.
   
   - Let users specify minumum occupancy value when filtering particles in tomography pipeline, allowing for more flexible particle selection.
   
   - Start saving intermediate ML models during training of MiLoPYP refinement module to give users more flexibility when selecting a model for evaluation.
   
   - Override value of tilt-axis angle extracted from .mdoc files and show a warning instead telling users to change this setting in the Data Import block.

   - Add ability to control minimum occupancy values during particle filtering in tomo pipeline.
   
   - Add ability to control the number of iterations during tilt-series coarse alingment and change default number of correlation iterations to 1 to improve alignment accuracy.
   
   - Improve accuracy and consistency of virion picking in tomography pipeline by transitioning to the use of A units.

   :fa:`bug` **Bug fixes**
   
   - Fix bug that ocurred when propagating configuration settings from parent to downstream blocks.

   - Force ``Calculate reconstruction`` block to always execute a single iteration.
   
   - Fix tomography import blocks to correctly retrieve existing parameter values and particle coordinates.

   - Fix bug that ocurred when submitting merge jobs in the CLI that caused walltime parameter to be ignored.

   - Fix bug in MiLoPYP workflow that caused the wrong compressed file to be downloaded to the local machine.

   - Fix bug that prevented fiducial markers from being properly removed from some tilts when erasing gold.
   
   - Fix error that ocurred when trying to re-calculate tomograms using a GPU-accelerated reconstruction method.
   
   - Fix bug that caused an error when no particles were found using template matching during 3D particle picking.
   
   - Fix issue in the single-particle pipeline with application of gain reference files in the .gain format.

   - Fix error that ocurred when trying to save files ending in period (.) on cloud-based blob storage systems.

v0.7.0
------

.. nextpyp:: Released 5/5/2025
   :collapsible: open
   
   :fa:`star` **New features**
   
   - New blocks for running :doc:`MiLoPYP<guide/milopyp>` as described in `Huang et al., 2024 <https://www.nature.com/articles/s41592-024-02403-6>`_, including visualization of class labels and UMAP embeddings, with detected particles passed to downstream 3D refinement blocks.

   - New block architecture, with dedicated training and evaluation blocks, streamlines neural network (NN) workflows, offering greater flexibility and real-time visualization of loss functions and results from NN-based operations.

   - A new suite of tomography particle picking blocks provides an intuitive, standalone workflow with support for size-based, template matching, geometry-based, manual, and imported particle picking.
  
   - A simplified block architecture streamlines 3D refinement and classification, improving usability, while the legacy version remains available for older projects.

   - 3D particle picking via GPU-accelerated template search is supported through integration with `pytom-match-pick <https://sbc-utrecht.github.io/pytom-match-pick/>`_, with automatic transfer of particle orientations to downstream refinement blocks.

   - Size-based particle picking, as described in `Jin et al., 2024 <https://doi.org/10.1016/j.yjsbx.2024.100104>`_, enables fast particle detection in 3D using only the particle radius, with automatic masking of artifacts and contamination.

   - 3D tomogram segmentation with `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_ (evaluation only) enables rapid segmentation of tomograms using a pre-trained model.

   - Tomogram denoising with `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_ supports training and evaluation, with automatic half-tomogram generation and visualization of loss functions and denoised results.

   - Tomogram denosing with `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_ supports training and evaluation, with real-time monitoring of results and feeding of denoised tomograms into downstream blocks such as particle picking or segmentation.

   - New blocks for continuous heterogeneity analysis using `tomoDRGN <https://github.com/bpowell122/tomodrgn>`_, streaming training metrics and outputs in real-time, and visualization of UMAP, PCA embeddings, cluster centroids, and 3D volumes (beta).

   - New ab-initio refinement strategy enables de novo structure determination through iterative alignment of random particle subsets, with optional shape masking and multi-class refinement for heterogeneous datasets (beta).

   - Beam-tilt refinement and correction as implemented in cisTEM is now available in nextPYP, allowing users to improve the resolution of reconstructions.

   - Standalone mode is now also supported in the :doc:`command line interface (CLI)<cli/installation>`, allowing users to run it on a desktop or local computer without a cluster or web server.

   - nextPYP users can now be mapped to Operating System (OS) users, allowing job processes to run under individual OS accounts, ensuring compliance with resource policies and filesystem-controlled file access.

   - A new system of :doc:`SLURM cluster templates<reference/templates>` offers greater flexibility in job submission, allowing customization to fit various cluster environments and scheduler configurations.

   - Import tilt-series alignments from external programs in IMOD format using \*.xf and \*.tlt files, useful for processing challenging datasets that are hard to align automatically.

   - Export clean 3D particle coordinates in IMOD format (sva/\*.spk files) from any refinement or classification block for use in other programs.

   - New option lets users export particle stacks for compatibility with external programs, despite nextPYP workflows not saving them to optimize storage.

   - Refinement metadata is now stored in cisTEM's binary format, enabling up to 2x faster refinement and classification, with smaller, quicker-to-read files. The previous format is still available for legacy projects.

   - Automatic density-based masking is now available during 3D refinement, applying an adaptive shape mask to the reference map at each refinement iteration to improve reconstruction quality.

   - An option to save `*.mrc` files in 16-bit precision has been added, offering up to 50% storage savings (enabled by default), reducing storage needs for large datasets.
 
   - The "Only" option in the Jobs menu lets you quickly select and run individual blocks with a single click, simplifying workflows in projects with many blocks.

   - In addition to Relion 4, nextPYP now supports importing Relion 5 tomography projects, allowing users to take advantage of new features while continuing to use other packages.

   - New documentation offers expanded tutorials, user guides, and setup instructions, including detailed installation steps for clusters and workstations, and comprehensive coverage of new features and cryo-ET workflows.

   :fa:`plus-square` **Improvements**

   - Improved efficiency and robustness for handling large single-particle and tomography datasets, with optimizations in data handling, processing speed, and memory management.

   - IMOD tilt-series alignment and reconstruction now provide enhanced control with additional parameters, offering users more flexibility to customize settings for their specific datasets.

   - The ``Show advanced options`` checkbox now applies globally, ensuring consistency across all dialog forms and remembering the setting for improved convenience.

   - Reshaping image options have been moved from the **Reconstruction** tab to the **Tilt-series alignment** tab, streamlining the workflow and making the settings more intuitive.

   - Users can specify how many times nextPYP should retry failed SLURM jobs, ensuring successful completion of runs even during temporary issues.

   - Improved handling of micrographs/tilt-series that have few or no particles after filtering.

   - Report the residual error of IMOD's fiducial model during tilt-series alignment, providing a measure of alignment quality to help users assess accuracy.

   - Improved handling of tilt-series from rectangular detectors, with automatic rotation to ensure correct orientation and efficient processing throughout the workflow.

   :fa:`bug` **Bug fixes**
   
   - Fixed a bug in the navigation bar of refinement blocks that occurred when multiple classes were used.

   - Fixed a bug related to applying IMOD anisotropic diffusion denoising during the refinement process.

   - Fixed a bug that prevented launch task parameters from being applied when starting sessions.

   - Fixed a bug that caused incorrect binning to be applied during manual virion picking.

   - Fixed a bug that prevented tomogram dimensions and binning from updating correctly.

   - Fixed a bug that prevented tomograms from being recalculated in AreTomo when reconstruction parameters were modified.

   - Various bug fixes and performance improvements.

v0.6.5
------
.. nextpyp:: Released 4/6/2024
   :collapsible: open

   :fa:`plus-square` **Improvements**

   - Update format of logger messages to more clearly show the nextPYP version and resources assigned to each job.

   - Use same tilt-axis angle convention for aligning tilt-series using IMOD and AreTomo2.

   :fa:`bug` **Bug fixes**

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

   :fa:`plus-square` **Improvements**

   - Implement mechanism to isolate logs from failed jobs in the Logs tab.

   - Add support for project names with special characters.

   - Remove many commonly used parameters from the advanced category.

   - Add progress bars during export of metadata to .star format.

   - Allow export of particle coordinates from streaming sessions.

   - Check that .order files have the same number of entries as images in the tilt-series.

   :fa:`bug` **Bug fixes**

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

   :fa:`plus-square` **Improvements**

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

   :fa:`bug` **Bug fixes**

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

   :fa:`plus-square` **Improvements**

   - Expose additional parameters for frame alignment when using MotionCor3.

   - Remove unnecessary tabs from tomography refinement blocks.

   - Display slurm job launch information in the logs window.

   - Allow users to specify resources for the launch task on the Sessions side.

   :fa:`bug` **Bug fixes**

   - Fix bugs in parameter definitions when running movie frame alignment.

   - Fix bugs in the management of slurm's GRES options when submitting jobs to the scheduler.

   - Fix bug with movie drifts being deleted from the database when tilt-series were re-processed.

v0.6.1
------
.. nextpyp:: Released 1/30/2024
   :collapsible: open

   :fa:`star` **New features**

   - Produce metadata for 3D visualization using `ArtiaX <https://github.com/FrangakisLab/ArtiaX>`_ for all refinement blocks. See the :doc:`user guide<guide/chimerax_artiax>` for details.

   - Enable dose weighting and magnification correction options during frame alignment and averaging.

   - Allow specification of SLURM account for all job types to improve portability.

   :fa:`plus-square` **Improvements**

   - Expose full set of options when using MotionCor3 for frame alignment.

   - Allow specification of GPU resources using Gres option to allow selection of specific types of graphics cards, e.g., gpu:A100:1.

   - Add support for multiple date formats when reading metadata from .mdoc files.

   - Add support for .gain reference files and automatically resize corresponding .eer movies in data import blocks.

   :fa:`bug` **Bug fixes**

   - Fix issue when handling \*.tif files that have a \*.tiff extension.

   - Fix issue with multiprocessing library when using NFS mounts as local scratch.

   - Fix bug in single-particle sessions when using unbinned images for 2D classification.

   - Fix bug when picking particles using neural network-based approach on non-square tomograms.

   - Fix bug that prevented GPU jobs from running because the jobs were sent to the CPU queue.

v0.6.0
------
.. nextpyp:: Released 1/21/2024
   :collapsible: open

   :fa:`star` **New features**

   - Allow use of `MotionCor3 <https://github.com/czimaginginstitute/MotionCor3>`_ for movie frame alignment (GPU required).

   - Allow use of `AreTomo2 <https://github.com/czimaginginstitute/AreTomo2>`_ for tilt-series alignment and reconstruction (GPU required).

   - Allow use of `Topaz <https://github.com/tbepler/topaz>`_ for 2D particle picking and 3D denoising (GPU recommended).

   - Produce .bild files after each refinement iteration for 3D visualization in Chimera/ChimeraX.

   - Automatic determination of CTF handedness during pre-processing of tilt-series.

   :fa:`plus-square` **Improvements**

   - Allow mix-and-match of IMOD and AreTomo2 for tilt-series alignment and tomogram reconstruction.

   - Automatically submit jobs to a GPU partition when running tasks that require GPU acceleration.

   - Display version number and amount of allocated memory at the beginning of every job.

   - Change default memory allocation for launch task to 4GB and add Resources tab to all data import blocks.

   - Simplify Resources tab by hiding unnecessary parameters depending on the block type.

   - Implement GPU resource management policies for slurm and standalone modes.

   - Show per-particle score distribution for all tomography refinement blocks and improve plot layout.

   - Allow use of slurm's GRES (generic resource scheduling) when submitting jobs to a cluster.

   :fa:`bug` **Bug fixes**

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

   :fa:`star` **New features**

   - Implement interactive measuring tool for micrographs and tomograms.

   - Allow multiple sessions when user login mode is enabled.

   :fa:`plus-square` **Improvements**

   - Sort classes in increasing order in Class View panel.

   :fa:`bug` **Bug fixes**

   - Fix issues when limiting total number of tasks in slurm scheduler.

v0.5.2
------
.. nextpyp:: Released 11/18/2023
   :collapsible: open

   :fa:`star` **New features**

   - Add support for PACEtomo tilt-series in streaming Sessions.

   :fa:`plus-square` **Improvements**

   - Parallelize reconstruction step during 3D classification for faster speeds.

   - Add new options to flip maps in post-processing block.

   - Simplify installation instructions and setup process.

   :fa:`bug` **Bug fixes**

   - Fix issue with location of executables for neural network-based particle picking.

   - Fix issue with re-calculation of binned tomograms when reconstruction parameters change.

   - Fix issue with re-calculation of particle coordinates when no particles were found.

   - Correctly display particle size in tomography pre-processing block statistics.

v0.5.1
------
.. nextpyp:: Released 11/04/2023
   :collapsible: open

   :fa:`star` **New features**

   - Import frame tilt-series data using mdoc files produced by PACEtomo.

   :fa:`plus-square` **Improvements**

   - Allow typing iteration number in navigation bar for refinement blocks.

   - Show refinement/bundle IDs in ``Per-particle Score`` and ``Exposure Weights`` tabs for refinement blocks.

   :fa:`bug` **Bug fixes**

   - Fix issue with display of tomograms with arbitrary thickness.

   - Fix broken CLI commands and update CLI tutorials.

v0.5.0
------
.. nextpyp:: Released 10/26/2023
   :collapsible: open

   - This was the first release of nextPYP.