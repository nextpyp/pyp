===================
Tomography tutorial
===================

This tutorial shows how to convert tilt-series from the `HIV-1 Gag (EMPIAR-10164) <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ dataset into a ~3A resolution structure.

We first download and decompress a ``.tbz`` file containing a subset of 5 tilt-series (down-sampled 2x compared to the original data), and an initial model:

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_tomo_tutorial.tbz
  tar xvfz nextpyp_tomo_tutorial.tbz

Step 1: Create a new project
============================

Next, we create an empty folder where all files for the tutorial will be saved:

.. code-block:: bash

    mkdir EMPIAR-10164
    cd EMPIAR-10164

Step 2: Pre-processing
======================

Data pre-processing consists of movie frame alignment, tilt-series alignment, tomogram reconstruction, CTF estimation and virion detection:

.. code-block:: bash

    # launch pre-processing
    
    pyp -data_mode tomo                                 \
        -scope_voltage 300                              \
        -scope_pixel 1.35                               \
        -data_path="ABSOLUTE_PATH_TO_FRAMES/*.tif"      \
        -ctf_max_res 5.0                                \
        -scope_tilt_axis 85.3                           \
        -tomo_rec_binning 8                             \
        -tomo_rec_thickness 2048                        \
        -tomo_vir_method "auto"                         \
        -tomo_vir_rad 500.0                             \
        -slurm_tasks 23                                 \
        -slurm_memory 92                                \
        -movie_pattern "TILTSERIES_SCANORD_ANGLE.tif"


Step 3 (optional): Virion segmentation
======================================

In this step we use ``IMOD`` over a remote X11 connection to interactively select virion segmentation thresholds. Execute the command below and select the column in the image where the yellow curve more closely matches the membrane. Go over all virions in the tilt-series and save the model when you are done:

.. code-block:: bash

    ( export tomoedit=tomoedit &&
        singularity exec -B /your_binds --no-home -B $HOME/.ssh /path_to_container/pyp.sif
        /opt/pyp/bin/run/pyp -vir -skip )

.. tip::

    To skip a virion, simply select the rightmost column (these virions will be removed from the downstream processing).


Step 4: Particle detection
==========================

Detect spikes using the membrane values selected above:

.. code-block:: bash

    pyp -tomo_vir_detect_method "mesh"          \
        -tomo_vir_detect_dist 8                 \
        -tomo_vir_detect_band 800               \
        -tomo_spk_rad 50.0


Step 5: Reference-based refinement
==================================

If a 3D reference is available, the ``csp`` command can be used to align the particle projections using constrained refinement:

.. code-block:: bash

    # launch coarse refinement

    csp -refine_parfile_tomo=`pwd`/frealign/EMPIAR-10164_original_volumes.txt     \
        -refine_model="EMPIAR-10164_init_ref.mrc"   \
        -particle_mw 300.0                          \
        -particle_rad 150.0                         \
        -particle_sym "C6"                          \
        -extract_box 192                            \
        -extract_bin 2                              \
        -extract_fmt frealign                       \
        -refine_fboost                              \
        -refine_maxiter 2                           \
        -refine_rhref "8.0"                         \
        -csp_UseImagesForRefinementMax 10           \
        -csp_refine_particles                       \
        -csp_NumberOfRandomIterations 50000         \
        -csp_ToleranceParticlesShifts 50.0          \
        -csp_ToleranceParticlesPhi 10.0             \
        -csp_ToleranceParticlesTheta 10.0           \
        -reconstruct_mintilt -50                    \
        -reconstruct_maxtilt 50

.. tip::

    - To only search for in-plane rotations (i.e., rotation angle Psi), set the tolerance of the other two rotations ``csp_ToleranceParticlesPhi`` and ``csp_ToleranceParticlesTheta`` to zero.
    - ``csp`` can also use initial alignments from other software packages such as Relion or EMAN. For example, see :doc:`Tomo import/export <tomo_import_export>` to import alignments from Relion.

Step 6: Fully constrained refinement
====================================

New, we do additional local refinement:

.. code-block:: bash

    # launch coarse refinement

    csp -refine_maxiter 5                           \
        -refine_rhref "8:10:8:6"                    \
        -csp_OptimizerStepLength 100.0              \
        -csp_NumberOfRandomIterations 0             \
        -csp_ToleranceParticlesShifts 20            \
        -csp_ToleranceParticlesPhi 20.0             \
        -csp_ToleranceParticlesPsi 20.0             \
        -csp_ToleranceParticlesTheta 20.0           \
        -csp_refine_micrographs                     \
        -dose_weighting_enable                      \
        -dose_weighting_fraction 4

All results from 3D refinement are saved in the folder ``frealign/maps``, including png files for visual inspection corresponding to each refinement iteration.

.. tip::

    Tolerance parameters determine the range used for searching. If you think particle alignments or tilt-series alignments are not accurate, you can increase the corresponding tolerances.

Step 7: Filter particles
========================

The next step is to remove particles with low correlation scores:

.. code-block:: bash

    mv frealign/maps frealign/fully_constrained && mkdir frealign/maps

    pcl -clean_parfile=`pwd`/frealign/fully_constrained/EMPIAR-10164_r01_05.bz2         \
        -clean_threshold 2.5                                                            \
        -clean_dist 10.0                                                                \
        -clean_mintilt -15.0                                                            \
        -clean_maxtilt 15.0                                                             \
        -clean_min_num_projections 1                                                    \
        -clean_check_reconstruction

Step 8 (optional): Permanently remove bad particles
===================================================

It is often a good idea to permanently remove any bad particles identified in the previous step:

.. code-block:: bash

    pcl -clean_discard


Step 9: Region-based refinement before masking
==============================================

The following command performs region-based constrained alignment:

.. code-block:: bash

    mv frealign/maps frealign/filter_particles && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/filter_particles/EMPIAR-10164_r01_02_clean.bz2       \
        -refine_model=`pwd`/frealign/filter_particles/EMPIAR-10164_r01_02.mrc"              \
        -particle_rad 100.0                                                                 \
        -extract_box 384                                                                    \
        -extract_bin 1                                                                      \
        -refine_iter 2                                                                      \
        -refine_maxiter 3                                                                   \
        -refine_rhref "6:5"                                                                 \
        -csp_UseImagesForRefinementMax 4                                                    \
        -csp_refine_particles                                                               \
        -csp_refine_micrographs                                                             \
        -csp_ToleranceParticlesShifts 20.0                                                  \
        -csp_Grid "8,8,2"


Step 10: Create shape mask
==========================

The next step is to create a shape mask:

.. code-block:: bash

    mv frealign/maps frealign/region_refine && mkdir frealign/maps

    pmk -mask_model=`pwd`/frealign/region_refine/EMPIAR-10164_r01_03.mrc     \
        -mask_threshold 0.42                                                 \
        -mask_normalized                                                     \
        -mask_edge_width 8


Step 11: Region-based refinement after masking
==============================================

Next, we do further refinement using the mask calculated in the previous step:

.. code-block:: bash

    mv frealign/maps frealign/mask && mv frealign/region_refine frealign/maps

    csp -refine_maxiter 6                               \
        -refine_rhref "6:5:5:4:3.5"                     \
        -refine_maskth=`pwd`/frealign/mask/mask.mrc"


Step 12: Particle-based CTF refinement
======================================

In this step we refine the CTF parameters on a per-particle basis:

.. code-block:: bash

    csp -refine_maxiter 7                                                       \
        -refine_rhref "3.1"                                                     \
        -no-csp_refine_micrographs                                              \
        -no-csp_refine_particles                                                \
        -csp_refine_ctf                                                         \
        -csp_UseImagesForRefinementMax 10


Step 13: Movie frame refinement
===============================

Next, we refine the raw movie frames against the most recent 3D reconstruction:

.. code-block:: bash

    mv frealign/maps frealign/ctf_refine && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/ctf_refine/EMPIAR-10164_r01_07.bz2       \
        -refine_model=`pwd`/frealign/ctf_refine/EMPIAR-10164_r01_07.mrc         \
        -particle_rad 80.0                                                      \
        -extract_fmt frealign_local                                             \
        -refine_iter 2                                                          \
        -refine_maxiter 2                                                       \
        -refine_rhref "3.2"                                                     \
        -csp_transreg                                                           \
        -no-csp_refine_ctf                                                      \
        -csp_spatial_sigma 200.0                                                \
        -csp_frame_refinement                                                   \
        -csp_UseImagesForRefinementMax 4


Step 14: Refinement after movie frame refinement
================================================

Using the refined frame averages for each tilt, we perform additional constrained refinement:

.. code-block:: bash

    csp -refine_maxiter 3                           \
        -refine_rhref "3.3"                         \
        -csp_refine_micrographs                     \
        -csp_refine_particles                       \
        -no-csp_frame_refinement                    \
        -csp_ToleranceMicrographShifts 10.0         \
        -csp_ToleranceMicrographTiltAngles 1.0      \
        -csp_ToleranceMicrographTiltAxisAngles 1.0  \
        -csp_ToleranceParticlesPsi 1.0              \
        -csp_ToleranceParticlesPhi 1.0              \
        -csp_ToleranceParticlesTheta 1.0            \
        -csp_ToleranceParticlesShifts 10.0          \
        -csp_RefineProjectionCutoff 2


Step 15: Map sharpening
=======================

The final step is to sharpen the map and produce FSC plots:

.. code-block:: bash

    psp -sharpen_input_map=`pwd`/frealign/maps/EMPIAR-10164_r01_half1.mrc   \
        -sharpen_automask_threshold 0.35                                    \
        -sharpen_adhoc_bfac -50
