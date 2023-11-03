===================
Tomography tutorial
===================

This tutorial shows how to process tilt-series from the `HIV-1 Gag (EMPIAR-10164) <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ dataset.

.. note::
    The following subset of tilt-series is used: TS_01, TS_03, TS_43, TS_45, and TS_54.

1 Create a new project
======================

.. code-block:: bash

    mkdir EMPIAR-10164
    cd EMPIAR-10164

2 Pre-processing
================

Data pre-processing consists of movie frame alignment, tilt-series alignment, tomogram reconstruction, CTF estimation and virion detection:

.. code-block:: bash

    # launch pre-processing
    pyp -data_mode tomo                                 \
        -scope_voltage 300                              \
        -scope_pixel 1.35                               \
        -data_path="/path_to_frames/*.tif"              \
        -ctf_max_res 5.0                                \
        -scope_tilt_axis 85.3                           \
        -tomo_rec_binning 8                             \
        -tomo_rec_thickness 2048                        \
        -tomo_vir_method "auto"                         \
        -tomo_vir_rad 500.0                             \
        -slurm_tasks 23                                 \
        -movie_pattern "TILTSERIES_SCANORD_ANGLE.tif"


.. note::
    You can redo desired steps by deleting corresponding files:
    - Frame alignment: ``ali/{name}.mrc``


3 (optional) Virion segmentation
=======================

Select the column with the yellow curve that best represents the membrane:

.. code-block:: bash

    ( export tomoedit=tomoedit &&
        singularity exec -B /your_binds --no-home -B $HOME/.ssh /path_to_container/pyp.sif
        /opt/pyp/bin/run/pyp -vir -skip )

.. tip::
    Select the *column* that best follows the membrane. To skip a virion, simply select the rightmost column (no spike search will be performed on these virions).


4 Particle detection
=================

Detect spikes using the membrane values selected above:

.. code-block:: bash

    pyp -data_mode tomo                         \
        -tomo_vir_detect_method "mesh"          \
        -tomo_vir_detect_dist 8                 \
        -tomo_vir_detect_band 800               \
        -tomo_spk_rad 50.0


5 Reference-based refinement
==============================

If a 3D reference is available, ``csp`` can align the particle projections using constrained refinement.

.. code-block:: bash

    # launch coarse refinement

    csp -refine_parfile="path_to_alignment.txt"     \
        -refine_model="EMPIAR-10164_init_ref.mrc"   \
        -particle_mw 300.0                          \
        -particle_rad 150.0                         \
        -particle_sym "C6"                          \
        -extract_box 192                            \
        -extract_bin 2                              \
        -extract_fmt frealign                       \
        -refine_skip                                \
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
    To only search one angle (i.e., psi), please set the tolerance of other refined rotations (i.e., ``csp_ToleranceParticlesPhi``, ``csp_ToleranceParticlesTheta``) to zero.

6 Fully constrained refinement
===============================================================

CSP can also use initial alignments from other software packages such as Relion or EMAN sub-volume averaging. You may find :doc:`Tomo import/export <tomo_import_export>` useful to perform sub-volume averaging in Relion.


.. note::
    Before exporting projects to Relion, you will need to run the following command to obtain an initial .par file.

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

All results from 3D refinement are saved in ``frealign/maps`` and include png files for each refinement iteration for visual inspection.

.. tip::
    The tolerance parameters determine the range used for searching, so if you think particle alignments or tilt-series alignments are not precise, you will need to increase the corresponding tolerances.


7 Filter particles
===============================

.. code-block:: bash
    
    mv frealign/maps frealign/fully_constrained && mkdir frealign/maps

    pcl -clean_parfile `pwd`/frealign/fully_constrained/EMPIAR-10164_r01_05.par.bz2     \
        -clean_threshold 2.5                                                            \
        -clean_dist 10.0                                                                \
        -clean_mintilt -15.0                                                            \
        -clean_maxtilt 15.0                                                             \
        -clean_min_num_projections 1                                                    \
        -clean_check_reconstruction



8  (optional): Permanently remove bad particles
================

.. code-block:: bash

    pcl -clean_discard


9 Region-based local refinement before masking
==================

.. code-block:: bash
    
    mv frealign/maps frealign/filter_particles && mkdir frealign/maps

    csp -refine_parfile `pwd`/frealign/filter_particles/EMPIAR-10164_r01_02_clean.par.bz2   \
        -refine_model `pwd`/frealign/filter_particles/EMPIAR-10164_r01_02.mrc"              \
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


10 Create shape mask
====================================

.. code-block:: bash
    
    mv frealign/maps frealign/region_refine && mkdir frealign/maps

    pmk -mask_model `pwd`/frealign/region_refine/EMPIAR-10164_r01_03.mrc    \
        -mask_threshold 0.42                                                \
        -mask_normalized                                                    \
        -mask_edge_width 8


11 Region-based local refinement
==================

.. code-block:: bash

    mv frealign/maps frealign/mask && mv frealign/region_refine frealign/maps

    csp -refine_maxiter 6                               \
        -refine_rhref "6:5:5:4:3.5"                     \
        -refine_maskth `pwd`/frealign/mask/mask.mrc"


12 Particle-based CTF refinement
==================

.. code-block:: bash

    csp -refine_maxiter 7                                                       \
        -refine_rhref "3.1"                                                     \
        -no-csp_refine_micrographs                                              \
        -no-csp_refine_particles                                                \
        -csp_refine_ctf                                                         \
        -csp_UseImagesForRefinementMax 10


13 Movie frame refinement
==================

.. code-block:: bash
    
    mv frealign/maps frealign/ctf_refine && mkdir frealign/maps

    csp -refine_parfile `pwd`/frealign/ctf_refine/EMPIAR-10164_r01_07.par.bz2   \
        -refine_model `pwd`/frealign/ctf_refine/EMPIAR-10164_r01_07.mrc         \
        -particle_rad 80.0                                                      \
        -extract_fmt frealign_local                                             \
        -refine_iter 2                                                          \
        -refine_maxiter 2                                                       \
        -refine_rhref "3.2"                                                     \
        -refine_spatial_sigma 200.0                                             \
        -refine_transreg                                                        \
        -no-csp_refine_ctf                                                      \
        -csp_frame_refinement                                                   \
        -csp_UseImagesForRefinementMax 4


14 Refinement after movie frame refinement
==================

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


15 Map sharpening
==================

.. code-block:: bash
    
    mv frealign/maps frealign/frame_refine && mkdir frealign/maps

    psp -sharpen_input_map `pwd`/frealign/frame_refine/EMPIAR-10164_r01_half1.mrc   \
        -sharpen_automask_threshold 0.35                                            \
        -sharpen_adhoc_bfac -50
