=======================
Classification tutorial
=======================

This tutorial shows how to convert raw tilt-series from `EMPIAR-10304 (E. coli. ribosomes) <https://www.ebi.ac.uk/empiar/EMPIAR-10304/>`_ into a ~4.9A resolution structure and resolve 8 different conformations.

We first use the command below to download and decompress a tbz file containing: 1) a script to download the raw tilt-series from EMPIAR, 2) corresponding metadata with tilt angles and acquisition order, and 3) an initial model:

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_class_tutorial.tbz
  tar xfz nextpyp_class_tutorial.tbz
  source download_10304.sh

.. note::

  Downloading the raw data from EMPIAR can take several minutes.

Step 1: Create a new project
============================

Next, we create an empty folder where all files for the tutorial will be saved:

.. code-block:: bash

    mkdir EMPIAR-10304
    cd EMPIAR-10304

Step 2: Pre-processing
======================

The next command does data pre-processing consisting of movie frame alignment, tilt-series alignment, tomogram reconstruction, CTF estimation:

.. code-block:: bash

    # launch pre-processing
    pyp -data_path="/path_to_raw_data/tilt*.mrc"    \
        -data_mode tomo                             \
        -scope_pixel 2.1                            \
        -scope_voltage 300                          \
        -scope_tilt_axis 0.0                        \
        -movie_no_frames                            \
        -ctf_max_res 5.0                            \
        -tomo_rec_binning 12                        \
        -tomo_rec_thickness 3072                    \
        -no-tomo_rec_format                         \
        -tomo_rec_erase_fiducials                   \
        -slurm_tasks 42                             \
        -slurm_memory 420

.. note::

    Nominal tilt angles (stored in ``*.rawtlt`` files) and acquisition order (stored in ``*.order`` files) for each tilt-series are provided with the raw data.

Step 3: Particle detection
==========================

The next step is to detect ribosome particles using a size-based approach:

.. code-block:: bash

    pyp -tomo_spk_method "auto"             \
        -tomo_spk_rad 80                    \
        -tomo_spk_stdtimes_cont_3d 2.0      \
        -tomo_spk_min_size_3d 60            \
        -tomo_spk_dilation_3d 100           \
        -tomo_spk_radiustimes_3d 2.0        \
        -tomo_spk_inhibit_3d                \
        -tomo_spk_stdtimes_filt_3d 2.0      \
        -tomo_spk_detection_width_3d 40.0


Step 4: Reference-based refinement
==================================

If a 3D reference is available, we use the ``csp`` command to align particle projections using constrained refinement:

.. code-block:: bash

    # launch coarse refinement
    csp -refine_parfile=`pwd`/frealign/EMPIAR-10304_original_volumes.txt    \
        -refine_model="/path_to_raw_data/10304_ref_bin4.mrc"                \
        -particle_mw 2000                               \
        -particle_rad 150                               \
        -extract_box 64                                 \
        -extract_bin 4                                  \
        -refine_skip                                    \
        -extract_fmt frealign                           \
        -refine_rhref "22.0"                            \
        -refine_fboost                                  \
        -reconstruct_mintilt -50                        \
        -reconstruct_maxtilt 50                         \
        -csp_ctf_handedness                             \
        -csp_refine_particles                           \
        -csp_UseImagesForRefinementMin 15               \
        -csp_UseImagesForRefinementMax 25               \
        -csp_NumberOfRandomIterations 5000000           \
        -csp_ToleranceParticlesPhi 180.0                \
        -csp_ToleranceParticlesTheta 180.0              \
        -csp_ToleranceParticlesPsi 180.0                \
        -csp_ToleranceParticlesShifts 50.0

Step 5: Filter particles
========================

The next step is to remove particles with low correlation scores:

.. code-block:: bash

    mv frealign/mapsfrealign/reference_based && mkdir frealign/maps

    pcl -clean_parfile=`pwd`/frealign/reference_based/EMPIAR-10304_r01_02.par.bz2   \
        -clean_threshold 15.0                                                       \
        -clean_dist 20.0                                                            \
        -clean_mintilt -7.0                                                         \
        -clean_maxtilt 7.0                                                          \
        -clean_min_num_projections 1                                                \
        -clean_check_reconstruction

Step 6  (optional): Permanently remove bad particles
====================================================

It is often a good idea to permanently remove any bad particles identified in the previous step:

.. code-block:: bash

    pcl -clean_discard


Step 7: Fully constrained refinement
====================================

In this step we do additional refinement using the raw data (without binning):

.. code-block:: bash

    mv frealign/maps frealign/particle_filter && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/particle_filter/EMPIAR-10304_r01_02_clean.par.bz2    \
        -refine_model=`pwd`/frealign/particle_filter/EMPIAR-10304_r01_02.mrc                \
        -extract_box 256                                                                    \
        -extract_bin 1                                                                      \
        -refine_skip                                                                        \
        -refine_iter 2                                                                      \
        -refine_maxiter 3                                                                   \
        -refine_rhref "18:14"                                                               \
        -csp_refine_micrographs                                                             \
        -csp_OptimizerStepLength 100.0                                                      \
        -csp_UseImagesForRefinementMin 15                                                   \
        -csp_UseImagesForRefinementMax 25                                                   \
        -csp_NumberOfRandomIterations 0                                                     \
        -csp_ToleranceParticlesPsi 30.0                                                     \
        -csp_ToleranceParticlesPhi 30.0                                                     \
        -csp_ToleranceParticlesTheta 30.0                                                   \
        -csp_ToleranceParticlesShifts 30.0                                                  \
        -dose_weighting_enable                                                              \
        -dose_weighting_fraction 4                                                          \
        -dose_weighting_global

All results from 3D refinement are saved in the folder ``frealign/maps``, including png files for visual inspection corresponding to each refinement iteration.

Step 8: Create shape mask
=========================

The next step is to create a shape mask:

.. code-block:: bash

    mv frealign/maps frealign/fully_constrained && mkdir frealign/maps

    pmk -mask_model=`pwd`/frealign/fully_constrained/EMPIAR-10304_r01_03.mrc    \
        -mask_threshold 0.4                                                     \
        -mask_normalized                                                        \
        -mask_edge_width 8


Step 9: Region-based local refinement
=====================================

The following command performs region-based constrained alignment:

.. code-block:: bash

    mv frealign/maps frealign/mask && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/fully_constrained/EMPIAR-10304_r01_03.par.bz2    \
        -refine_model=`pwd`/frealign/fully_constrained/EMPIAR-10304_r01_03.mrc          \
        -refine_maskth=`pwd`/frealign/mask/mask.mrc"                                    \
        -refine_iter 2                                                                  \
        -refine_maxiter 6                                                               \
        -refine_rhref "12:10:8:6:5"                                                     \
        -csp_UseImagesForRefinementMin 18                                               \
        -csp_UseImagesForRefinementMax 22                                               \
        -csp_ToleranceMicrographTiltAngles 5.0                                          \
        -csp_ToleranceMicrographTiltAxisAngles 5.0                                      \
        -csp_ToleranceParticlesPsi 5.0                                                  \
        -csp_ToleranceParticlesPhi 5.0                                                  \
        -csp_ToleranceParticlesTheta 5.0                                                \
        -csp_ToleranceParticlesShifts 20.0                                              \
        -csp_Grid "8,8,2"


Step 10: Particle-based CTF refinement
======================================

In this step we refine the CTF parameters on a per-particle basis:

.. code-block:: bash

    mv frealign/maps frealign/region_based && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/region_based/EMPIAR-10304_r01_06.par.bz2     \
        -refine_model=`pwd`/frealign/region_based/EMPIAR-10304_r01_06.mrc           \
        -refine_iter 2                                                              \
        -refine_maxiter 2                                                           \
        -refine_rhref "4.5"                                                         \
        -no-csp_refine_particles                                                    \
        -no-csp_refine_micrographs                                                  \
        -csp_refine_ctf                                                             \
        -csp_UseImagesForRefinementMin 15                                           \
        -csp_UseImagesForRefinementMax 25                                           \
        -csp_ToleranceMicrographDefocus1 2000                                       \
        -csp_ToleranceMicrographDefocus2 2000

Step 11: Additional region-based refinement after CTF refinement
================================================================

The following command does additional region-based refinement:

.. code-block:: bash

    mv frealign/maps frealign/ctf_refine && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/ctf_refine/EMPIAR-10304_r01_02.par.bz2   \
        -refine_model=`pwd`/frealign/ctf_refine/EMPIAR-10304_r01_02.mrc         \
        -refine_iter 2                                                          \
        -refine_maxiter 4                                                       \
        -refine_rhref "6:5:4.5"                                                 \
        -csp_refine_particles                                                   \
        -csp_refine_micrographs                                                 \
        -no-csp_refine_ctf                                                      \
        -csp_OptimizerStepLength 20.0                                           \
        -csp_UseImagesForRefinementMin 18                                       \
        -csp_UseImagesForRefinementMax 22                                       \
        -csp_ToleranceMicrographShifts 20.0                                     \
        -csp_Grid "16,16,4"                                                     \
        -dose_weighting_fraction 2


Step 12: 3D classification
==========================

In the last step we perform 3D classification into 8 classes:

.. code-block:: bash

    mv frealign/maps frealign/region_based_2 && mkdir frealign/maps

    csp -refine_parfile=`pwd`/frealign/region_based_2/EMPIAR-10304_r01_04.par.bz2   \
        -refine_model=`pwd`/frealign/region_based_2/EMPIAR-10304_r01_04.mrc         \
        -refine_iter 2                                                              \
        -refine_maxiter 20                                                          \
        -no-refine_skip                                                             \
        -refine_fboost                                                              \
        -refine_rhref "8"                                                           \
        -no-csp_refine_particles                                                    \
        -no-csp_refine_micrographs                                                  \
        -class_num 8                                                                \
        -class_rhcls 8.0                                                            \
        -dose_weighting_weights=`pwd`/frealign/weights/global_weight.txt"

All results will be saved in the ``frealign/maps`` folder.

.. seealso::

    * :doc:`Tomography tutorial<tomography>`
    * :doc:`Single-particle tutorial<single_particle>`
