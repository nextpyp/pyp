========================
Single-particle tutorial
========================

This tutorial shows how to process single-particle raw movies from `T20S proteasome (EMPIAR-10025) <https://www.ebi.ac.uk/empiar/EMPIAR-10025/>`_ into a high-resolution 3D structure.

1 Create a new project
======================

The first step is to create a new folder where all the data will be stored.

.. code-block:: bash

    # create new project

    mkdir T20S
    cd T20S

2 Pre-processing
================

Data pre-processing consists of doing movie frame alignment, CTF estimation and particle picking:

.. code-block:: bash

    # launch pre-processing

    pyp -data_mode spr                                  \
        -scope_voltage 300                              \
        -scope_pixel 0.66                               \
        -data_path="PATH_TO/spr_tutorial/14*.tif"       \
        -gain_reference="PATH_TO/spr_tutorial/Gain.mrc" \
        -gain_flipv                                     \
        -detect_rad 80                                  \
        -detect_method all                              \
        -detect_dist 40                                 \
        -slurm_tasks 7                                  \
        -slurm_memory 70                                \
        -slurm_merge_tasks 7                            \
        -slurm_merge_memory 70


The parameters for all pre-processing operations can be controlled using option groups of the form:
  - ``--movie_*``: options for movie frame alignment
  - ``--ctf_*``: options for estimation of the CTF
  - ``--detect_*``: options for particle picking

Some examples of options for data pre-processing are:
  - ``--movie_iters 5``: frame alignment iterations
  - ``--ctf_max_res 4``: max resolution used for CTF fitting
  - ``--detect_method auto``: particle picking method

The results of data pre-processing are saved in the ``ctf/`` and ``box/`` folders under the project directory (png files for visual inspection are saved in these locations as well).

.. tip::
    Use ``pyp --help`` to get a complete list of options. The parameter list is very long and is organized into groups to facilitate navigation. For example, all parameters to control gain correction are under ``-gain_*``.

3 Reference-based refinement
============================

This step runs coarse 3D refinement to assign particle orientations using an external reference as initial model. The default protocol for 3D refinement consists of running 4 iterations of global search:

.. code-block:: bash

    # launch coarse refinement

    csp -particle_mw 700.0            \
        -particle_rad 85.0            \
        -particle_sym "D7"            \
        -extract_box 128              \
        -extract_bin 4                \
        -extract_fmt frealign         \
        -refine_metric frealignx      \
        -refine_mode 0                \
        -refine_maxiter 5             \
        -refine_rhref  "8:7:6"        \
        -refine_fboost                \
        -reconstruct_cutoff "0"       \
        -refine_model PATH_TO/spr_tutorial/initial_model.mrc

Almost every aspect of 3D refinement, reconstruction and classification is configurable. This is done using groups of parameters similar to those used for ``pyp``. The main groups of options for ``csp`` are: 

- ``--extract_*``: options for particle extraction
- ``--refine_*``: options for orientation and translation search
- ``--reconstruct_*``: options for 3D reconstruction
- ``--class_*``: options for 3D classification
- ``--dose_weighting_*``: options for exposure weighting

These are some examples of options for ``csp``:

- ``--refine_iter 2``: first iteration of refinement
- ``--refine_maxiter 8``: total number of iterations
- ``--refine_rhref 4``: highest resolution to use for refinement
- ``--refine_metric frealignx``: version of frealign/cistem to use for refinement and reconstruction
- ``--refine_mode 1``: search mode can be global (0), local (1)
- ``--refine_mask "1,1,1,1,1"``: 5 search parameters are rotation angles phi, theta, psi, and shifts x, y (1: enable, 0: disable) 

All results from 3D refinement are saved in ``frealign/maps`` and include png files for each refinement iteration for visual inspection.

.. tip::
    For some ``csp`` parameters, a colon separated list of values can be provided to specify different values for each iteration. For example, ``--refine_rhref=12:10:8:4`` tells ``csp`` to use a 12A resolution cutoff during the first refinement iteration, 10A during the second iteration and so forth.

4 Filter bad particles
======================

This step remove bad particles based the refinement scores for the reconstruction:

.. code-block:: bash

    # make new project folder
    cd .. 
    mkdir TS20S_clean
    cd TS20S_clean

    # launch Filter bad particles 

    pcl -data_parent "PATH_TO/T20S"                                      \
        -clean_spr_auto                                                  \
        -clean_dist 20                                                   \
        -clean_parfile "PATH_TO/T20S/frealign/maps/T20S_r01_05.par.bz2"  \
        -clean_check_reconstruction                                      \
        -no-clean_discard                                                \
        -refine_model "PATH_TO/T20S/frealign/maps/T20S_r01_05.mrc"

5 Permanently remove bad particles
==================================

Remove bad particles from metadata permanently:

.. code-block:: bash
    
    pcl -clean_discard    \
        -no-clean_check_reconstruction


6 Particle refinement
=====================

The next step is to do local alignments using a lower level of binning and only using clean particles:

.. code-block:: bash

    # save coarse refinement results

    mv frealign/maps frealign/maps_clean

    # launch fine refinement

    csp -extract_box 256                                                            \
        -extract_bin 2                                                              \
        -refine_mode 1                                                              \
        -refine_metric new                                                          \
        -reconstruct_cutoff="1"                                                     \
        -refine_iter 2                                                              \
        -refine_rhref "6:4:3"                                                       \
        -refine_maxiter 6                                                           \
        -refine_parfile `pwd`/frealign/maps_clean/TS_20S_clean_r01_02_clean.par.bz2 \
        -refine_model `pwd`/frealign/maps_clean/TS_20S_clean_r01_02.mrc

.. note::
    Every time ``pyp`` or ``csp`` is ran, the value of all parameters are saved to the project directory in the ``.pyp_config.toml`` file. This means that when calling the program multiple times, you only need to specify the options that change from previous runs. For example, if you execute the ``pyp`` command above and then you want to use a smaller radius for particle detection, you would only need to run: ``pyp -particle_rad 60``. 

7 Create shape mask
===================

.. code-block:: bash

    pmk -mask_model `pwd`/frealign/maps/TS_20S_clean_r01_06.mrc  \
        -mask_threshold 0.3

8 Fine refinement
=================
.. code-block:: bash

    csp -refine_iter 7                               \
        -refine_maxiter 7                            \
        -refine_fboost                               \
        -refine_maskth `pwd`/frealign/maps/mask.mrc


9 Particle-based CTF refinement
==================

.. code-block:: bash

    csp -refine_maxiter 8       \
        -refine_csp_refine_ctf  \
        -refine_csp_Grid "8,8"


10 Movie frame refinement
========================

The step is to perform particle frame refinement that refines particle trajectories across frames:

.. code-block:: bash

    # save fine refinement results

    mv frealign/maps frealign/maps_fine

    # clean up frame metadata (NOTE: clean up again if you want to redo his step)

    rm csp/*local*

    # launch frame refinement

    csp -extract_fmt frealign_local                                             \
        -refine_rhref 3.0                                                       \
        -refine_iter 2                                                          \
        -refine_maxiter 3                                                       \
        -refine_skip                                                            \
        -csp_frame_refinement                                                   \
        -no-refine_rotreg                                                       \
        -refine_transreg                                                        \
        -refine_transreg_method spline                                          \
        -refine_spatial_sigma 15.0                                              \
        -refine_parfile  `pwd`/frealign/maps_refine/TS_20S_clean_r01_07.par.bz2 \
        -refine_model `pwd`/frealign/maps_refine/TS_20S_clean_r01_07.mrc        \
        -no-csp_refine_ctf


.. note::
    If the metadata associated with a given operation (e.g., frame alignment, CTF estimation, particle picking) already exists in the directory structure, that particular operation will be skipped and the information contained in the metadata will be used. If you change a parameter that affects CTF estimation for example, the metadata associated with the CTF ``ctf/*.ctf`` will be deleted so it can be recomputed using the new settings. If you change a parameter that affects the frame alignment routine, the metadata ``ali/*.xf`` will be deleted and the frames will be realigned using the new settings.


.. tip::

    A history of pyp commands used for each project is kept in the ``.pyp_history`` file. 


11 Dose weighting reconstruction
================================

The step is to perform dose-weighting reconstruction that maximizes the contribution from high-quality frames:

.. code-block:: bash

    # launch dose-weighting reconstruction

    csp -extract_fmt frealign_local     \
        -refine_metric new              \
        -dose_weighting_enable          \
        -dose_weighting_fraction 4      \
        -dose_weighting_transition 0.75 \
        -reconstruct_num_frames 38      \
        -refine_iter 4                  \
        -refine_maxiter 4               \
        -refine_skip                    \
        -no-csp_frame_refinement


12 Particle refinement on refined frame averages
================================================

The step is to refine particle rotation and translation on refined particle frames, which have higher SNR:

.. code-block:: bash


    # launch frame refinement

    csp -extract_fmt frealign_local     \
        -refine_rhref 3.0               \
        -refine_iter 5                  \
        -refine_maxiter 5               \
        -no-refine_skip                 \
        -no-csp_frame_refinement

.. note::
    After this step is done, repeating step 9 and step 11 for multiple iterations until convergence is encouraged. Please always enable dose weighting reconstruction to ensure the reference used for refinement is as high resolution as possible. 

13 Map sharpening
==================

Rename ``frealign/maps`` to ``frealign/frame_refine`` and create a new ``frealign/maps``

.. code-block:: bash

    psp -sharpen_input_map "frealign/frame_refine/*_r01_half1.mrc"      \
        -sharpen_automask_threshold 0.5                                 \
        -sharpen_adhoc_bfac -50
