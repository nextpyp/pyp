========================
Single-particle tutorial
========================

This tutorial shows how to process single-particle raw movies from `T20S proteasome (EMPIAR-10025) <https://www.ebi.ac.uk/empiar/EMPIAR-10025/>`_ into a ~3A resolution structure.

We first download and decompress a ``.tbz`` file containing a subset of 20 movies, the gain reference, and an initial model:

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_spr_tutorial.tbz
  tar xvfz nextpyp_spr_tutorial.tbz


Step 1: Create a new project
============================

The first step is to create a new folder where all the data will be stored.

.. code-block:: bash

    # create new project

    mkdir T20S
    cd T20S

Step 2: Pre-processing
======================

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

The results of data pre-processing are saved in the ``pkl/`` and ``webp/`` folders under the project directory.

.. tip::
    Use ``pyp --help`` to get a complete list of options. The parameter list is very long and is organized into groups to facilitate navigation. For example, all parameters to control gain correction are under ``-gain_*``.

Step 3: Reference-based refinement
==================================

This step runs coarse 3D refinement to assign particle orientations using an external reference as initial model. The default protocol for 3D refinement consists of running 4 iterations of global search:

.. code-block:: bash

    # launch coarse refinement

    csp -particle_mw 700.0            \
        -particle_rad 85.0            \
        -particle_sym "D7"            \
        -extract_box 128              \
        -extract_bin 4                \
        -extract_fmt frealign         \
        -refine_mode 0                \
        -refine_maxiter 4             \
        -refine_rhref "8:7:6"         \
        -refine_fboost                \
        -no-refine_skip               \
        -no-refine_fssnr              \
        -no-refine_priors             \
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
    For some ``csp`` parameters, a colon separated list of values can be provided to specify different values for each iteration. For example, ``--refine_rhref="12:10:8:4"`` tells ``csp`` to use a 12A resolution cutoff during the first refinement iteration, 10A during the second iteration and so forth.

Step 4: Filter bad particles
============================

This step removes bad particles based on assigned particle scores during refinement. We first need to create a new ``T20S_clean`` folder:

.. code-block:: bash

    # make new project folder
    
    cd ..
    mkdir T20S_clean
    cd T20S_clean

    # filter bad particles

    pcl -data_parent=`pwd`/../T20S                                   \
        -clean_spr_auto                                              \
        -clean_dist 20                                               \
        -clean_parfile=`pwd`/../T20S/frealign/maps/T20S_r01_04.bz2   \
        -clean_check_reconstruction                                  \
        -no-clean_discard                                            \
        -refine_model=`pwd`/../T20S/frealign/maps/T20S_r01_04.mrc

.. tip::
    Check the results in the ``frealign/maps`` folder to confirm that the filtering operation was successful.

Step 5: Permanently remove bad particles
========================================

Remove bad particles from metadata (this step cannot be undone):

.. code-block:: bash

    pcl -clean_discard                      \
        -no-clean_check_reconstruction


Step 6: Particle refinement
===========================

The next step is to do local alignments using a lower level of binning (using only clean particles). We first need to rename ``frealign/maps`` to ``frealign/maps_clean``:

.. code-block:: bash

    # save coarse refinement results

    mv frealign/maps frealign/maps_clean

    # launch fine refinement

    csp -extract_box 256                                                            \
        -extract_bin 2                                                              \
        -refine_mode 1                                                              \
        -reconstruct_cutoff="1"                                                     \
        -refine_iter 2                                                              \
        -refine_rhref "6:4:3"                                                       \
        -refine_maxiter 6                                                           \
        -refine_fboost                                                              \
        -no-refine_skip                                                             \
        -refine_parfile=`pwd`/frealign/maps_clean/T20S_clean_r01_02_clean.bz2       \
        -refine_model=`pwd`/frealign/maps_clean/T20S_clean_r01_02.mrc

.. note::
    Every time ``pyp`` commands are executed, the parameters are saved in a ``.pyp_config.toml`` file in the project directory. This means that parameter values are "remembered" and you only need to specify the ones that change between consecutive runs. For example, if you executed the ``csp`` command above and you want to run an additional refinement iteration, you can just run: ``csp -refine_maxiter 7``.

Step 7: Create shape mask
=========================

This step will create a shape mask using the most recent reconstruction:

.. code-block:: bash

    pmk -mask_model=`pwd`/frealign/maps/T20S_clean_r01_06.mrc  \
        -mask_threshold 0.3

Step 8: Fine refinement
=======================

Next, we will perform additional refinement iterations using the shape mask:

.. code-block:: bash

    csp -refine_iter 7                               \
        -refine_maxiter 8                            \
        -refine_maskth=`pwd`/frealign/maps/mask.mrc


Step 9: Particle-based CTF refinement
=====================================

This step refines the CTF per-particle using an 8x8 grid:

.. code-block:: bash

    csp -refine_maxiter 9       \
        -csp_refine_ctf         \
        -csp_Grid_spr "8,8"

Step 10: Movie frame refinement
===============================

This step refines shifts for movie frames of each particle using the most recent 3D reconstruction as reference. We first need to rename ``frealign/maps`` to ``frealign/maps_fine``:

.. code-block:: bash

    # save fine refinement results

    mv frealign/maps frealign/maps_fine

    # launch frame refinement

    csp -extract_fmt frealign_local                                             \
        -refine_rhref "3.0"                                                     \
        -refine_iter 2                                                          \
        -refine_maxiter 3                                                       \
        -refine_skip                                                            \
        -refine_parfile=`pwd`/frealign/maps_fine/T20S_clean_r01_09.bz2          \
        -refine_model=`pwd`/frealign/maps_fine/T20S_clean_r01_09.mrc            \
        -csp_frame_refinement                                                   \
        -csp_UseImagesForRefinementMax 60                                       \
        -csp_transreg                                                           \
        -csp_spatial_sigma 15.0                                                 \
        -no-csp_refine_ctf

.. note::

    If the metadata associated with a given operation (e.g., frame alignment, CTF estimation, particle picking) already exists in the directory structure, that particular operation will be skipped and the information contained in the metadata will be used. If you change a parameter that affects CTF estimation for example, the metadata associated with the CTF will be deleted so it can be recomputed using the new settings. If you change a parameter that affects the frame alignment routine, the corresponding metadata will be deleted and the frames will be realigned using the new settings.

.. tip::

    A history of commands issued for each project is kept in the ``.pyp_history`` file.


Step 11: Dose weighting
=======================

This step performs per-frame dose-weighting to increase the contribution of high-quality frames:

.. code-block:: bash

    # launch dose-weighting reconstruction

    csp -extract_fmt frealign_local     \
        -dose_weighting_enable          \
        -dose_weighting_fraction 4      \
        -dose_weighting_transition 0.75 \
        -refine_iter 4                  \
        -refine_maxiter 4               \
        -no-csp_frame_refinement


Step 12: Particle refinement after frame alignment
==================================================

This step does additional 3D refinement using the drift-corrected particles and the dose-weighted reconstruction:

.. code-block:: bash

    # launch frame refinement

    csp -refine_iter 5                  \
        -refine_maxiter 5               \
        -no-refine_skip

Step 13: Map sharpening
=======================

The final step does masking, sharpening, and produces FSC resolution plots:

.. code-block:: bash

    psp -sharpen_input_map=`pwd`/frealign/frame/*_r01_half1.mrc  \
        -sharpen_automask_threshold 0.5                          \
        -sharpen_adhoc_bfac -50

.. note::

    Output maps and FSC plots will be saved in the ``frealign/maps`` folder.
