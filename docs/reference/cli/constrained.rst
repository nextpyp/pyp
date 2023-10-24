.. warning::

    Not up to date

=============================================
Constrained single-particle refinement (CSPR)
=============================================

Description
===========

Constrained single-particle refinement (CSPR) can be used to improve resolution after an initial map is obtained using either single-particle analysis (SPA) or sub-volume averaging (SVA). CSPR requires extracting individual particle frames (SPA) or projections from the tilt-series (SVA). Depending on how many particles were used for 3D reconstruction, the resulting number of projections can be prohibitely large. PYP solves this problem by bypassing the need to store the intermediate tilt/frame stacks and directly produces partial reconstructions that can later be merged into a final map. 

.. note::
    For a dataset of 10,000 sub-volumes and 41 images per tilt-series, the total number of particle projections will be 410,000. Producing particle stacks of this size will have a large storage footprint and introduce delays associated with I/O that can slow down the refinement process. 

Preparation
===========

CSPR uses previously obtained particle alignment parameters to build an initial model and use it as reference for further refinement. The alignments can be obtained with external tools, but the user is responsible for converting the format of the metadata to match PYP's convention.

Execution
=========

Once the SPA/SVA alignment information has been obtained, CSPR can be launched using the following command:

.. code-block:: bash
    
    # run CSPR
    cd my_project
    csp -csp_refine "path_to_sva_alignments.txt"    \
        -csp_model "path_to_sva_map.mrc"            \
        -particle_fmt frealign                      \
        -tasks_per_arr 2 )

.. note::
    * After running this command, the only output you will see is the reconstruction in the ``frealign/maps`` folder and the corresponding metadata in the ``ali/`` folder. CSPR will use the existing configuration files ``.pyp_config``, ``frealign.config``, and ``parameters.config`` present in the project directory.
    
    * If you want to run multiple iterations, the ``frealing.parameter`` file settings will be used to control the execution (the alignments from the previous iteration will be saved automatically in the `ali/` folder. If you want to start fresh and ignore any previous alignments, you can run ``csp -csp_skip True``.
    
.. tip::

    With the option ``-tasks_per_arr`` you can specify how many micrographs, movies or tilt-series you want process in each job array. If you have lots of particles per movie or tilt-series, you will want to use a smaller ``-tasks_per_arr`` to prevent filling up the local scratch of the nodes.

Troubleshooting
===============

Problems with running CSPR are difficult to track down due to the `concealed` nature of the execution and the many optimizations done to make the code run faster. The best strategy to debug the code is to run CSPR in interactive mode.

Compute resources
=================

CSPR runs in parallel using PYP's native SLURM and MPI support and may require significant resources depending on the size of the dataset and the CPUs available.
