Single-Particle Session
=======================

``PYP`` can be run during data collection to provide live feedback on data quality.

The daemon carries out three functions:

    1) Transfer and compress the raw data from the camera computer to a target destination (typically on an HPC system)
    2) Pre-processing of movies/tilt-series (including frame alignment, tilt-series alignment, CTF estimation and particle picking)
    3) 2D classification

The preferred way to run ``streampyp`` is to have the hard drive from the camera computer mounted on the HPC file system (access through ssh is not supported).

Launching ``streampyp``
-----------------------

The simplest way to run the program, is to provide: 1) the absolute path to the raw data, 2) the absolute path to the gain reference, and 3) an output directory:

.. code-block:: bash

    # launch streampyp
    streampyp 
        -scope_pixel 1.08                      \
        -scope_voltage 300                     \
        -gain_reference=gainPath               \
        -data_path=dataPath                    \
        -stream_transfer_target=outputName     \
        -stream_session_name sessionName       \
        -stream_session_group groupName        \
        -stream_transfer_remove                \
        -detect_rad 100                        \
        -class2d_num 50                        \
        -class2d_box 128                       \
        -class2d_bin 2                         \
        -slurm_tasks 7                         \
        -slurm_memory 14                       \
        -slurm_class2d_tasks 70                \
        -slurm_class2d_memory 140

.. note::

    * Raw and meta data produced during pre-processing are stored in the output directory specified using the parameter ``-stream_transfer_target``. Sub-folders with the group and session names will be automatically created under the output path, for example: ``${output_dir}/groupName/sessionName``.

.. important::

    One drawback of running ``streampyp`` from the command line is that it needs to stay running for the duration of the session. This is typically achieved using programs like `screen <https://www.gnu.org/software/screen/manual/screen.html>`_ or `tmux <https://github.com/tmux/tmux/wiki>`_.