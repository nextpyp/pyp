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

..

    Using a configuration file (optional)
    -------------------------------------

    ``streampyp`` can also be configured through the ``config.toml`` configuration file. The following is a typical example:

    .. code-block:: toml

        # This section controls streampyp
        [stream]

        # Directory used for storing the raw data and pre-processing results
        targetPath = '/work/cryoem'

        # Microscope parameters
        [stream.krios]

        # scope voltage
        voltage = 300

        # k3 camera parameters
        [stream.krios.k3]

        # Root directory where images from this detector are saved
        path = '/cifs/Krios-gatan'

        # Apply a vertical flip to the gain reference before multiplication
        gain_flipv = true

        # falcon camera parameters
        [stream.krios.falcon]

        # Root directory where images from this detector are saved
        path = '/cifs/Krios-EPUData/MMC,/cifs/Krios-OffloadData'

        # Do not apply a vertical flip to the gain reference before multiplication
        gain_flipv = false

    .. note::

        You can add additional microscope and detector configurations by simply adding new sections to the configuration file.

    .. tip::

        Multiple comma separated paths can be specified for each detector in the ``path`` variable. This is useful for cases where the movie frames are saved separately from the frame averages.

    When the ``stream`` section is provided, the ``streampyp`` can be run as follows:

    .. code-block:: bash

        # launch streampyp
        streampyp 
            -scope_pixel 1.08                  \
            -stream_session_name sessionName   \
            -stream_session_group groupName    \
            -data_mode spr                     \
            -stream_transfer_remove            \
            -detect_rad 100

    .. note::

        * The ``-session_name`` option should match exactly the name of the folder where the data is saved. For example, if raw data is saved to ``/cifs/Krios/K3/sessionName``, you need to use ``-session_name sessionName``.

        * The ``-session_group`` option determines the name of the folder where the data will be saved. This allows to keep all sessions from a specific user under the same folder.

    .. important::

        The entry ``[stream][target]`` specifies the output directory (same as ``-output_dir`` above) and there are options to specify different paths for each microscope/detector combinations. The values for the ``-stream_scope_profile`` and ``-stream_camera_profile`` options should have matching entries in the ``config/config.toml`` file. For example, if using ``-stream_scope_profile krios``, there should be an entry ``[stream][krios]`` in the configuration file.

        One drawback of running ``streampyp`` from the command line is that it needs to stay running for the duration of the data collection session. This is typically achieved using programs like `screen <https://www.gnu.org/software/screen/manual/screen.html>`_ or `tmux <https://github.com/tmux/tmux/wiki>`_. Once the web interface of pyp is released, this will no longer be needed.

    .. tip::

        For better responsiveness, some SLURM instances offer a special partition for shorter/quicker jobs. To take advantage of this feature, you need to specify an appropriate value for the ``[slurm][queue]`` entry in the ``config/config.toml`` file. A typical value would be ``--partition=quick``.

        ``streampyp`` has additional options to control its behavior. You can get the complete list by doing ``streampyp -h``.
