=================
Compute resources
=================

``nextPYP`` can be run in a **Desktop workstation** or a **Compute cluster**. A Desktop workstation is simpler to setup and can be used to process small to medium sized datasets. For most datasets, however, a compute cluster will enable significantly faster processing.

.. tab-set::
  :sync-group: running_mode

  .. tab-item:: Desktop workstation
    :sync: standalone

    ``nextPYP`` autmatically detects what resources are installed locally in the server (number of CPU cores and GPU cards) and allocates jobs accordingly.

  .. tab-item:: Compute cluster
    :sync: cluster

    In this mode, resources are managed by the `SLURM <https://slurm.schedmd.com/>`_ scheduler. Commonly managed resources include CPU threads, RAM, GPUs, and local scratch space.

Resource management
-------------------

``nextPYP`` uses three types of processes:

- **Thread**: A single-threaded process running on one CPU core (or hyper-threaded core)
- **Task**: A set of *threads* used to process a single micrograph or tilt-series
- **Array**: A set of *tasks* used to process an entire dataset

Most jobs in ``nextPYP`` consist of three phases: *Launch*, *Split* and *Merge*. The *Launch* phase is typically a lightweigth task used to initiate data processing. In the *Split* phase, multiple tasks run in parallel (typically one for each micrograph/tilt-series). Finally,the *Merge* phase is a task that consolidates the results from the *Split* phase, such as combining intermediate data to generate a single 3D reconstruction. Each phase has distinct computational needs, so resources are allocated separately for each one.

Real time information about jobs is available in the `Jobs panel <../guide/overview.html#jobs-panel>`_.

Resource allocation
-------------------

Each processing block in ``nextPYP`` includes a **Resources** tab where you can allocate resources for each phase of a job.

.. figure:: ../images/tutorial_tomo_pre_process_jobs.webp
  :alt: Job submission options

The **Resources** tab is divided into three sections, one for each phase (*Launch*, *Split* and *Merge*):

.. comment:
   Looks like we're using sphinx-design for panels now?
   The panels in sphinx-design seem to be a bit different than panels from our old lib, sphinx-panels.
   See: https://sphinx-design.readthedocs.io/en/pydata-theme/dropdowns.html

.. nextpyp:: Launch task options
  :collapsible: open

  Launch, Threads
    Number of threads used when launching jobs.

    **Default**: 1

  Launch, Memory per thread (GB)
    Amount of memory per thread requested when launching jobs.

    **Default**: 4

  Launch, Walltime (dd-hh:mm:ss)
    Set a limit on the total run time when launching jobs. When the time limit is reached, SLURM will terminate the job.

    **Default**: 1-00:00:00

  Split, Gres
    List of Generic resources (Gres) to request for the split job. This option can be used to request specific resources available in your SLURM instance, for example, ``lscratch:500,shrd=64```, which requests 500 GB of local scratch space and 64 GB of shared memory. The ``lscratch`` resource is typically used for temporary storage, while the ``shrd`` resource is used for shared memory. To check the available resources in your SLURM instance, run the command: ``sinfo -o "%100N  %30G"``. If you are unsure about this, please contact your system administrator.

    **Default**: None

.. nextpyp:: Split task options
  :collapsible: open

  Split, Threads
    Number of threads used to process a micrograph or tilt-series.
  
    **Default**: 1
    
  Split, Total threads
    Maximum number of threads to run simultaneously. Setting it to ``0`` removes any limits, deferring entirely to SLURM’s limits. This option can help manage how resources are distributed between multiple ``nextPYP`` jobs. For example, if the number of threads is set to 7 and the total number of threads is set to 21, then 3 jobs will be run simultaneously, each using 7 threads. If the total number of threads is set to ``0``, then SLURM will determine how many jobs to run simultaneously based on the available resources and any account quotas.

    **Default**: 0
  
  Split, Memory per thread (GB)
    Amount of memory per thread requested for each split task.
  
    **Default**: 4
    
  Split, Walltime (dd-hh:mm:ss)
    Set a limit on the total run time for each split task. When the time limit is reached, SLURM will terminate the job.

    **Default**: 1-00:00:00
    
  Split, Bundle size
    Number of tasks to group into a bundle. Tasks within a bundle are processed one after the other, sequentially. For example, if there are 100 tasks and the bundle size is set to 10, then 10 jobs with 10 tasks each will be processed in parallel. This option can help manage how resources are distributed and potentially reduce storage utilization.

    **Default**:  1

  Split, Gres
    List of Generic resources (Gres) to request for the split job.

    **Default**: None

.. nextpyp:: Merge task options
  :collapsible: open

  Merge, Threads
    Number of threads used to run the merge task.
  
    **Default**: 1

  Merge, Memory per thread (GB)
    Amount of memory per thread used to run the merge task.

    **Default**: 4

  Merge, Walltime (dd-hh:mm:ss)
    Set a limit on the total run time for the merge task. When the time limit is reached, SLURM will terminate the job.

    **Default**: 1-00:00:00

  Split, Gres
    List of Generic resources (Gres) to request for the merge task.

    **Default**: None

.. warning::
    Users are responsible for ensuring that the requested combination of resources is available in the HPC environment where ``nextPYP`` is running. If the requested resource combination is unavailable, the job will be left in a ``PENDING`` state, potentially indefinitely. To fix this, users can cancel the job and resubmit it with a different combination of resources.
    
.. tip::
    To check the status of a job, go to the **Jobs** panel, click on the :fa:`file-alt text-primary` icon next to the job, and select the **Launch** tab.

GPU resources
-------------

Although the core functionality of ``nextPYP`` operates exclusively on CPUs, certain operations do require GPU access. In most cases, users cannot choose between running jobs on CPUs or GPUs, this is determined by the specific requirements of each job. Only a few exceptions exist, and in those cases, a checkbox will be available to enable or disable GPU usage.

List of programs and operations that require GPUs:

- **Particle picking**: Neural network-based article picking (training and inference)
- **MiLoPYP**: Cellular pattern mining and localization (training and inference)
- **MotionCor3**: Motion correction of micrographs or tilt movies
- **AreTomo2**: Tilt-series alignment and tomographic reconstruction
- **Membrain-seg**: Tomogram segmentation using pre-trained neural networks
- **Topaz**: Tomogram denoising using pre-trained neural networks
- **IsoNet**: Tomogram denoising using neural networks (training and inference)
- **CryoCARE**: Tomogram denoising using neural networks (training and inference)
- **Pytom-match-pick**: Particle picking using template matching
- **tomoDRGN**: Heterogeneity analysis using neural networks (training and inference)

Jobs that use any of the above programs will be submitted to the SLURM scheduler using the ``--gres=gpu:1`` option. This means that one GPU will be requested for each job.

Specific GPUs
^^^^^^^^^^^^^

.. tab-set::
  :sync-group: running_mode

  .. tab-item:: Desktop workstation
    :sync: standalone

    In this mode, there is usually only one type of GPU available, so specifying GPU types is unnecessary.

  .. tab-item:: Compute cluster
    :sync: cluster

    To run a job on a specific GPU resource, users can set the ``Split, Gres`` parameter in the **Resources** tab of a block. For example, to use an H100 card, set ``Split, Gres`` to ``gpu:H100:1``. 

    .. note::
        
        For this to work, your SLURM instance must have a generic resource (Gres) named ``H100`` defined. To check the available resources in your SLURM instance, run the command: ``sinfo -o "%100N  %30G"``. If you are unsure about this, please contact your system administrator.

Multiple GPUs
^^^^^^^^^^^^^

Some of the programs listed above support multi-GPU execution. To enable this, set the ``Split, Gres`` option to ``gpu:n`` where ``n`` is the number of GPUs you want to request, for example, ``gpu:2``.

.. tab-set::
  :sync-group: running_mode

  .. tab-item:: Desktop workstation
    :sync: standalone

    In this mode, there is typically only type of GPU available, so the instructions above are sufficient.

  .. tab-item:: Compute cluster
    :sync: cluster

    In this mode, users can choose specific resources by setting the ``Split, Gres`` parameter in the **Resources** tab of a block. For example, to use 2 H100 cards, set ``Split, Gres`` to ``gpu:H100:2``.
