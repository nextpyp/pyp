=================
Compute resources
=================

``nextPYP`` can be run in standalone mode or in an HPC environment using the `SLURM <https://slurm.schedmd.com/>`_ workload manager. Standalone mode is simpler to setup and can be used to process small to medium sized datasets. For most datasets, however, an instance with access to a SLURM cluster will enable significantly faster processing.

All operations in ``nextPYP`` are executed on a per-micrograph or per-tilt-series basis. As a consequence, compute resources are specified per micrograph/tilt-series (independent of the total number of micrographs/tilt-series in a dataset).

Threads, tasks, and arrays
--------------------------

There are three types of processes used in ``nextPYP``:

- **Thread**: Single-threaded process running on one CPU core (or hyper-threaded core)
- **Task**: Set of *threads* used to process a single micrograph or tilt-series
- **Array**: Set of *tasks* used to process an entire dataset

Most jobs in ``nextPYP`` have a *Launch*, *Split* and *Merge* phases. The *Launch* phase is typically a lightweigth job used to initiate data processing. In the *Split* phase, multiple tasks are launched and executed in parallel (one task for each micrograph/tilt-series). During the *Merge* phase, information from the *Split* phase is condensed, for example, to produce a single 3D reconstruction from all micrographs or tilt-series in a dataset. Since each phase has different computational requirements, resources are allocated separately for each of them.

Real time information about jobs can be found in the `Jobs panel <../guide/overview.html#jobs-panel>`_.

Each processing block in ``nextPYP`` has a **Resources** tab that allows specifying resources for each job phase:

.. figure:: ../images/tutorial_tomo_pre_process_jobs.webp
  :alt: Job submission options

The **Resources** tab consists of three sections, one for each phase (*Launch*, *Split* and *Merge*):

.. dropdown:: Launch task options
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    .. rst-class:: dl-parameters

    Threads (launch task)
      Number of threads used when launching jobs. Setting this parameter to ``0`` will allocate all cores available in a compute node.

      **Default**: 1

    Memory (launch task)
      Amount of memory requested for launching jobs (GB), Setting this value to ``0`` will tell SLURM to use all available memory in a node.

      **Default**: 0

    Walltime (launch task)
      Exceeding this limit will cause SLURM to terminate the job.

      **Default**: 2:00:00 (hh:mm:ss)

    Partition (launch task)
      Select SLURM partition from the drop down menu.

      **Default**: Set by admin

.. dropdown:: Split task options
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    .. rst-class:: dl-parameters
    
    Threads per task
      Number of threads used to process each micrograph or tilt-series. Setting this parameter to ``0`` will allocate all cores available in a compute node to each task.
    
      **Default**: 1
      
    Max number of tasks
      Maximum number of tasks to run simultaneously. This parameter controls the total number of tasks ``nextPYP`` should run for a particular job. Setting this number to ``0`` will not impose any limits beyond the ones set by SLURM. If a user is running multiple jobs, this setting can be used to manage the resources allocated to each job.

      **Default**: 0
    
    Memory per task
      Amount of memory requested per task (GB), Setting this value to ``0`` will tell SLURM to use all available memory in a node.
    
      **Default**: 0 (uses all memory available)
      
    Walltime per task
      Walltime for each task. Exceeding this time limit will cause SLURM to terminate the jobs.

      **Default**: 2:00:00 (hh:mm:ss)  
      
    Bundle size
      Number of tasks to process as a bundle. Elements of a bundle are processed sequentially.

      **Default**:  1

    CPU partition
      Select SLURM partition from the drop down menu.
         
      **Default**: Set by admin

    GPU partition
      Select SLURM GPU partition from the drop down menu.
         
      **Default**: Set by admin

.. dropdown:: Merge task options
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    .. rst-class:: dl-parameters

    Threads (merge task)
      Number of threads used to execute the merge task. Setting this parameter to ``0`` will allocate all cores available in a compute node.
    
      **Default**: 1

    Memory (merge task)
      Amount of memory requested for the merge task (GB), Setting this value to ``0`` will tell SLURM to use all available memory in a node.

      **Default**: 0

    Walltime (merge task)
      Walltime for each task. Exceeding this limit will cause SLURM to terminate the job.

      **Default**: 2:00:00 (hh:mm:ss)

    Partition (merge task)
      Select SLURM partition from the drop down menu.
         
      **Default**: Set by admin

.. note::
    Users are responsible for ensuring that the combination of resources requested is available in the HPC environment where ``nextPYP`` is running.
    
    
.. tip::
    To get information on the status of a job, go to the **Jobs** panel and click on the :fa:`file-alt text-primary` icon next to the job.