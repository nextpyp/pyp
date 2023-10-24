#######################
Single-particle session
#######################

This tutorial shows how to process micrographs on-the-fly during data collection. ``nextPYP`` implements streaming using three parallel tracks:

* **File transfer** from the microscope computer to a permanent storage location

* **Data pre-processing** consisting of frame alignment, CTF estimation and particle picking

* **2D classification** to monitor particle quality

Each track is displayed in a separate row and has its own **Status** indicator and **Controls** to :badge:`Start,badge-primary`, :badge:`Restart,badge-primary`, :badge:`Clear,badge-primary`, or :badge:`Stop,badge-primary` the session and to display the :badge:`Logs,badge-primary`

Step 1: Create a new session
----------------------------

.. dropdown:: Streaming is organized in sessions. A new session should be created for each data collection run
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Go to the **Dashboard** and click on :badge:`Go to Sessions,badge-primary`

    * Start a Single-particle session by clicking on :badge:`+ Start Single-particle,badge-primary`

    * Give the session a `Name` and assign a `Group` from the dropdown menu (see :doc:`Administration<../reference/admin>` to create and manage user groups)

    * Go to the **Image data** tab:

      .. tabbed:: Image data

        * Select the ``Location`` of the raw data by clicking on the icon :fa:`search,text-primary` and navigating to the folder where the movies are saved

        * Click on the **Gain reference** tab

      .. tabbed:: Gain reference

        * Specify the location and parameters for the gain reference as needed

        * Click on the **Microscope parameters** tab

      .. tabbed:: Microscope parameters

        * Set the ``Pixel size (A)`` and ``Acceleration voltage (kV)``

        * Click on the **Particle detection** tab

      .. tabbed:: Particle detection

        * Set the ``Particle radius (A)``

        * Click on the **Resources** tab

      .. tabbed:: Resources

        * Set the number of ``Threads per task``, ``Memory per task``, and other resources as needed (see :doc:`Computing resources<../reference/computing>`)

    * :badge:`Save,badge-primary` your settings


Step 2: Launch the streaming session
------------------------------------

.. dropdown:: Start data pre-processing
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Go to the **Operation** tab and :badge:`Start,badge-primary` the daemon from the **Controls** panel

    * You may stop the session at any time using the :badge:`Cancel,badge-primary` button

    * Inspect the results by navigating to the **Plots**, **Table**, **Gallery**, **Micrographs**, and **2D Classes** tabs

Step 3 (optional): Adjust data processing parameters
----------------------------------------------------

.. dropdown:: Change data processing parameters during a session
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * You can change the data processing settings during a session by going to the **Settings** tab and saving your changes

    * Restart the corresponding daemon tracks for the changes to take effect

Step 4: Copy or delete a session
--------------------------------

.. dropdown:: Start a session using settings from an existing session or delete a session
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * You can create a new session by copying the settings of an existing one by clicking on the icon :fa:`copy, text-primary`

    * You can delete a session by clicking on the icon :fa:`trash, text-primary`. This will permanently delete the session and all associated files

Step 5 (optional): Filter micrographs and export metadata
---------------------------------------------------------

.. dropdown:: Filter micrographs and export to external programs in star format
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * You can filter micrographs according to different criteria by going to the **Table** tab. Type a filter name and click :badge:`Save,badge-primary`. Add and apply filters as needed and click :badge:`Save,badge-primary` when you are done

    * Click :badge:`Export,badge-primary` to export the data in ``star`` format. A dialog will appear where you can specify the resources for the export job. After clicking on :badge:`Export,badge-primary` a new job will appear in the **Operation** tab and you will be able to check its status and see the location of the exported data by clicking on the icon :fa:`eye, text-primary`.

.. seealso::

    * :doc:`Tomography session<stream_tomo>`
    * :doc:`Single-particle tutorial<spa_empiar_10025>`
    * :doc:`Tomography tutorial<tomo_empiar_10164>`
    * :doc:`Classification tutorial<tomo_empiar_10304>`