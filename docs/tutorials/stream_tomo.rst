#####################
Tomography on-the-fly
#####################

This tutorial shows how to process tilt-series on-the-fly during data collection. ``nextPYP`` implements streaming using two parallel tracks:

* **File transfer** from the microscope computer to a permanent storage location

* **Data pre-processing** consisting of frame alignment, tilt-series alignment, CTF estimation, tomogram reconstruction and particle picking

Each track is displayed in a separate row and has its own **Status** indicator and **Controls** to :badge:`Start,badge-primary`, :badge:`Restart,badge-primary`, :badge:`Clear,badge-primary`, or :badge:`Stop,badge-primary` the session and to display the **Logs**.

Step 1: Create a new session
----------------------------

.. dropdown:: Streaming is organized in sessions. A new session should be created for each data collection run
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * From the **Dashboard**, click on :badge:`Go to Sessions,badge-primary`

    * Start a Tomography session using :badge:`+ Start Tomography,badge-primary`

    * Give the session a `Name` and assign a `Group` from the dropdown menu (see :doc:`Administration<../reference/admin>` to create and assign users to groups)

    * Go the **Raw data** tab:

      .. tabbed:: Raw data

        * Select the ``Location`` of the raw data by clicking on the icon :fa:`search,text-primary` and navigating to the folder where the tilts are saved

        * Click on the **Gain reference** tab

      .. tabbed:: Gain reference

        * Specify the path and parameters for the gain reference

        * Click on the **Microscope parameters** tab

      .. tabbed:: Microscope parameters

        * Specify ``Pixel size (A)``, ``Acceleration voltage (kV)``, and the approximate ``Tilt-axis angle (degrees)``

        * Click on the **Session settings** tab

      .. tabbed:: Session settings

        * Set ``Number of tilts`` to the number of tilts in each tilt-series. This parameter tells ``nextPYP`` when a tilt-series is complete and ready to be processed

        * Click on the **Frame alignment** tab

      .. tabbed:: Frame alignment

        * Select ``Single-file tilt-series`` if acquiring tilt-series as a single file. Otherwise, provide the ``Frame pattern`` to let ``nextPYP`` know what files to look for

        * Click on the **Resources** tab

      .. tabbed:: Resources

        * Select the number of ``Threads per task``, ``Memory per task``, and other relevant parameters (see :doc:`Computing resources<../reference/computing>`)

    * Click :badge:`Save,badge-primary` to save your settings


Step 2: Launch the streaming session
------------------------------------

.. dropdown:: Start data pre-processing
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Go to the **Operation** tab and :badge:`Start,badge-primary` the daemon from the **Controls** panel

    * You may stop the daemon at any time using the :badge:`Cancel,badge-primary` button

    * To inspect the streaming results, navigate to the **Plots**, **Table**, **Gallery** and **Tilt Series** tabs

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

.. dropdown:: Delete or Start a session using settings from an existing session
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * You can delete a session by clicking on the icon :fa:`trash, text-primary`. This will delete the session and all files produced during pre-processing

    * You can create a new session with the same settings as an existing session using the icon :fa:`copy, text-primary`

Step 5 (optional): Filter tilt-series and export metadata
---------------------------------------------------------

.. dropdown:: Filter tilt-series and export to external programs in star format
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * You can filter tilt-series according to different criteria in the **Table** tab. Type a filter name and click :badge:`Save,badge-primary`. Add and apply filters as needed and click :badge:`Save,badge-primary` when you are done

    * Click :badge:`Export,badge-primary` to export the data in star format. A dialog will appear where you can specify the resources to run the export job. After clicking :badge:`Export,badge-primary`, a new job will appear in the **Operation** tab and you will be able to check its status and see the location of the exported data by clicking on the icon :fa:`eye, text-primary`.

.. seealso::

    * :doc:`Single-particle session<stream_spr>`
    * :doc:`Tomography tutorial<tomo_empiar_10164>`
    * :doc:`Classification tutorial<tomo_empiar_10304>`
    * :doc:`Single-particle tutorial<spa_empiar_10025>`