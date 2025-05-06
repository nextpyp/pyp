##################
Tomography session
##################

This tutorial shows how to process tilt-series on-the-fly during data collection. 

.. admonition::

  * A sample session is available in `the demo instance of nextPYP <https://demo.nextpyp.app/#/session/tomography/ezlP0oGzfmCPUkqB>`_.

``nextPYP`` implements streaming using two parallel tracks:

* **File transfer** from the microscope computer to a permanent storage location

* **Data pre-processing** consisting of frame alignment, tilt-series alignment, CTF estimation, tomogram reconstruction and particle picking

Each track is displayed in a separate row and has its own **Status** indicator and **Controls** to :bdg-primary:`Start`, :bdg-primary:`Restart`, :bdg-primary:`Clear`, or :bdg-primary:`Stop` the session and display the **Logs**.

Step 1: Create a new session
----------------------------

.. nextpyp:: Streaming is organized in sessions. A new session should be created for each data collection run
  :collapsible: open

  * From the **Dashboard**, click on :bdg-primary:`Go to Sessions`

  * Start a Tomography session using :bdg-primary:`+ Start Tomography`

  * Give the session a `Name` and assign a `Group` from the dropdown menu (see :doc:`Administration<../reference/admin>` to create and assign users to groups). *Optional*: change the default folder where the data for the session will be saved

  * Go the **Raw data** tab:

    .. md-tab-set::

      .. md-tab-item:: Raw data

        * Select the ``Location`` of the raw data by clicking on the icon :fa:`search` and navigating to the folder where the tilt-series are saved

        * Click on the **Gain reference** tab

      .. md-tab-item:: Gain reference

        * Specify the path and parameters for the gain reference

        * Click on the **Microscope parameters** tab

      .. md-tab-item:: Microscope parameters

        * Specify ``Pixel size (A)``, ``Acceleration voltage (kV)``, and the approximate ``Tilt-axis angle (degrees)``

        * Click on the **Session settings** tab

      .. md-tab-item:: Session settings

        * Set ``Number of tilts`` to the number of tilts in each tilt-series. This parameter tells ``nextPYP`` when a tilt-series is complete and ready to be processed

        * Select a ``Raw data transfer`` method between "move", "copy", or "link". **Warning**: "move" will copy the raw data to the session folder and delete it from the original location!

        * Click on the **Frame alignment** tab

      .. md-tab-item:: Frame alignment

        * Select ``Single-file tilt-series`` if acquiring tilt-series as a single file. Otherwise, provide the ``Frame pattern`` to let ``nextPYP`` know what files to look for

        * Click on the **Resources** tab

      .. md-tab-item:: Resources

        * Select the number of ``Threads per task``, ``Memory per task``, and other relevant parameters (see :doc:`Computing resources<../reference/computing>`)

  * Click :bdg-primary:`Save` to save your settings


Step 2: Launch the session
--------------------------

.. nextpyp:: Start data pre-processing
  :collapsible: open

  * Go to the **Operation** tab and :bdg-primary:`Start` the daemon from the **Controls** panel

  * You may stop the daemon at any time using the :bdg-primary:`Cancel` button

  * Monitor storage utilization, data transfer progress, and speed in the **Operation** tab

  * To inspect the streaming results, navigate to the **Plots**, **Table**, **Gallery** and **Tilt Series** tabs

Step 3 (optional): Change processing parameters
-----------------------------------------------

.. nextpyp:: Change data processing parameters during a session
  :collapsible: open

  * You can change the data processing settings during a session by going to the **Settings** tab, adjusting parameters as needed, and saving your changes

  * Restart the ``Data pre-processing`` daemon track for the changes to take effect

Step 4: Copy or delete a session
--------------------------------

.. nextpyp:: Delete or Start a session using settings from an existing session
  :collapsible: open

  * You can delete a session by clicking on the icon :fa:`trash`. This will delete the session and all files produced during pre-processing. Running seesions need to be canceled before they can be deleted

  * You can create a new session with the same settings as an existing session using the icon :fa:`copy`

Step 5 (optional): Filter and export tilt-series
------------------------------------------------

.. nextpyp:: Filter tilt-series and export in star format
  :collapsible: open

  * Filter tilt-series according to different criteria in the **Table** tab. Type a filter name and click :bdg-primary:`Save`. Add and apply filters as needed and click :bdg-primary:`Save` when you are done

  * Click :bdg-primary:`Export` to export the data in star format. A dialog will appear where you can specify the resource parameters to run the export job. After clicking :bdg-primary:`Export`, a new job will appear in the **Operation** tab and you will be able to check its status and see the location of the exported data by clicking on the icon :fa:`eye`.
