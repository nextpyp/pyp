#######################
Single-particle session
#######################

This tutorial shows how to process micrographs on-the-fly during data collection. 

.. admonition::

  * A sample session is available in `the demo instance of nextPYP <https://demo.nextpyp.app/#/session/singleParticle/ezhDW8jzkdLCAzWP>`_.

``nextPYP`` implements streaming using three parallel tracks:

* **File transfer** from the microscope computer to a permanent storage location

* **Data pre-processing** consisting of frame alignment, CTF estimation and particle picking

* **2D classification** to monitor particle quality

Each track is displayed in a separate row and has its own **Status** indicator and **Controls** to :bdg-secondary:`Start`, :bdg-secondary:`Restart`, :bdg-secondary:`Clear`, or :bdg-secondary:`Stop` the session and display the :bdg-primary:`Logs`

Step 1: Create a new session
----------------------------

.. nextpyp:: Streaming is organized in sessions. A new session should be created for each data collection run
  :collapsible: open

  * Go to the **Dashboard** and click on :bdg-primary:`Go to Sessions`

  * Start a Single-particle session by clicking on :bdg-primary:`+ Start Single-particle`

  * Give the session a `Name` and assign a `Group` from the dropdown menu (see :doc:`Administration<../reference/admin>` to create and manage user groups). *Optional*: change the default folder where the data for the session will be saved

  * Go to the **Raw data** tab:

    .. md-tab-set::

      .. md-tab-item:: Raw data

        * Select the ``Location`` of the raw data by clicking on the icon :fa:`search` and navigating to the folder where the movies are saved

        * Click on the **Gain reference** tab

      .. md-tab-item:: Gain reference

        * Specify the location and parameters for the gain reference as needed

        * Click on the **Microscope parameters** tab

      .. md-tab-item:: Microscope parameters

        * Set the ``Pixel size (A)`` and ``Acceleration voltage (kV)``

        * Click on the **Session settings** tab

      .. md-tab-item:: Session settings

        * Select a ``Raw data transfer`` method between "move", "copy", or "link". **Warning**: "move" will copy the raw data to the session folder and delete it from the original location!

        * Click on the **Particle detection** tab

      .. md-tab-item:: Particle detection

        * Set the ``Particle radius (A)``

        * Select "auto" or "all" as the ``Method``

        * Click on the **Resources** tab

      .. md-tab-item:: Resources

        * Set ``Split, Threads`` and other resources as needed (see :doc:`Computing resources<../reference/computing>`)

  * :bdg-primary:`Save` your settings


Step 2: Launch the session
--------------------------

.. nextpyp:: Start data pre-processing
  :collapsible: open

  * Go to the **Operation** tab and click :bdg-primary:`Start` in the **Controls** panel. This will also start the ``Data pre-processing`` and ``2D classification`` daemons

  * You may stop the session at any time using the :bdg-primary:`Cancel` button

  * Monitor storage utilization, data transfer progress, and speed in the **Operation** tab

  * Inspect the streaming results by navigating to the **Plots**, **Table**, **Gallery**, **Micrographs**, and **2D Classes** tabs

Step 3 (optional): Change processing parameters
-----------------------------------------------

.. nextpyp:: Change data processing parameters during a session
  :collapsible: open

  * You can change the data processing settings during a session by going to the **Settings** tab, adjusting parameters as needed, and saving your changes

  * Restart the ``Data pre-processing`` and/or the ``2D classification`` daemon tracks for the changes to take effect

Step 4: Copy or delete a session
--------------------------------

.. nextpyp:: Start a session using settings from an existing session or delete a session
  :collapsible: open

  * You can create a new session by copying the settings of an existing one by clicking on the icon :fa:`copy`

  * You can delete a session by clicking on the icon :fa:`trash`. This will permanently delete the session and all associated files. Running seesions need to be canceled before they can be deleted

Step 5 (optional): Filter and export micrographs
------------------------------------------------

.. nextpyp:: Filter micrographs and export in star format
  :collapsible: open

  * Filter micrographs according to different criteria by going to the **Table** tab. Type a filter name and click :bdg-primary:`Save`. Add and apply filters as needed and click :bdg-primary:`Save` when you are done

  * Click :bdg-primary:`Export` to export the data in ``star`` format. A dialog will appear where you can specify the resource parameters for the export job. After clicking on :bdg-primary:`Export` a new job will appear in the **Operation** tab and you will be able to check its status and see the location of the exported data by clicking on the icon :fa:`eye`.
