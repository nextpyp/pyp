
=============================
Single-particle import/export
=============================

Requirements
------------

- Star file from 3D classification or refinement jobs (``*_data.star``)
- Star file with motion correction results (``MotionCor/{job_id}/corrected_micrographs.star``)
- Absolute path to the corresponding Relion project folder

Procedure
---------

A single command is needed to convert the project to the ``pyp`` format:

.. code-block:: bash

    rlp -import_read_star -import_refine_star "path_to_refinement_star_file" -import_relion_path "RELION_project_path" -import_motion_star "path_to_corrected_micrographs.star" -scope_pixel pixel_size -scope_voltage voltage -data_path path_to_raw_movies -data_mode "spr" -import_mode "SPA_STAR"

After running this command, everything necessary to continue refinement in ``pyp`` will be created.

==========================================
Export single-particle data in star format
==========================================

Export data from ``pyp``
------------------------

- Go to the ``pyp`` project directory and run:

.. code-block:: bash

    csp -export_enable                              \
        -export_parfile "path_to_parfile"           \
        -export_location "path_to_export_location"

A star file will be saved in the export_location folder with a basename corresponding to the dataset name.

- Import the project into Relion

  1. Start a new Relion project:

  .. code-block:: bash

      cd relion/

      # create links tp pyp frame averages into Micrographs folder
      mkdir Micrographs
      cd Micrographs
      ln -s {pyp_project_path}/ali/*.mrc .  # link the *_DW.mrc files if you want to use dose-weighted averages
      cd ..

      # start Relion
      relion &


  2. If not doing movie correction in Relion, import micrographs using the star file converted from ``PYP``

  3. Re-extract particles in Relion (either re-center based on the particles shifts or not)

  4. Do refinement or classification with the re-extracted particles

.. tip::

    If you want to use Relion's motion correction results, you will need to import the raw movies instead of the frame averages. The ``_rlnMicrographName`` column in the star file converted from ``pyp`` is in the following format: `Micrographs/{image_name}.mrc`

    Bfore importing the raw movies into Relion, we need to edit the ``_rlnMicrographName`` column as follows:

    .. code-block:: bash

        sed -i 's/Micrographs/MotionCor\/{job_id}\/{where_the_aligned_images_saved}/g' {dataset_name}.star

.. note::
    If there are ``.`` characters in the micrograph names, Relion will substitute them by ``_``. In this case, you can use a similar ``sed``  command to edit the star file before importing the data.
