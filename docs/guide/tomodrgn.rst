=================================
Continuous variability (tomoDRGN)
=================================

``nextPYP`` supports running `tomoDRGN <https://github.com/bpowell122/tomodrgn>`_ on existing tomography projects

Requirements
------------

- An existing :bdg-secondary:`Particle refinement` block with a succesful consensus refinement
- Access to a GPU

tomoDRGN workflow
-----------------

``nextPYP`` follows the general protocol described in the `tomoDRGN documentation <https://bpowell122.github.io/tomodrgn/index.html>_`

Preparation
~~~~~~~~~~~~

#. Since tomoDRGN uses particle projections extracted from the tilt-series, we first need to produce the particles stacks. This is done in the :bdg-secondary:`Particle refinement` block, by going to the **Extraction** tab and selecting the option ``Save particle stacks``
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` 

Learn structural heterogeneity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`tomoDRGN (train-vae)`
#. Select the ``Input file (*.star)`` by navigating to the ``stacks`` directory in the default location and selecting the file ``*_particles.star``
#. Adjust any tomoDRGN parameters as needed, please see `Command Usage <https://bpowell122.github.io/tomodrgn/command_usage/index.html>_` for details
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. This will run tomoDRGN's ``train-vae`` and ``convergence-vae`` commands
#. Check the results by navigating to the :bdg-primary:`tomoDRGN (train-vae)` block

Analyze structural heterogeneity (analyze)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Click on ``DRGN Model`` (output of the :bdg-secondary:`tomoDRGN (train-vae)` block) and select :bdg-primary:`tomoDRGN (analyze)`
#. Adjust any tomoDRGN parameters as needed, please see `Command Usage <https://bpowell122.github.io/tomodrgn/command_usage/index.html>_` for details
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. This will run tomoDRGN's ``analyze`` command
#. Check the results by navigating to the :bdg-primary:`tomoDRGN (analyze)` block

Analyze structural heterogeneity (analyze-volumes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Click on ``DRGN Model`` (output of the :bdg-secondary:`tomoDRGN (train-vae)` block) and select :bdg-primary:`tomoDRGN (analyze-volumes)`
#. Adjust any tomoDRGN parameters as needed, please see `Command Usage <https://bpowell122.github.io/tomodrgn/command_usage/index.html>_` for details
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. This will run tomoDRGN's ``analyze-volumes`` command
#. Check the results by navigating to the :bdg-primary:`tomoDRGN (analyze-volumes)` block

Select particle subsets (filter-star)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Click on ``DRGN Particles`` (output of the :bdg-secondary:`tomoDRGN (analyze)` or :bdg-secondary:`tomoDRGN (analyze-volumes)` blocks) and select :bdg-primary:`tomoDRGN (filter-star)`
#. Select the list of classes you want filter
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. This will run tomoDRGN's ``filter-star`` command

Further refine selected particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Click on ``Particles`` (output of the :bdg-secondary:`tomoDRGN (filter-star)` block) and select :bdg-primary:`Particle refinement`
#. Set the neccesary parameters
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`
#. Check the results by navigating to the :bdg-primary:`Particle refinement` block

.. figure:: ../images/tomodrgn_workflow.webp
    :alt: tomoDRGN workflow
