.. _TroubleshootingInstructions:

Troubleshooting
===============

If something does not work, try

* Installing the latest version of the library (see :ref:`InstallationInstructions`)

* Importing TensorFlow in the following way:

.. code-block:: python

  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  tf.enable_resource_variables()
  tf.enable_eager_execution()


* Creating an issue_ on GitHub

.. _issue: https://github.com/Bihaqo/t3f/issues/new
