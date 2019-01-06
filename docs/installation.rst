.. _InstallationInstructions:

Installation
============

T3f assumes you have Python 2.7, 3.4, 3.5, or 3.6 and a working TensorFlow installation (tested versions are from 1.10 to 1.12, see here_ for TF installation instructions).

.. _here: https://www.tensorflow.org/install/

We don't include TF into pip requirements since the installation of TensorFlow varies depending on your setup.

Then, to install the stable version, run

.. code-block:: bash

   pip install t3f

To install the latest version, run

.. code-block:: bash

   git clone https://github.com/Bihaqo/t3f.git
   cd t3f
   pip install .

Note that using the neural module (`t3f.nn`) may require additional dependencies
such as Keras.
