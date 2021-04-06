.. _InstallationInstructions:

Installation
============

T3f assumes you have Python 3.6 and a working TensorFlow installation (tested with TF 2.4, see here_ for TF installation instructions).

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
