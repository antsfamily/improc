Introduction
==================

What's improc
--------------

``improc`` is an image processing toolbox, maintained by `antsfamily <https://github.com/antsfamily>`_, you can download from `here <https://github.com/antsfamily/improc>`_.


- ``blkptcs`` (images <---> blocks, images -->patches) : spliting image into blocks, sampling patches from images, show blocks or patches, selecting patches or blocks(``std`` or ``var``), fight image blocks back to images  and so on.
- ``encoding`` : huffman encoding and decoding
- ``evaluation`` : functions of assessment
- ``utils`` : utility functions --> prep-rocessing(scale, normalization/denormalization)


How to use improc
-------------------

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install it by:

.. code-block:: bash
   :caption: install improc
   :linenos:
   :emphasize-lines: 0,2

   pip install -r requirements.txt
   python setup.py sdist
   sudo python setup.py install --record files.txt


or you can add it's path into ``PYTHONPATH`` environment variable by::

   export PYTHONPATH=/mnt/d/library/zhi/improc:$PYTHONPATH



Uninstallation
~~~~~~~~~~~~~~~~~~~~~

You can use the following commands to uninstall it:

.. code-block:: bash
   :caption: install improc
   :linenos:
   :emphasize-lines: 0,3

   pip uninstall improc
   # or
   cat files.txt | xargs rm -rf

or you can just remove the environment that you have added::

   export PYTHONPATH=/mnt/d/library/zhi/improc:$PYTHONPATH





Usage
~~~~~~~~~~~~~~~~~~~~~


.. code-block:: bash
   :caption: install improc
   :linenos:
   :emphasize-lines:

   import improc as imp
   imp.__version__  # print the version of the tool
   ?imp.imgs2blks # see help information of imgs2blks


You can see the `test` and `examples` folders for more demos.



