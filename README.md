
# What is improc?


improc is an image processing tool for :

- ``blkptcs`` (images <---> blocks, images -->patches) : spliting image into blocks, sampling patches from images, show blocks or patches, selecting patches or blocks(``std`` or ``var``), fight image blocks back to images  and so on.
- ``encoding`` : huffman encoding and decoding
- ``evaluation`` : functions of assessment
- ``utils`` : utility functions --> prep-rocessing(scale, normalization/denormalization)

# Installation

you can install it by:

```python
python setup.py sdist
sudo python setup.py install --record files.txt
```

or you can add it's path into ``PYTHONPATH`` environment:

```bash
export PYTHONPATH=/mnt/d/library/zhi/improc:$PYTHONPATH
```


# Uninstallation

use the following commands to uninstall it:

```python
pip uninstall improc
# or
cat files.txt | xargs rm -rf

```

or you can just remove the environment that you added:

```bash
export PYTHONPATH=/mnt/d/library/zhi/improc:$PYTHONPATH
```


# Usage



```python
import improc as imp
imp.__version__  # print the version of the tool
?imp.imgs2blks # see help information of imgs2blks

```

You can see the `test` and `examples` folders for more demos.


# Contact


Email: zhiliu.mind@gmail.com






