# autotext
Run FLAML, Auto Sklearn, and other AutoML tools on a dataframe with text.

Usage:

First, create features from text (with possibly many variants):

```
python autofeatures.py settings.json
```
See example settings files under `settings`.

The above will output CSV files into the `data` directory.

Second, run AutoML algos to build and assess a classification model:

```
python autotext.py settings.json
```
See example settings files under `settings`.

`autotext.py` expects a single train/val/test dataset triplet, and will run
AutoML tools as appropriate. Results will be output into the `out` directory
using a randomly-generated run name.

