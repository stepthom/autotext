# autotext
Run Auto Sklearn (and other ML algos) on a dataframe with text.

Usage:

First, create features from text (with possibly many variants):

```
python autofeatures.py settings.json
```
See example settings files under `settings`.

The above will output CSV files into the `data` directory.

Second, run algos to build a classification model:

```
python autotext.py settings.json
```

