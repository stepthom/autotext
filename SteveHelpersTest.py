import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


from SteveHelpers import dump_json, get_data_types
from SteveHelpers import SteveCorexWrapper
from SteveHelpers import SteveCategoryCoalescer
from SteveHelpers import SteveCategoryImputer
from SteveHelpers import SteveNumericNormalizer
from SteveHelpers import SteveDateExtractor
from SteveHelpers import SteveLatLongDist
from SteveHelpers import SteveMissingIndicator
from SteveHelpers import SteveAutoFeatLight

df = pd.DataFrame(data={
    'f1': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'e'],
    'f2': [1.,   2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
    'f3': ['a', 'a', 'a', np.nan, 'b', 'b', 'b', 'c', 'c', np.nan, 'e'],
    'f4': [100.,   293.,  322.,  4.,  5.,  6.,  7.,  8.,  9., -10., 11.],
    'f5': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    'f6': [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    'f7': ['2018-01-01 00:00:00', '2018-02-01 00:00:00', '2018-03-01 00:00:00', '2018-04-01 00:00:00',
          '2019-01-01 00:00:00', '2019-01-02 00:00:00', '2019-01-10 00:00:00', '2019-01-23 00:00:00',
          '2029-12-12 12:13:14', '2018-01-01 00:00:00', '2018-01-01 00:00:00'],
    'f8': ['Vancouver', 'Toronto', 'Edmonton', 'York', 'Quebec City', 'Winnipeg', 'Victoria', 'Toronto City Hall', 'Halifax', 'British Columbia', 'St Johns' ],
    'f9': [49.246292, 43.651070, 53.631611, 43.76153, 46.829853, 49.895077, 48.407326, 43.653908, 44.651070, 53.726669, 47.560539],
    'f10': [-123.116226, -79.347015, -113.323975, -79.411079, -71.254028,  -97.138451, -123.329773,  -79.384293, -63.582687, -127.647621, -52.712830],
    'f11': [1.,   np.nan,  3.,  4.,  5.,  6.,  7.,  np.nan,  9., 10., 11.],
})

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df['f7'] = pd.to_datetime(df['f7'])

print(df)



cat_cols, num_cols, bin_cols, float_cols, date_cols = get_data_types(df, '', '')


from sklearn.pipeline import Pipeline

print("Test CategoryCoalescer")
prep = SteveCategoryCoalescer(keep_top=25, cat_cols=['f1'], )
prep.fit(df)
print(prep.transform(df))

print("Test CategoryCoalescer")
prep = SteveCategoryCoalescer(keep_top=3, cat_cols=['f1'], )
prep.fit(df)
print(prep.transform(df))

print("Test CategoryCoalescer")
prep = SteveCategoryCoalescer(keep_top=1, cat_cols=['f1'], )
prep.fit(df)
print(prep.transform(df))

print("Test CategoryCoalescer")
prep = SteveCategoryImputer(cat_cols=['f1', 'f3'], )
prep.fit(df)
print(prep.transform(df))

print("Test NumericNormalizer")
prep = SteveNumericNormalizer(float_cols=['f2', 'f4'], )
prep.fit(df)
print(prep.transform(df))

print("Test CorexWrapper")
prep = SteveCorexWrapper(bin_cols=['f5', 'f6'], )
prep.fit(df)
print(prep.transform(df))

print("Test DateExtractor")
prep = SteveDateExtractor(date_cols=['f7'], )
prep.fit(df)
print(prep.transform(df))

print("Test LatLongDist")
prep = SteveLatLongDist(lat_col='f9', long_col='f10', point_list=[('Toronto', (43.651070, -79.347015)), ('Vancouver', (49.246292, -123.116226 ))])
prep.fit(df)
print(prep.transform(df))

print("Test MissingIndicator")
prep = SteveMissingIndicator(num_cols=['f2', 'f11'])
prep.fit(df)
print(prep.transform(df))

print("Test AutoFeatLight")
prep = SteveAutoFeatLight(num_cols=['f2', 'f4', 'f9'], compute_ratio=True, compute_product=True, scale=False)
prep.fit(df)
print(prep.transform(df))
