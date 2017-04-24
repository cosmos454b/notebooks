# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 07:00:18 2017

@author: SphereProjection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

### Import Dataset
data = pd.read_csv("mpg.csv", na_values=['na']);

##### Label based access
sub1 = data.loc[0:5,['mpg','displacement']]

###### subsetting one column
sub2 = data['mpg']

####### filtering and subsetting
sub3 = data[data['mpg']>15]