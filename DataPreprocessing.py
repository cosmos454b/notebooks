### Common Steps to preprocess data for machine learning
###  Import Libraries 

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

### Import Dataset
data = pd.read_csv("mpg.csv", na_values=['na']);

#dataLines = []
#with open ("mpg.csv", "r") as dataFile:
#    fileContents = dataFile.read().split("\n");
#    for line in fileContents:
#        dataLines.append(line)
#data = pd.DataFrame(dataLines)

# Subset raw data to get dataframe
df = data.iloc[:,0:]

####Inorder to drop missing values, do this on the main dataframe.
df = df.dropna(axis=0)


###### SECTION 2 : GENERATE FEATURE MATRIX
# Subset feature matrix with dependent variable
X = df[['cylinders','displacement','horsepower','weight','acceleration']]
### Cleanse feature matrix
X=X.apply(pd.to_numeric,errors='coerce')
### Fill missing data
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
X=imputer.fit_transform(X)
y = df[['mpg']].values

###	Encode Categorical Data
#labelEncoder_X =  LabelEncoder()
#X[:,3] = labelEncoder_X.fit_transform(X[:,3])
#onehotEncoder = OneHotEncoder(categorical_features=[3])
#X = onehotEncoder.fit_transform(X).toarray()


#### Split data into train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)

### Standardize features scale. SCale both x and y, expect if y is categorical
Xscaler = StandardScaler()
X_train = Xscaler.fit_transform(X_train)
X_test = Xscaler.transform(X_test)
Yscaler = StandardScaler()
X_train = Yscaler.fit_transform(X_train)
X_test = Yscaler.transform(X_test)