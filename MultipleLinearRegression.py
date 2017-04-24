import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns;
### Import Dataset
data = pd.read_csv("50_Startups.csv", na_values=['na']);
X = data[['R&D Spend','Administration','Marketing Spend','State']].values
y = data[['Profit']].values

##Encode categorical variables
labelEncoder_X =  LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
onehotEncoder = OneHotEncoder(categorical_features=[3])
X = onehotEncoder.fit_transform(X).toarray()
#### Avoid dummy variable trap
X = X[:,1:]
### Fill missing data
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
X=imputer.fit_transform(X)

#### Split data into train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)

sns.pairplot(data, hue="species");
### Standardize features scale
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
scaler_Y = StandardScaler()
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test) 

### Simple Linear Regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=False)
regressor.fit(X_train,Y_train)

###Prediction
y_pred = regressor.predict(X_test);
y_pred_inverse = scaler_Y.inverse_transform(y_pred)
y_test_inverse = scaler_Y.inverse_transform(Y_test)

#### Evaluating mode;
from sklearn.metrics import r2_score
r2Score = r2_score(Y_test, y_pred)

## Building the optimal model using Backward Elimation
import statsmodels.api as sm
## append column with 1 at  front X
X = sm.add_constant(X)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
#print(regressor_OLS.summary())

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())