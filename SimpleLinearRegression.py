import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
### Import Dataset
data = pd.read_csv("Salary_Data.csv", na_values=['na']);
df=data.apply(pd.to_numeric,errors='coerce')
X = df[['YearsExperience']].values
y = df[['Salary']].values


### Fill missing data
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
X=imputer.fit_transform(X)


#### Split data into train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)

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

### Visualization
#plt.scatter(X_train,Y_train,color='red')
#plt.plot(X_train,regressor.predict(X_train),color='blue')
#plt.title('Salary vs Experience')
#plt.xlabel('Experience')
#plt.ylabel('Salary')
#plt.show()

import seaborn as sns;
g = sns.JointGrid(x=X_train, y=Y_train)
g = g.plot_joint(plt.scatter, color="g",edgecolor="white")
g = plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.annotate("R sq:"+str(r2Score), xy=(0, 2))



denomarlized_y_coeff = (regressor.coef_/scaler_X.scale_)*scaler_Y.scale_;
denomarlized_intercept = (regressor.intercept_/scaler_X.scale_)*scaler_Y.scale_;
print('weights: ')
print(denomarlized_y_coeff)
print('Intercept: ')
print(denomarlized_intercept)


import statsmodels.api as sm
## append column with 1 at  front X
X_train = sm.add_constant(X_train)
regressor_OLS = sm.OLS(endog=Y_train,exog=X_train).fit()
print(regressor_OLS.summary())