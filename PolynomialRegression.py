import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("Position_Salaries.csv", na_values=['na']);
X = data.iloc[:,1:2].values
y = data[['Salary']].values 


### Simple Linear Regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=False)
regressor.fit(X,y)

### Visualization
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Position vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

### Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X);
poly_regressor = LinearRegression(normalize=False)
poly_regressor.fit(X_poly,y)
### Visualization
plt.scatter(X,y,color='red')
plt.plot(X,poly_regressor.predict(X_poly),color='blue')
plt.title('Position vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()