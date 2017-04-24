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



####Decison Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0);
regressor.fit(X,y);
yPred = regressor.predict(6.5);


####Random Tree
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators=20,random_state=0);
forest_regressor.fit(X,y);
yPred_forest = forest_regressor.predict(6.5);

### Visualization
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,forest_regressor.predict(X_grid),color='blue')
plt.title('Position vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
