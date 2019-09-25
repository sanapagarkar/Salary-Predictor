import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting decision tree regression model to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting 
y_pred = regressor.predict([[6.5]])

#visualizing the decision tree regression results
plt.scatter(x,y,color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Decision Tree Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualizing the decision tree regression results (for higher resolution)
x_grid=np.arange(min(x), max(x), 0.01)
x_grid= x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid), color='blue')
plt.title('Decision Tree Reg')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()