import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.scatter(x,y,color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Linear Reg')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

x_grid=np.arange(min(x), max(x), 0.1)
x_grid= x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Linear Reg')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict(np.array([6.5]).reshape(1,1))

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))