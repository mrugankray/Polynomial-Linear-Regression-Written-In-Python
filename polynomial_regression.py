import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing libraries
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#fit linear regressor in dataset
from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg.fit(X,Y)

#fitting polynomial regressor into dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
line_reg_2 = LinearRegression()
line_reg_2.fit(X_poly,Y)

#visualising the linear regression 
plt.scatter(X,Y,color = 'red')
plt.plot(X,line_reg.predict(X))
plt.title('Truth or Bluff(linear regression)')
plt.xlabel('Experience lavel')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid,line_reg_2.predict(poly_reg.fit_transform(X_grid)))
plt.title('Truth or Bluff(polynomial regression)')
plt.xlabel('Experience lavel')
plt.ylabel('Salary')
plt.show()

#predicting a new result using linear regression
line_reg.predict(6.5)

#predicting a new result using polynomial  regression
line_reg_2.predict(poly_reg.fit_transform(6.5))