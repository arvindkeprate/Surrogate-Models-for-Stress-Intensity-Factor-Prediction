import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import pandas as pd
names = ["ICS", "Sigma", "RFL"]
data = pd.read_csv("IndiaConf2018.csv", names=names)
array = data.values
X = array[:,0:2]
Y = array[:,2]
X_train = X[0:100,:]
X_validation = X[100:200,:]
Y_train = Y[0:100]
Y_validation = Y[100:200]
regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(X_train)
print("Coeff: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)
print("MSE: %.2f" %np.mean((regr.predict(X_validation)-Y_validation)**2))
print("Variance Score: %.2f" %regr.score(X_validation, Y_validation))




