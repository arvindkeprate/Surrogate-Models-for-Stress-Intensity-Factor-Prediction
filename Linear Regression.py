>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.linear_model import LinearRegression
>>> import pandas as pd
>>> names = ["Load", "a", "c", "SIF"]
>>> data = pd.read_csv("Prac.csv", names=names)
>>> array = data.values
>>> X = array[:,0:3]
>>> Y = array[:,3]
>>> X_train = X[0:70,:]
>>> X_validation = X[70:100,:]
>>> Y_train = Y[0:70]
>>> Y_validation = Y[70:100]
>>> from sklearn import linear_model
>>> regr = linear_model.LinearRegression()
>>> regr.fit(X_train,Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
>>> print("Coeff: \n", regr.coef_)
Coeff: 
 [  3.85804223  31.55557969  32.65807434]
>>> print("Intercept: \n", regr.intercept_)
>>> print("MSE: %.2f" %np.mean((regr.predict(X_validation)-Y_validation)**2))
MSE: 1064.49
>>> print("Variance Score: %.2f" %regr.score(X_validation, Y_validation))
Variance Score: 0.97
>>> 




