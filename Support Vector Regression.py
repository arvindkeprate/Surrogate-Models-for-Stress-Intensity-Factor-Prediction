import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
names = ["ICS", "Sigma", "RFL"]
data = pd.read_csv("IndiaConf2018.csv", names=names)
array = data.values
X = array[:,0:2]
Y = array[:,2]
X_train = X[0:100,:]
Y_train = Y[0:100]
X_validation = X[100:200,:]
Y_validation = Y[100:200]
svr_rbf =SVR(kernel="rbf", C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_validation)
#print(y_rbf)
df_rbf = pd.DataFrame(y_rbf, columns = ["RFL"])
#print(df_rbf)
writer = pd.ExcelWriter("SVRIndia.xlsx", engine = "xlsxwriter")
df_rbf.to_excel(writer, sheet_name="Sheet1")
writer.save()
#print("MSE: %.2f" %np.mean((y_rbf-Y_validation)**2))
