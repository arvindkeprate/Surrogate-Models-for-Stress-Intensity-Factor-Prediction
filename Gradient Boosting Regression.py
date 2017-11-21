import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, loss="huber")
y_gbr = clf.fit(X_train, Y_train).predict(X_validation)
print(y_gbr)
df_gbr = pd.DataFrame(y_gbr, columns = ["RFL"])
writer = pd.ExcelWriter("GBRIndia.xlsx", engine = "xlsxwriter")
df_gbr.to_excel(writer, sheet_name="Sheet1")
writer.save()

# partial plots and other plots commands
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
features = [0, 1, 2]
target_feature = (0, 1)
pdp, axes = partial_dependence(clf, target_feature,X=X_train, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
ax.set_xlabel(names[target_feature[0]])
<matplotlib.text.Text object at 0x043BD430>
ax.set_ylabel(names[target_feature[1]])
<matplotlib.text.Text object at 0x043CE3B0>
ax.set_zlabel('Partial dependence')
<matplotlib.text.Text object at 0x044D2270>
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
<matplotlib.colorbar.Colorbar object at 0x043CEF30>
plt.suptitle('Partial dependence of SIF on Load & a')
<matplotlib.text.Text object at 0x09F105F0>
plt.subplots_adjust(top=0.8)
plt.show()
