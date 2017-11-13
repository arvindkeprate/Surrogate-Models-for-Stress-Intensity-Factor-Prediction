Python 3.4.4 (v3.4.4:737efcadf5a6, Dec 20 2015, 19:28:18) [MSC v.1600 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import pandas as pd
>>> names = ["Load", "a", "c", "SIF"]
>>> data = pd.read_csv("Prac.csv", names=names)
>>> array = data.values
>>> X = array[:,0:3]
>>> Y = array[:,3]
>>> X_train = X[0:70,:]
>>> Y_train = Y[0:70]
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, loss="huber")
>>> clf.fit(X_train, Y_train)
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='huber', max_depth=4,
             max_features=None, max_leaf_nodes=None,
             min_impurity_split=1e-07, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
>>> Y_test = Y[70:100]
>>> X_test = X[70:100]
>>> fig = plt.figure()
>>> from sklearn.ensemble.partial_dependence import partial_dependence
>>> from sklearn.ensemble.partial_dependence import plot_partial_dependence
>>> features = [0, 1, 2]
>>> target_feature = (0, 1)
>>> pdp, axes = partial_dependence(clf, target_feature,X=X_train, grid_resolution=50)
>>> XX, YY = np.meshgrid(axes[0], axes[1])
>>> Z = pdp[0].reshape(list(map(np.size, axes))).T
>>> from mpl_toolkits.mplot3d import Axes3D
>>> ax = Axes3D(fig)
>>> surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
>>> ax.set_xlabel(names[target_feature[0]])
<matplotlib.text.Text object at 0x043BD430>
>>> ax.set_ylabel(names[target_feature[1]])
<matplotlib.text.Text object at 0x043CE3B0>
>>> ax.set_zlabel('Partial dependence')
<matplotlib.text.Text object at 0x044D2270>
>>> ax.view_init(elev=22, azim=122)
>>> plt.colorbar(surf)
<matplotlib.colorbar.Colorbar object at 0x043CEF30>
>>> plt.suptitle('Partial dependence of SIF on Load & a')
<matplotlib.text.Text object at 0x09F105F0>
>>> plt.subplots_adjust(top=0.8)
>>> plt.show()
