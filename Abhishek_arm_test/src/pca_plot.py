from sklearn.svm import SVC
import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

from sklearn.preprocessing import StandardScaler

train_path = "output/flights_pass_1_na_0.csv"
eval_path = "testinput/all_test_with_failures_clean.csv"
#X, Y, X_test, Y_test, dataset = util.load_dataset_new(train_path, eval_path)
x_train_org, y, x_valid_org, y_eval, dataset = util.load_dataset_new(train_path, eval_path)
sc_X = StandardScaler()
X_Train = util.add_intercept(sc_X.fit_transform(x_train_org))
X_Test = util.add_intercept(sc_X.fit_transform(x_valid_org))

##X = iris.data
##y = iris.target
X = X_Train
y = y

pca = PCA(n_components=2)
Xreduced = pca.fit_transform(X)
Xtestreduced = pca.transform(X_Test)

def give_error(y_out, y):
    cnt = 0
    cntfour = 0
    for i in range(len(y_out)):
        if (y_out[i] == y[i]):
            cnt += 1
    return cnt / len(y_out)

def make_meshgrid(x, y, h=.2):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = SVC(kernel='rbf')
clf = model.fit(Xreduced, y)
valid_out = model.predict(Xtestreduced)
train_out = model.predict(Xreduced)
print(give_error(valid_out,y_eval))
print(give_error(train_out,y))
fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of RBF SVM ')
# Set-up grid for plotting.
X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decison surface using the PCA transformed/projected features')
ax.legend()
plt.show()
