print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import util
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

##h = .02  # step size in the mesh
h = .5  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

##classifiers = [
##    KNeighborsClassifier(3),
##    SVC(kernel="linear", C=0.025),
##    SVC(gamma=2, C=1),
##    GaussianProcessClassifier(1.0 * RBF(1.0)),
##    DecisionTreeClassifier(max_depth=5),
##    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
##    MLPClassifier(alpha=1),
##    AdaBoostClassifier(),
##    GaussianNB(),
##    QuadraticDiscriminantAnalysis()]
##classifiers = [
##    KNeighborsClassifier(3),
##    SVC(kernel="linear"),
##    SVC(gamma='auto')]

names = ["Linear SVM", "RBF SVM",
         "Decision Tree",
         "Gaussian Naive Bayes"]
classifiers = [
    SVC(kernel="linear"),
    SVC(gamma='auto'),
    DecisionTreeClassifier(max_depth=5),
    GaussianNB()]
##names = ["RBF SVM"]
##classifiers = [
##    SVC(gamma='auto')]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
train_path = 'output/flights_pass_1_na_0.csv'
eval_path = 'output/flights_new_till_03dec.csv'
x_1, y_1, x_eval,y_eval, data_frame = util.load_dataset_new(train_path,eval_path)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(x_1)
##X = iris.data
##y = iris.target
pca = PCA(n_components=2)

X = pca.fit_transform(X_Train)
y = y_1

linearly_separable = (X, y)

##datasets = [make_moons(noise=0.3, random_state=0),
##            make_circles(noise=0.2, factor=0.5, random_state=1),
##            linearly_separable
##            ]
datasets = [linearly_separable]
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    ##X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.01, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    ##cm = ListedColormap(['#FFAAAA', '#AAFFAA'])
    ##cm_bright = ListedColormap(['#FF0000', '#00FF00'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ##ax = plt.subplot(len(datasets), len(classifiers), i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        ##score = clf.score(X_train, y_train)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ##ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
        ax.contourf(xx, yy, Z, cmap=cm)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
plt.tight_layout()
plt.show()