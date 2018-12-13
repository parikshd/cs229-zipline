import matplotlib.pyplot as plt
import numpy as np
import util
from sklearn.preprocessing import StandardScaler
import scipy as sc
import pandas as pd

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    ##
    ##
    # Load training set
    x_train_org, y_train,x_eval_org,y_eval, data_frame = util.load_dataset_new(train_path,eval_path)

    # Feature Scaling
    sc_X = StandardScaler()
    x_train= util.add_intercept(sc_X.fit_transform(x_train_org))
    x_eval= util.add_intercept(sc_X.fit_transform(x_eval_org))
    #all_zeros = np.where(~x_train.any(axis=0))[0]
    #print(all_zeros)


    print("Train shape:" + str(x_train.shape))
    print("Eval shape:" + str(x_eval.shape))
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train, 0.1)
    y_train_out_real = np.dot(x_train, clf.theta)

    #print(y_train_out)
    p_eval = clf.predict(x_eval)

    def give_error(y_out, y):
        cnt = 0
        for i in range(len(y_out)):
            if (y_out[i] == y[i]):
                cnt +=1
        return cnt/len(y_out)
    #print(give_error(p_eval,y_eval))
    print(p_eval, y_eval)
    ##print(np.int(np.round(np.abs(p_eval))+1), y_eval)
    # for i in range(len(p_eval)):
    #     if (Y_Pred_first_pass[i] == 0):
    #         Y_Pred_first_pass[i] = 1
    #     else:
    #         Y_Pred_first_pass[i] = 2
    #         Y_Pred_second_pass = classifier_second_pass.predict(X_test_transformed[i].reshape(1, -1))
    #         if (Y_Pred_second_pass == 1):
    #             print("got 1 value")
    #             Y_Pred_first_pass[i] = 4

    #print('Validation MSE: {:g}'.format(np.mean((p_eval - y_eval) ** 2)))


def plot(x_eval, p_eval, x_train, y_train, save_path):
    plt.figure(figsize=(12, 8))

    # Plot data
    plt.scatter(x_train, y_train, marker='x', c='blue', alpha=.5)
    sorted_idx = np.argsort(x_eval, axis=None)
    plt.scatter(x_eval[sorted_idx], p_eval[sorted_idx],
                marker='o', c='red', alpha=.5)

    plt.savefig(save_path)
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x_train = None
        self.y_train = None

    def fit(self, x, y, l):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x_train = x
        self.y_train = y
        self.theta = self.normal_eq_theta_reg(self.x_train, self.y_train, 1)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        if self.x_train is None or self.y_train is None:
            raise RuntimeError('Must call fit before predict.')
        m = self.x_train.shape[0]
        m_new = x.shape[0]
        y_predict = np.array([])
        for k in range(0,m_new):
            wi = np.array([])
            for i in range(0,m):
                wi_int = self.weight(self.x_train[i], x[k], self.tau)
                wi = np.concatenate(([wi], [wi_int]), axis=None)
            W = np.diag(wi/2)
            #print(W)
            theta_new = self.normal_eq_theta_lwr(self.x_train, W, self.y_train)
            ##theta_new = self.normal_eq_theta_reg(self.x_train, self.y_train, 0.05)
            y_new = np.dot(theta_new, x[k])
            y_predict = np.concatenate(([y_predict], [y_new]), axis=None)
        y_out = y_predict

        return y_out

    def weight(self, x, x_input, tau):
        z = np.linalg.norm(np.subtract(x_input,x), 2)
        w = np.exp((-1/2)*(np.power(z,2)/np.power(tau,2)))
        return w

    def normal_eq_theta_lwr(self, x, W, y):
        a = np.dot(x.T, W)
        b = np.dot(a, x)
        c = np.dot(a, y)
        #a = np.matmul(x.T, W)
        #b = np.matmul(a, x)
        #c = np.matmul(a, y)
        #print(np.linalg.det(b))
        #exit(1)
        theta = np.dot(np.linalg.inv(b), c)
        return theta

    def normal_eq_theta_reg(self, x, y,l):
        a = np.dot(x.T,x) + (l * np.eye(x.shape[1]))
        b = np.dot(x.T,y)
        theta = np.dot(np.linalg.inv(a),b)
        return theta

        # *** END CODE HERE ***

