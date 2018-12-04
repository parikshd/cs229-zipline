import matplotlib.pyplot as plt
import numpy as np
import util
from sklearn.preprocessing import StandardScaler
import scipy as sc

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train,x_eval,y_eval = util.load_dataset_new(train_path,eval_path)

    # Feature Scaling
#    sc_X = StandardScaler()
#    x_train = sc_X.fit_transform(x_train)
#    X_test_transformed = sc_X.fit_transform(x_eval)

    print("Train shape:" + str(x_train.shape))
    print("Eval shape:" + str(x_eval.shape))

    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    p_eval = clf.predict(x_eval)

    print(p_eval)
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
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        if self.x is None or self.y is None:
            raise RuntimeError('Must call fit before predict.')

        y_hat = []
        for x_i in x:
            w_i = self._get_weights(x_i)
            print("W Shape:" + str(w_i.shape))
            self.theta = self._get_theta(w_i)

            y_hat.append(self.theta.T.dot(x_i))

        y_hat = np.array(y_hat)

        return y_hat

    def _get_weights(self, x):
        """Get LWR weights for an example x."""
        x_diff = x - self.x
        w = np.exp(-np.sum(x_diff ** 2, axis=1) / (2 * self.tau ** 2))
        w = np.diag(w)

        return w

    def normal_eq_theta_lwr(self, x, W, y):
        a = np.dot(x.T,W)
        b = np.dot(a,x)
        c = np.dot(a,y)
        theta = np.dot(np.linalg.inv(b),c)
        return theta

    def _get_theta(self, w):
        """Get theta (linear coefficients) given inputs and weights."""
        #x, y = self.x, self.y
        #A = sc.sparse.csc_matrix(x.T.dot(w).dot(x))
        #theta = sc.sparse.linalg.inv(A).dot(x.T).dot(w).dot(y)
        theta = self.normal_eq_theta_lwr(self.x, w, self.y)
        return theta
        # *** END CODE HERE ***

