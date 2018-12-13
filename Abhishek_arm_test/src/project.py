import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import util
from sklearn.preprocessing import StandardScaler


def add_intercept(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x


def load_dataset(csv_path, label_present=False):
    # Validate label_col argument
    allowed_label_cols = 'highest_failure_level.id'
    label_col = 'highest_failure_level.id'
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    print(len(headers))
    #print(headers)
    x_cols = [i for i in range(len(headers)) if headers[i] != label_col]
    print(x_cols)
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype='str')

    if label_present:
        l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
        print(l_cols)
        labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols, dtype='str')

    print(labels)
    if label_present:
        return inputs, labels
    else:
        return inputs

def clean_dataset(file1):
    return


def main(file1):
    print("Running main")
    train_path = "output/flights_pass_1_na_0.csv"
    eval_path = "testinput/all_test_with_failures_clean.csv"
    #X, Y, X_test, Y_test, dataset = util.load_dataset_new(train_path, eval_path)
    x_train_org, y_train, x_valid_org, y_valid, dataset = util.load_dataset_new(train_path, eval_path)

    sc_X = StandardScaler()
    x_train = util.add_intercept(sc_X.fit_transform(x_train_org))
    x_valid = util.add_intercept(sc_X.fit_transform(x_valid_org))

    ###plot correlation matrix
    corr_after_dropping = dataset.corr()
    labels = corr_after_dropping.columns.values
    plt.matshow(corr_after_dropping)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_after_dropping, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(corr_after_dropping.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    #ax.set_xticklabels(labels, size=1)
    ax.set_yticklabels(labels, size=5)
    plot_path = 'output/correlation_plot'
    plt.savefig(plot_path)

    ##Scatter
    #headers = list(dataset.columns.values)
    ##scatter = pd.DataFrame(dataset, columns=headers)
    #scatter = pd.DataFrame(dataset)
    #my_scatter = scatter_matrix(scatter)
    #plt.savefig("output/flightscatter")

    def give_error(y_out, y):
        cnt = 0
        for i in range(len(y_out)):
            if (y_out[i] == y[i]):
                cnt +=1
        return cnt/len(y_out)

    ##Normal Eq
    tau = 0.1
    lwr = LinearReg_normal_eq_locally_weighted(tau)
    lwr.x_train = x_train
    lwr.y_train = y_train
    lwr.x_valid = x_valid
    theta_train = lwr.fit(x_train, y_train, 0.05)

    y_train_out = sigmoid(x_train, theta_train)
    y_valid_out_ne = sigmoid(x_valid,theta_train)

    y_train_out_1 = np.where(y_train_out > 0.65, 1, 0)
    y_valid_out_ne_1 = np.where(y_valid_out_ne > 0.65, 1, 0)

    print(give_error(y_valid_out_ne_1, y_valid))
    print(give_error(y_train_out_1, y_train))
    ##print(y_valid_out_ne_1)
    #print(y_valid_out_ne)
    ##print(y_valid)
    ##LWR
    ###tau_array = np.array([10])
    ###r2_valid_lwr = 0
    ###for i in range(0, len(tau_array)):
    ###    lwr.tau = tau_array[i]
    ###    y_valid_out_lwr = lwr.predict(x_valid)
    ###    y_valid_out_lwr_1 = np.where(y_valid_out_lwr > 0.65, 1, 0)
    ###    print(give_error(y_valid_out_lwr_1, y_valid))

    ##Gradient descent
    linear_reg = LinearRegression_gradient_descent()
    linear_reg.x_train = x_train
    linear_reg.y_train = y_train
    l1_l2_factor = np.array([1,2])
    ##learning_rate = 4.85e-5
    lambda_array = np.array([10, 0.5])
    learning_rate = 1e-5
    cost_limit = 1e-12
    r2_train_gd = 0
    r2_valid_gd = 0
    for i in range(0, len(l1_l2_factor)):
    ##for i in range(0, 0):
        theta_train = linear_reg.fit(x_train,y_train, lambda_array[i], learning_rate, cost_limit, l1_l2_factor[i])

        y_train_out = linear_reg.predict(x_train)
        y_valid_out = linear_reg.predict(x_valid)

        y_train_out_1 = np.where(y_train_out > 0.6, 1, 0)
        y_valid_out_1 = np.where(y_valid_out > 0.6, 1, 0)

        print(give_error(y_valid_out_1, y_valid))
        print(give_error(y_train_out_1, y_train))
        ##print('r2_train_gradient_descent_L',l1_l2_factor[i], 'regularization = ', r2_train_gd, 'for lambda = ', lambda_array[i])
        ##print('r2_valid_gradient_descent_L',l1_l2_factor[i], 'regularization = ', r2_valid_gd, 'for lambda = ', lambda_array[i])


def sigmoid(x, theta):
     z = np.dot(x,theta)
     h_x = (1/(1 + np.exp(-z)))
     return h_x

class LinearRegression_gradient_descent(object):
    eps = 1e-5
    def costfunction_old(self, x, y, theta):
        m,n = x.shape
        h_x = np.dot(x, theta)
        a = h_x-y
        b = a ** 2
        J = np.sum(b)/(2*m)
        return J

    def costfunction(self, x, y, theta):
        h_x = self.sigmoid(x, theta)
        J = (-1 / x.shape[0]) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))
        return J

    def sigmoid(self, x, theta):
         z = np.dot(x,theta)
         h_x = (1/(1 + np.exp(-z)))
         return h_x

    def fit(self, x, y, lambda_1, alpha, cost_limit, l1_l2_factor):
        iteration = 0
        m,n = x.shape
        self.theta = np.zeros(x.shape[1])
        delta = np.Infinity
        J = self.costfunction(x, y, self.theta)
        J_delta = J
        J_save = J
        m = x.shape[0]
        J_arr = []
        while J_delta >= cost_limit:
            iteration += 1
            theta_save = self.theta
            J_save = J
            J_delta_save = J_delta
            theta_new = self.theta
            h_x = np.dot(x, self.theta)
            theta_new[0] = self.theta[0] - ((alpha/m) * np.dot(x[:,0].T, (h_x - y)))
            if (l1_l2_factor == 1):
                theta_new[1:n] = self.theta[1:n]*(1+(alpha*lambda_1/m)) - (alpha/m) * np.dot(x[:,1:n].T, (h_x-y))
            else:
                theta_new[1:n] = self.theta[1:n]*(1+(self.theta[1:n]*alpha*lambda_1/m)) - (alpha/m) * np.dot(x[:,1:n].T, (h_x-y))
            delta = np.linalg.norm(np.subtract(theta_new,self.theta), 1)
            self.theta = theta_new
            J_new = self.costfunction(x, y, self.theta)
            J_arr.append(J_new)
            J_delta = J - J_new
            J = J_new
            if (J_delta < 0):
                J = J_save
                self.theta = theta_save
        #print(J_delta, np.sum(J), iteration)
        #print(J_arr)
        ##plt.plot(J_arr);
        ##plt.xlabel('Iterations');
        ##plt.ylabel('Cost Function');
        ##plt.show()

        return self.theta

    def predict(self, x):
        y = self.sigmoid(x,self.theta)
        return y

class LinearReg_normal_eq_locally_weighted(object):
    def __init__(self, tau, eps=1e-5, theta_0=None, verbose=True):
        self.x = None
        self.y = None
        self.theta = theta_0
        self.eps = eps
        self.verbose = verbose

    def sigmoid(self, x, theta):
         z = np.dot(x,theta)
         h_x = (1/(1 + np.exp(-z)))
         return h_x

    def weight(self, x, x_input, tau):
        z = np.linalg.norm(np.subtract(x_input,x), 2)
        w = np.exp((-1/2)*(np.power(z,2)/np.power(tau,2)))
        return w

    def normal_eq_theta_lwr(self, x, W, y):
        a = np.dot(x.T,W)
        b = np.dot(a,x)
        c = np.dot(a,y)
        theta = np.dot(np.linalg.inv(b),c)
        return theta

    def normal_eq_theta(self, x, y):
        a = np.matmul(x.T,x)
        b = np.matmul(x.T,y)
        theta = np.dot(np.linalg.inv(a),b)
        return theta
    def normal_eq_theta_reg(self, x, y,l):
        a = np.matmul(x.T,x) + (l * np.eye(x.shape[1]))
        b = np.matmul(x.T,y)
        theta = np.dot(np.linalg.inv(a),b)
        return theta

    def fit(self, x, y, l):
        ##self.theta = self.normal_eq_theta(self.x_train, self.y_train)
        self.theta = self.normal_eq_theta_reg(self.x_train, self.y_train, l)
        return(self.theta)

    def predict(self, x):
        m = self.x_train.shape[0]
        m_new = x.shape[0]
        y_predict = np.array([])
        for k in range(0,m_new):
            wi = np.array([])
            for i in range(0,m):
                wi_int = self.weight(self.x_train[i], x[k], self.tau)
                wi = np.concatenate(([wi], [wi_int]), axis=None)
            W = np.diag(wi/2)
            theta_new = self.normal_eq_theta_lwr(self.x_train, W, self.y_train)
            y_new = np.dot(x[k], theta_new)
            y_predict = np.concatenate(([y_predict], [y_new]), axis=None)
        y_out = (1/(1 + np.exp(-y_predict)))
        return y_out
