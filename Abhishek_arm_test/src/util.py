import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def find_correlation(data, threshold=0.8, remove_negative=False):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.
    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value.
    remove_negative: Boolean
        If true then features which are highly negatively correlated will
        also be returned for removal.
    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corr_mat = data.corr()
    if remove_negative:
        corr_mat = np.abs(corr_mat)
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def load_dataset_new(train_path, test_data_path, train_label='highest_failure_level.id'):

    datasets = pd.read_csv(train_path)
    datasets_req = pd.read_csv(train_path)
    datasets_test = pd.read_csv(test_data_path)

    # print(datasets.columns[117])
    # print(datasets.columns[633])
    # print(datasets.columns[115])

    remove_columns = find_correlation(datasets)
    datasets.drop(remove_columns, axis=1, inplace=True)

#    features = [line.rstrip('\n') for line in open('remove_features.txt')]

#    datasets.drop(features, axis=1, inplace=True)
    train_columns = datasets.columns
    test_columns = datasets_test.columns

    to_del_test_columns = np.setdiff1d(test_columns, train_columns)
    datasets_test.drop(to_del_test_columns, axis=1, inplace=True)

    datasets = datasets[datasets_test.columns]

    print("Model trained on " + str(datasets.shape[0]) + " flights with " + str(datasets.shape[1]) + " features")
    print("Ruuning tests on " + str(datasets_test.shape[0]) + " flights")

    #flight_id_index = datasets_test.columns.get_loc("config.flight_id")

    for idx, row in datasets.iterrows():
        if datasets.loc[idx, 'highest_failure_level.id'] == 1:
            datasets.loc[idx, 'highest_failure_level.id'] = 0
        if datasets.loc[idx, 'highest_failure_level.id'] == 2:
            datasets.loc[idx, 'highest_failure_level.id'] = 1
        if datasets.loc[idx, 'highest_failure_level.id'] == 4:
            datasets.loc[idx, 'highest_failure_level.id'] = 2

    x_y_total = datasets
    y_total = x_y_total['highest_failure_level.id']
    del x_y_total['highest_failure_level.id']

    x_total_1 = np.array(x_y_total)
    y_total = np.array(y_total)

    X = x_total_1
    Y = y_total

    # test set
    x_y_total_test = datasets_test
    y_total_test = x_y_total_test['highest_failure_level.id']
    del x_y_total_test['highest_failure_level.id']

    x_total_1_test = np.array(x_y_total_test)
    y_total_test = np.array(y_total_test)

    X_test = x_total_1_test
    Y_test = y_total_test

    return X,Y,X_test,Y_test,datasets

def load_dataset(csv_path, label_col='highest_failure_level.id', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """
    print("loading data.")
    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('highest_failure_level.id')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)
