# Kernel SVM
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the datasets

def give_error(y_out, y):
    cnt = 0
    cntfour = 0
    for i in range(len(y_out)):
        if (y_out[i] == y[i]):
            cnt += 1
    return cnt / len(y_out)

datasets = pd.read_csv('output/flights_pass_1_na_0.csv')
print(datasets.shape)

datasets['original_failure_level'] = datasets['highest_failure_level.id']

for idx, row in datasets.iterrows():
    if  datasets.loc[idx,'highest_failure_level.id'] == 1:
        datasets.loc[idx,'highest_failure_level.id'] = 0
    if  datasets.loc[idx,'highest_failure_level.id'] == 2:
        datasets.loc[idx,'highest_failure_level.id'] = 1
    if  datasets.loc[idx,'highest_failure_level.id'] == 4:
        datasets.loc[idx,'highest_failure_level.id'] = 1

df_model_second_pass = pd.read_csv('output/flights_pass_1_na_0.csv', low_memory=False)
df_all_copy_fail_crash = df_model_second_pass[df_model_second_pass['highest_failure_level.id'] != 1]

for idx, row in df_all_copy_fail_crash.iterrows():
    if  df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] == 2:
        df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] = 0
    if  df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] == 4:
        df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] = 1

x_y_total = datasets
y_total = x_y_total['highest_failure_level.id']
del x_y_total['highest_failure_level.id']

x_total_1 = np.array(x_y_total)
y_total = np.array(y_total)

X = x_total_1
Y = y_total


x_y_total_second_pass = df_all_copy_fail_crash
y_total_second_pass = x_y_total_second_pass['highest_failure_level.id']
del x_y_total_second_pass['highest_failure_level.id']

x_total_1_second_pass = np.array(x_y_total_second_pass)
y_total_second_pass = np.array(y_total_second_pass)

X_second_pass = x_total_1_second_pass
Y_second_pass = y_total_second_pass

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

lenthXTrain = X_Train.shape[1]
lenthXTest = X_Test.shape[1]

Y_Train_saved = X_Train[:,lenthXTrain-1]
Y_Test_saved = X_Test[:,lenthXTest-1]

print(X_Train.shape)
print(X_Test.shape)

print(lenthXTrain)
print(lenthXTest)

X_Train = np.delete(X_Train,[lenthXTrain-1],axis=1)
X_Test = np.delete(X_Test,[lenthXTest-1],axis=1)

print(X_Train.shape)
print(X_Test.shape)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

sc_X_second_pass = StandardScaler()
X_second_pass = sc_X_second_pass.fit_transform(X_second_pass)

# Fitting the classifier into the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier_second_pass = SVC(kernel = 'rbf', random_state = 0)


# Predicting the test set results
classifier.fit(X_Train,Y_Train)
classifier_second_pass.fit(X_second_pass,Y_second_pass)

# Y_pred_train = classifier.predict(X_Train)
# print(give_error(Y_pred_train,Y_Train))

Y_Pred_first_pass = classifier.predict(X_Test)
for i in range(len(Y_Pred_first_pass)):
    if (Y_Pred_first_pass[i] == 0):
        Y_Pred_first_pass[i] = 1
    else:
        Y_Pred_first_pass[i] = 2
        Y_Pred_second_pass = classifier_second_pass.predict(X_Test[i].reshape(1, -1))
        if (Y_Pred_second_pass == 1):
            print("got 1 value")
            Y_Pred_first_pass[i] = 4

print(give_error(Y_Pred_first_pass,Y_Test_saved))

Y_Pred_first_pass_train = classifier.predict(X_Train)
for i in range(len(Y_Pred_first_pass_train)):
    if (Y_Pred_first_pass_train[i] == 0):
        Y_Pred_first_pass_train[i] = 1
    else:
        Y_Pred_first_pass_train[i] = 2
        Y_Pred_second_pass_train = classifier_second_pass.predict(X_Train[i].reshape(1, -1))
        if (Y_Pred_second_pass_train == 1):
            print("got 1 value train")
            Y_Pred_first_pass_train[i] = 4

print(give_error(Y_Pred_first_pass_train,Y_Train_saved))
# Making the Confusion Matrix
