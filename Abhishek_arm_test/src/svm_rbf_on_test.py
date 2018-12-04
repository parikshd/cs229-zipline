# Kernel SVM
# Importing the libraries

import numpy as np
import pandas as pd
import sklearn

def give_error(y_out, y, x):
    cnt = 0
    cntfalse = 0
    for i in range(len(y_out)):
        if (y_out[i] == y[i]):
            cnt += 1
        else:
            #print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
            cntfalse += 1
            if (y_out[i] == 2):
                print("Flight " + str(int(x[i][flight_id_index])) + " might need maintaince, our algorithm predicted it would have mission failure!")
            if (y_out[i] == 4):
                print("Flight " + str(int(x[i][flight_id_index])) + " definitely needs maintaince, our algorithm predicted it would have flight failure!")
    print("Predicted " + str(cnt) + "/" + str(len(y_out)) + " correctly.")
    print("Predicted " + str(cntfalse) + "/" + str(len(y_out)) + " incorrectly.")
    return cnt / len(y_out)

datasets = pd.read_csv('output/flights_pass_1_na_0.csv')
datasets_test = pd.read_csv('testinput/flights_new_till_03dec.csv')

test_columns = datasets_test.columns
train_columns = datasets.columns

to_del_test_columns = np.setdiff1d(test_columns,train_columns)
datasets_test.drop(to_del_test_columns, axis=1, inplace=True)

datasets = datasets[datasets_test.columns]

print("Model trained on " + str(datasets.shape[0]) + " flights with " + str(datasets.shape[1]) + " features")
print("Ruuning tests on " + str(datasets_test.shape[0]) + " flights")

flight_id_index = datasets_test.columns.get_loc("config.flight_id")

for idx, row in datasets.iterrows():
    if  datasets.loc[idx,'highest_failure_level.id'] == 1:
        datasets.loc[idx,'highest_failure_level.id'] = 0
    if  datasets.loc[idx,'highest_failure_level.id'] == 2:
        datasets.loc[idx,'highest_failure_level.id'] = 1
    if  datasets.loc[idx,'highest_failure_level.id'] == 4:
        datasets.loc[idx,'highest_failure_level.id'] = 1

df_model_second_pass = pd.read_csv('output/flights_pass_1_na_0.csv', low_memory=False)
df_all_copy_fail_crash = df_model_second_pass[df_model_second_pass['highest_failure_level.id'] != 1]

df_all_copy_fail_crash = df_all_copy_fail_crash[datasets_test.columns]

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

#test set
x_y_total_test = datasets_test
y_total_test = x_y_total_test['highest_failure_level.id']
del x_y_total_test['highest_failure_level.id']

x_total_1_test = np.array(x_y_total_test)
y_total_test = np.array(y_total_test)

X_test = x_total_1_test
Y_test = y_total_test

x_y_total_second_pass = df_all_copy_fail_crash
y_total_second_pass = x_y_total_second_pass['highest_failure_level.id']
del x_y_total_second_pass['highest_failure_level.id']

x_total_1_second_pass = np.array(x_y_total_second_pass)
y_total_second_pass = np.array(y_total_second_pass)

X_second_pass = x_total_1_second_pass
Y_second_pass = y_total_second_pass

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test_transformed = sc_X.fit_transform(X_test)

sc_X_second_pass = StandardScaler()
X_second_pass = sc_X_second_pass.fit_transform(X_second_pass)

# Fitting the classifier into the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto')
classifier_second_pass = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto')

# Predicting the test set results
classifier.fit(X,Y)
classifier_second_pass.fit(X_second_pass,Y_second_pass)

# Y_pred_train = classifier.predict(X_Train)
# print(give_error(Y_pred_train,Y_Train))

Y_Pred_first_pass = classifier.predict(X_test_transformed)
for i in range(len(Y_Pred_first_pass)):
    if (Y_Pred_first_pass[i] == 0):
        Y_Pred_first_pass[i] = 1
    else:
        Y_Pred_first_pass[i] = 2
        Y_Pred_second_pass = classifier_second_pass.predict(X_test_transformed[i].reshape(1, -1))
        if (Y_Pred_second_pass == 1):
            print("got 1 value")
            Y_Pred_first_pass[i] = 4

accuracy = give_error(Y_Pred_first_pass,Y_test, X_test)
print("Accuracy:" + str(accuracy*100))

