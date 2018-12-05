import numpy as np
import pandas as pd
import sklearn
from sklearn import tree


def give_error(y_out,class_probabilities, y, x):
    cnt = 0
    cntfalse = 0
    for i in range(len(y_out)):
        if (y_out[i] == y[i]):
            #print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
            cnt += 1
        else:
            print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
            print("%success=" + str(class_probabilities[i][0]*100) + " %mission-failure=" + str(class_probabilities[i][1]*100) + " %flight-failure=" + str(class_probabilities[i][2]*100))
            cntfalse += 1
            if (y_out[i] == 2):
                print("Flight " + str(int(x[i][flight_id_index])) + " might need maintaince, our algorithm predicted it would have mission failure!")
            if (y_out[i] == 4):
                print("Flight " + str(int(x[i][flight_id_index])) + " definitely needs maintaince, our algorithm predicted it would have flight failure!")
    print("Predicted " + str(cnt) + "/" + str(len(y_out)) + " correctly.")
    print("Predicted " + str(cntfalse) + "/" + str(len(y_out)) + " incorrectly.")
    return cnt / len(y_out)

datasets = pd.read_csv("output/flights_pass_1_na_0.csv")
datasets_test = pd.read_csv('testinput/flights_new_till_03dec.csv')

test_columns = datasets_test.columns
train_columns = datasets.columns

to_del_test_columns = np.setdiff1d(test_columns,train_columns)
datasets_test.drop(to_del_test_columns, axis=1, inplace=True)

datasets = datasets[datasets_test.columns]

datasets['highest_failure_level.id'] = datasets['highest_failure_level.id'].astype(float)
datasets_test['highest_failure_level.id'] = datasets_test['highest_failure_level.id'].astype(float)

print("Model trained on " + str(datasets.shape[0]) + " flights with " + str(datasets.shape[1]) + " features")
print("Runing tests on " + str(datasets_test.shape[0]) + " flights")

flight_id_index = datasets_test.columns.get_loc("config.flight_id")

for idx, row in datasets.iterrows():
    if  datasets.loc[idx,'highest_failure_level.id'] == 1:
        datasets.loc[idx,'highest_failure_level.id'] = 0
    if  datasets.loc[idx,'highest_failure_level.id'] == 2:
        datasets.loc[idx,'highest_failure_level.id'] = 1
    if  datasets.loc[idx,'highest_failure_level.id'] == 4:
        datasets.loc[idx,'highest_failure_level.id'] = 2

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

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

Y_Pred_first_pass = clf.predict(X_test)
class_probabilities = clf.predict_proba(X_test)
for i in range(len(Y_Pred_first_pass)):
    # if class_probabilities[i][1] > 0.6:
    #     Y_Pred_first_pass[i] = 1.0
    # elif class_probabilities[i][2] > 0.6:
    #     Y_Pred_first_pass[i] = 2.0

    if class_probabilities[i][1] > class_probabilities[i][0]:
        Y_Pred_first_pass[i] = 1.0
    if (class_probabilities[i][2] > class_probabilities[i][1]) and (class_probabilities[i][2] > class_probabilities[i][0]):
        Y_Pred_first_pass[i] = 2.0

    if Y_Pred_first_pass[i] == 0.0:
        Y_Pred_first_pass[i] = 1.0
    elif Y_Pred_first_pass[i] == 1.0:
        Y_Pred_first_pass[i] = 2.0
    elif Y_Pred_first_pass[i] == 2.0:
        Y_Pred_first_pass[i] = 4.0

for i in range(len(Y_Pred_first_pass)):
    print(Y_Pred_first_pass[i])
    print(class_probabilities[i])

accuracy = give_error(Y_Pred_first_pass,class_probabilities,Y_test, X_test)
print("Accuracy:" + str(accuracy*100))