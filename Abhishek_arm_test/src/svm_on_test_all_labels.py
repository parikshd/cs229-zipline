# Kernel SVM
# Importing the libraries

import numpy as np
import pandas as pd
import sklearn
import util

def give_error(y_out,class_probabilities, y, x):
    cnt = 0
    cntfalse = 0
    cntBadones = 0
    cntActualTwos = 0
    for i in range(len(y_out)):
        if (y_out[i] == 1.0 and y[i] == 2.0):
            cntBadones += 1
        if (y[i] == 2.0):
            cntActualTwos += 1
        if (y_out[i] == y[i]):
            #print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
            cnt += 1
        else:
            if (y_out[i] != 1.0):
                print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
                print("%success=" + str(class_probabilities[i][0]*100) + " %mission-failure=" + str(class_probabilities[i][1]*100) + " %flight-failure=" + str(class_probabilities[i][2]*100))
            cntfalse += 1

            #     #print("Flight " + str(int(x[i][flight_id_index])) + " might need maintaince, our algorithm predicted it would have mission failure!")
            # if (y_out[i] == 4):
            #     #print("Flight " + str(int(x[i][flight_id_index])) + " definitely needs maintaince, our algorithm predicted it would have flight failure!")
    print("Predicted " + str((len(y_out) - cntBadones)) + "/" + str(len(y_out)) + " correctly.")
    print("Predicted " + str(cntfalse) + "/" + str(len(y_out)) + " incorrectly.")
    print("Predicted " + str(cntBadones) + "/" + str(cntActualTwos) + " incorrectly. ==> predicted 1, actual 2")
    model_failure = (cntActualTwos - cntBadones)/ cntActualTwos
    total_failure = cnt /len(y_out)

    return total_failure,model_failure

train_path = "output/flights_pass_1_na_0.csv"
#eval_path = "output/flights_pass_1_na_0.csv"
eval_path = "testinput/all_test_with_failures_clean.csv"
X, Y,X_test,Y_test,dataset = util.load_dataset_new(train_path,eval_path)

with open('featues_new.txt', 'w') as f:
    for item in dataset.columns:
        f.write("%s\n" % item)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test_transformed = sc_X.fit_transform(X_test)
# Fitting the classifier into the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto',probability=True)

# Predicting the test set results
classifier.fit(X,Y)

# Y_pred_train = classifier.predict(X_Train)
# print(give_error(Y_pred_train,Y_Train))
#w = classifier.coef_
#print('w = ',w)

Y_Pred_first_pass = classifier.predict(X_test_transformed)
class_probabilities = classifier.predict_proba(X_test_transformed)
for i in range(len(Y_Pred_first_pass)):
    if class_probabilities[i][1] > class_probabilities[i][0]:
        Y_Pred_first_pass[i] = 1.0
    else:
        Y_Pred_first_pass[i] = 0.0
    if (class_probabilities[i][2] > class_probabilities[i][1]) and (
            class_probabilities[i][2] > class_probabilities[i][0]):
        Y_Pred_first_pass[i] = 2.0

    if class_probabilities[i][0] >= 0.85:
        Y_Pred_first_pass[i] = 0.0
    if class_probabilities[i][1] >= 0.15:
        Y_Pred_first_pass[i] = 1.0
    if class_probabilities[i][2] >= 0.15:
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

total_accuracy,model_accuracy = give_error(Y_Pred_first_pass,class_probabilities,Y_test, X_test)
print("Total accuracy:" + str(total_accuracy*100))
print("Model accuracy:" + str(model_accuracy*100))


