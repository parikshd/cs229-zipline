# Kernel SVM
# Importing the libraries

import numpy as np
import pandas as pd
import sklearn
import util
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def give_error(y_out,y):
    cnt = 0
    cntfalse = 0
    for i in range(len(y_out)):
        if (y_out[i] == y[i]):
            #print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
            cnt += 1
        else:
            # print("Predicted:" + str(y_out[i]) + ",actual:" + str(y[i]))
            # print("%success=" + str(class_probabilities[i][0]*100) + " %mission-failure=" + str(class_probabilities[i][1]*100) + " %flight-failure=" + str(class_probabilities[i][2]*100))
            cntfalse += 1
            # if (y_out[i] == 2):
            #     #print("Flight " + str(int(x[i][flight_id_index])) + " might need maintaince, our algorithm predicted it would have mission failure!")
            # if (y_out[i] == 4):
            #     #print("Flight " + str(int(x[i][flight_id_index])) + " definitely needs maintaince, our algorithm predicted it would have flight failure!")
    print("Predicted " + str(cnt) + "/" + str(len(y_out)) + " correctly.")
    print("Predicted " + str(cntfalse) + "/" + str(len(y_out)) + " incorrectly.")
    return cnt / len(y_out)

train_path = "output/flights_pass_1_na_0.csv"
eval_path = "testinput/flights_new_till_03dec.csv"
X, Y,X_test,Y_test,dataset = util.load_dataset_new(train_path,eval_path)

# Fitting the classifier into the Training set
from sklearn.svm import SVC
regression_model = LinearRegression()
regression_model.fit(X, Y)

# Y_pred_train = classifier.predict(X_Train)
# print(give_error(Y_pred_train,Y_Train))
#w = classifier.coef_
#print('w = ',w)
print("Score:")
print(regression_model.score(X_test, Y_test))

y_predict = regression_model.predict(X_test)
#print(y_predict)
regression_model_mse = mean_squared_error(y_predict, Y_test)
print(regression_model_mse)


# pass -> 0
# mission -> 1
# flight - > 2

for i in range(len(y_predict)):
    old = str(y_predict[i])
    if y_predict[i] <= 0.5:
        y_predict[i] = 1.0
    elif y_predict[i] > 0.5 and y_predict[i] <= 1.5:
        y_predict[i] = 2.0
    elif y_predict[i] > 1.5:
        y_predict[i] = 4.0
    print("regress:" + old + ",new:" + str(y_predict[i]) + ",actual:" + str(Y_test[i]))

accuracy = give_error(y_predict,Y_test)
print("Accuracy:" + str(accuracy*100))

