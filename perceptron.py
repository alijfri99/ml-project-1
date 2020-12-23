import pandas as pd
from helper_functions import *
from sklearn.linear_model import Perceptron
from sklearn import metrics

train = pd.read_csv("dataset/train.csv")
train_labels = pd.read_csv("dataset/train_labels.csv")
test = pd.read_csv("dataset/test.csv")
test_labels = pd.read_csv("dataset/test_labels.csv")

columns = {'PassengerId': False, 'Pclass': False, 'Name': True, 'Sex': True, 'Age': False, 'SibSp': False,
           'Parch': False, 'Ticket': False, 'Fare': False, 'Cabin': True, 'Embarked': True}
prepare_columns(train, columns)
prepare_columns(test, columns)

ppn = Perceptron()
ppn.fit(train, train_labels.values.ravel())
pred_labels = ppn.predict(test)
print("Accuracy:", metrics.accuracy_score(test_labels, pred_labels))
