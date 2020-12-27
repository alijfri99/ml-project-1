import pandas as pd
from helper_functions import *
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def hybrid_tree(subset_size, max_count):

    train = pd.read_csv("dataset/train.csv")
    train_labels = pd.read_csv("dataset/train_labels.csv")
    test = pd.read_csv("dataset/test.csv")
    test_labels = pd.read_csv("dataset/test_labels.csv")

    columns = {'PassengerId': False, 'Pclass': False, 'Name': True, 'Sex': True, 'Age': False, 'SibSp': False,
               'Parch': False, 'Ticket': False, 'Fare': False, 'Cabin': True, 'Embarked': True}
    prepare_columns(train, columns)
    prepare_columns(test, columns)

    train_subset = train.head(subset_size)
    train_labels_subset = train_labels.head(subset_size)

    scaler = StandardScaler()
    train_subset = scaler.fit_transform(train_subset)

    classifier = Perceptron()
    classifier.fit(train_subset, train_labels_subset.values.ravel())
    coefs = abs(classifier.coef_)[0]
    coefs = pd.Series(coefs)
    coefs = coefs.nlargest(max_count)
    indices = coefs.index.values.tolist()

    chosen_columns = []
    for i in indices:
        chosen_columns.append(list(columns)[i])

    train = train[chosen_columns]
    test = test[chosen_columns]

    classifier = DecisionTreeClassifier()
    classifier.fit(train, train_labels.values.ravel())
    pred_labels = classifier.predict(test)
    accuracy = accuracy_score(test_labels, pred_labels)
    return accuracy, chosen_columns


maximum_accuracy = 0.0
optimal_subset_count = 0
optimal_max_count = 0
for subset_count in range(3, 801):
    for max_count in range(1, 12):
        if subset_count % 10 ==0:
            print(subset_count, max_count)
        accuracy, chosen_columns = hybrid_tree(subset_count, max_count)
        if accuracy > maximum_accuracy:
            maximum_accuracy = accuracy
            optimal_subset_count = subset_count
            optimal_max_count = max_count

print("Result:", optimal_subset_count, optimal_max_count)  # 304, 5 OR 446, 6 OR 606, 5 OR 670, 7

accuracy, chosen_columns = hybrid_tree(optimal_subset_count, optimal_max_count)
print(chosen_columns)
print("Accuracy:", accuracy)

