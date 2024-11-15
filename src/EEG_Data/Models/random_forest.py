import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("../output.csv")

# Get all columsn except the last one
x = df.iloc[:, :-1]

# Get the last column (target)
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=99
)

clf = RandomForestClassifier(
    criterion="gini", max_depth=8, min_samples_split=10, random_state=5
)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Accuracy")
print(accuracy_score(y_test, y_pred))

print("Cross Validation")
print(cross_val_score(clf, x_train, y_train, cv=10))

print("Classification Report")
print(classification_report(y_pred, y_test))
