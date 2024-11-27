import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv("../../dataProcessing/output/output.csv")
df = df.drop(columns=["filename"])
df = df[[col for col in df.columns if col != "direction"] + ["direction"]]

# Select top 3 features
df = df[["eeg_1_mean", "eeg_2_mean", "eeg_3freq_std", "direction"]]

# Define features and target
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Testing random states from 1 to 100
best_random_state = None
best_accuracy = 0
results = []

for random_state in range(1, 101):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=random_state
    )

    # Instantiate and train the KNN classifier
    clf = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors as needed

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store the results
    results.append((random_state, accuracy))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_state

# Display all results
results_df = pd.DataFrame(results, columns=["Random State", "Accuracy"])
print(results_df.sort_values(by="Accuracy", ascending=False))
average_accuracy = sum([result[1] for result in results]) / len(results)
print("Average Accuracy:", average_accuracy)
