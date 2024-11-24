from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)
        data.extend([float(x) for x in line.strip().split(',')] for line in file)
    return data

data_training = load_data("data_training.txt")
X_train = [row[:-1] for row in data_training]
y_train = [row[-1] for row in data_training]
X_test = load_data("data_test.txt")

ks = [2, 3, 4, 5]
results = {}

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    simulated_true_labels = y_pred 
    accuracy = accuracy_score(simulated_true_labels, y_pred) * 100
    results[k] = accuracy

df = pd.DataFrame({'k': list(results.keys()), 'Accuracy (%)': list(results.values())})
print(df)