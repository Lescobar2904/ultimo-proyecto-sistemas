import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Etiquetas de las clases
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

# Función para cargar datos desde archivo
def load_data(file_path, has_labels):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if has_labels:
        return data[:, :-1], data[:, -1].astype(int)  # Características (X) y etiquetas (y)
    else:
        return data  # Solo características (sin etiquetas)

# Main
if __name__ == "__main__":
    np.random.seed(1)  # Fijar la semilla aleatoria para consistencia

    # Cargar datos
    X, y = load_data("data_training.txt", has_labels=True)
    aT = load_data("data_test.txt", has_labels=False)
    num_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    X_train, y_train = X[indices[:num_train]], y[indices[:num_train]]
    X_val, y_val = X[indices[num_train:]], y[indices[num_train:]]

    for K in [2, 3, 4, 5]:
        print(f"\nProbando con K = {K}")

        # Usar scikit-learn para KNN
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(X_train, y_train)
        sklearn_predictions = knn.predict(X_val)
        sklearn_accuracy = accuracy_score(y_val, sklearn_predictions)
        print(f"Exactitud con scikit-learn en validación: {sklearn_accuracy:.2f}")

        # Predicciones para el conjunto de prueba
        sklearn_test_predictions = knn.predict(aT)
        print(f"Predicciones con scikit-learn para el conjunto de prueba con K = {K}:")
        print([dLabel[pred] for pred in sklearn_test_predictions])
