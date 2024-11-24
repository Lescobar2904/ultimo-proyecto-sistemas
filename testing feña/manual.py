import math
import numpy as np

# Etiquetas de las clases
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

# Función para cargar datos desde archivo
def load_data(file_path, has_labels):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if has_labels:
        return data[:, :-1], data[:, -1].astype(int)  # Características (X) y etiquetas (y)
    else:
        return data  # Solo características (sin etiquetas)

# Implementación del algoritmo KNN manual
def KNN(dData, aTest, K=3):
    X_train, y_train = dData
    distances = []
    for i, x in enumerate(X_train):
        dE = math.sqrt(np.sum((x - aTest) ** 2))
        distances.append((dE, y_train[i]))
    distances = sorted(distances, key=lambda x: x[0])[:K]
    class_count = {}
    for _, label in distances:
        class_count[label] = class_count.get(label, 0) + 1
    predicted_class = max(class_count, key=class_count.get)
    return predicted_class

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
    dD = (X_train, y_train)

    for K in [2, 3, 4, 5]:
        print(f"\nProbando con K = {K}")

        # Predicciones manuales
        manual_predictions = [KNN(dD, x, K) for x in X_val]
        manual_accuracy = sum(1 for i in range(len(y_val)) if manual_predictions[i] == y_val[i]) / len(y_val)
        print(f"Exactitud manual en validación: {manual_accuracy:.2f}")

        # Predicciones para el conjunto de prueba
        manual_test_predictions = [KNN(dD, sample, K) for sample in aT]
        print(f"Predicciones manuales para el conjunto de prueba con K = {K}:")
        print([dLabel[pred] for pred in manual_test_predictions])
