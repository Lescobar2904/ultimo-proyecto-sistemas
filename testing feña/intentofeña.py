import math
import numpy as np

dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

# Cargar datos desde archivo y manejar encabezados
def load_data(file_path, has_labels):
    # Saltar la primera fila si tiene encabezados
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if has_labels:
        return data[:, :-1], data[:, -1].astype(int)  # Características (X) y etiquetas (y)
    else:
        return data  # Solo características (sin etiquetas)

# Implementación de KNN
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
    return dLabel[predicted_class]

# Main
if __name__ == "__main__":
    # Cargar los datos
    X, y = load_data("data_training.txt", has_labels=True)
    aT = load_data("data_test.txt", has_labels=False)
    num_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    X_train, y_train = X[indices[:num_train]], y[indices[:num_train]]
    X_val, y_val = X[indices[num_train:]], y[indices[num_train:]]
    dD = (X_train, y_train)
    for K in [2, 3, 4, 5]:
        correct_predictions = 0
        print(f"\nProbando con K = {K}")
        for i, x in enumerate(X_val):
            predicted_label = KNN(dD, x, K)
            actual_label = dLabel[y_val[i]]
            correct_predictions += int(predicted_label == actual_label)
        accuracy = correct_predictions / len(X_val)
        print(f"Exactitud en validación: {accuracy:.2f}")
        print(f"Predicciones para el conjunto de prueba con K = {K}:")
        predictions = [KNN(dD, sample, K) for sample in aT]
        print(predictions)
