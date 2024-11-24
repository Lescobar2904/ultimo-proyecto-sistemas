import math
import numpy as np

# Diccionario de clases
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

# Función para calcular la distancia Euclidiana
def euclidean_distance(x, y):
    return math.sqrt(np.sum((x - y) ** 2))

# Función KNN
def KNN(dData, aTest, K=3):
    # Inicializar la lista de distancias
    distances = []
    
    # Calcular la distancia Euclidiana de cada punto de entrenamiento a aTest
    for i, x in enumerate(dData['X_train']):
        dE = euclidean_distance(x, aTest)  # Calculando la distancia Euclidiana
        distances.append((dE, dData['y_train'][i]))
    
    # Ordenar las distancias y seleccionar los K más cercanos
    distances = sorted(distances, key=lambda x: x[0])[:K]
    
    # Contar las clases de los K vecinos más cercanos
    class_count = {}
    for _, label in distances:
        class_count[label] = class_count.get(label, 0) + 1
    
    # Obtener la clase con mayor frecuencia (votación)
    predicted_class = max(class_count, key=class_count.get)
    
    # Devolver la clase predicha
    return predicted_class

# Función para dividir los datos en K folds (validación cruzada)
def cross_validate(X, y, K_folds=5, k_neighbors=3):
    # Dividir los datos en K folds
    fold_size = len(X) // K_folds
    accuracies = []
    all_predictions = {k: [] for k in range(K_folds)}  # Para almacenar las predicciones de cada fold
    
    # Iterar sobre cada fold
    for fold in range(K_folds):
        # Dividir en datos de entrenamiento y prueba
        X_test = X[fold * fold_size : (fold + 1) * fold_size]
        y_test = y[fold * fold_size : (fold + 1) * fold_size]
        X_train = np.concatenate([X[:fold * fold_size], X[(fold + 1) * fold_size:]], axis=0)
        y_train = np.concatenate([y[:fold * fold_size], y[(fold + 1) * fold_size:]], axis=0)
        
        # Guardar los datos de entrenamiento y prueba en un diccionario
        dData = {'X_train': X_train, 'y_train': y_train}
        
        # Evaluar en el conjunto de prueba
        correct = 0
        fold_predictions = []  # Guardar predicciones de este fold
        
        for i in range(len(X_test)):
            prediction = KNN(dData, X_test[i], K=k_neighbors)
            fold_predictions.append(dLabel[prediction])  # Almacenar la predicción en texto
            if prediction == dLabel[y_test[i]]:
                correct += 1
        
        # Calcular la precisión para este fold
        accuracy = correct / len(X_test)
        accuracies.append(accuracy)
        
        # Almacenar las predicciones de este fold
        all_predictions[fold] = fold_predictions
    
    # Devolver la precisión promedio de todos los folds y las predicciones de cada fold
    return np.mean(accuracies), all_predictions

# Cargar datos (ejemplo con datos generados)
# Aquí asumimos que tienes un archivo con las características y etiquetas
# En este caso, el conjunto de datos es generado aleatoriamente para la demostración
def load_data(file_path, has_labels=True):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if has_labels:
        return data[:, :-1], data[:, -1].astype(int)
    else:
        return data

# Cargar los datos de entrenamiento y prueba
X, y = load_data("data_training.txt", has_labels=True)

# Aplicar validación cruzada para diferentes valores de K
for k_neighbors in [2, 3, 4, 5]:
    accuracy, predictions = cross_validate(X, y, K_folds=5, k_neighbors=k_neighbors)
    print(f"\nPrecisión promedio con K = {k_neighbors}: {accuracy:.2f}")
    
    # Mostrar las predicciones por fold
    for fold, preds in predictions.items():
        print(f"Predicciones para el Fold {fold + 1}: {preds}")
