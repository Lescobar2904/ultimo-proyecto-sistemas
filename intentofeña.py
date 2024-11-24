import math
from collections import Counter
import random

# Diccionario para mapear etiquetas a nombres de clases
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

def load_data(file_path, has_class=True):
    """
    Cargar datos desde un archivo, ignorando la primera línea si contiene encabezados.
    
    Args:
    - file_path: Ruta al archivo de datos.
    - has_class: Indica si el archivo contiene la columna de clase.
    
    Returns:
    - Lista de datos: [[x1, x2, ..., clase]] si has_class=True, o [[x1, x2, ...]] si has_class=False.
    """
    data = []
    with open(file_path, 'r') as file:
        next(file)  # Saltar encabezados
        for line in file:
            values = list(map(float, line.strip().split(',')))
            data.append(values)
    return data

def split_data(data, train_ratio=0.8):
    """
    Dividir los datos en entrenamiento y validación.
    
    Args:
    - data: Lista de datos con características y etiquetas.
    - train_ratio: Proporción de datos para entrenamiento (entre 0 y 1).
    
    Returns:
    - train_data: Datos de entrenamiento.
    - test_data: Datos de validación.
    """
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

def KNN(dData, aTest, K=3):
    """
    Implementación del algoritmo KNN.
    
    Args:
    - dData: Lista de datos de entrenamiento.
    - aTest: Punto de prueba a clasificar.
    - K: Número de vecinos más cercanos.
    
    Returns:
    - Clase predicha para el punto de prueba.
    """
    aD = []
    for f in dData:
        features, label = f[:-1], int(f[-1])
        dE = math.sqrt(sum((a - b) ** 2 for a, b in zip(features, aTest)))  # Distancia Euclidiana
        aD.append((dE, label))
    aD = sorted(aD)[:K]  # Obtener los K vecinos más cercanos
    neighbors = [label for _, label in aD]
    most_common = Counter(neighbors).most_common(1)[0][0]  # Clase más frecuente
    return most_common

def calculate_accuracy(predictions, true_labels):
    """
    Calcular la precisión (accuracy) del modelo.
    
    Args:
    - predictions: Lista de clases predichas.
    - true_labels: Lista de clases verdaderas.
    
    Returns:
    - Accuracy: Porcentaje de predicciones correctas.
    """
    correct = sum([1 if p == t else 0 for p, t in zip(predictions, true_labels)])
    accuracy = (correct / len(true_labels)) * 100
    return accuracy

def classify_and_display(dTrain, dTest, K, true_labels):
    """
    Clasificar puntos de prueba, mostrar resultados y calcular el accuracy.
    
    Args:
    - dTrain: Lista de datos de entrenamiento.
    - dTest: Lista de puntos de prueba.
    - K: Número de vecinos más cercanos.
    - true_labels: Lista de etiquetas reales de los puntos de prueba.
    
    Returns:
    - Lista de clases predichas.
    """
    predictions = []
    print(f"\nClasificación para K={K}:")
    for i, test_point in enumerate(dTest, start=1):
        predicted_class = KNN(dTrain, test_point[:-1], K)
        predictions.append(predicted_class)
        print(f"Punto de prueba {i}: {test_point[:-1]} -> Clase predicha: {dLabel[predicted_class]}")
    
    # Calcular y mostrar accuracy
    accuracy = calculate_accuracy(predictions, true_labels)
    print(f"Accuracy para K={K}: {accuracy:.2f}%")
    
    return predictions

# Archivos de datos
training_file = "data_training.txt"
test_file = "data_test.txt"

# Cargar datos
data = load_data(training_file, has_class=True)

# Dividir datos en 80% entrenamiento y 20% validación
train_data, validation_data = split_data(data, train_ratio=0.8)

# Extraer etiquetas reales del conjunto de validación
validation_features = [x[:-1] for x in validation_data]
validation_labels = [int(x[-1]) for x in validation_data]

# Probar para valores de K (2, 3, 4, 5)
for k in range(2, 6):
    classify_and_display(train_data, validation_data, k, validation_labels)
