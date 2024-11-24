import math
from collections import Counter

# Diccionario para mapear etiquetas a nombres de clases
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

def load_data(file_path, has_class=True):
    """
    Cargar datos desde un archivo, ignorando la primera línea si contiene encabezados.
    
    Args:
    - file_path: Ruta al archivo de datos.
    - has_class: Indica si el archivo contiene la columna de clase.
    
    Returns:
    - Si has_class=True: Un diccionario {clase: [[x1, x2, x3, x4], ...]}.
    - Si has_class=False: Una lista de listas [[x1, x2, x3, x4], ...].
    """
    data = {} if has_class else []
    with open(file_path, 'r') as file:
        next(file)  # Saltar encabezados
        for line in file:
            values = list(map(float, line.strip().split(',')))
            if has_class:
                x, c = values[:-1], int(values[-1])  # Separar características y clase
                if c not in data:
                    data[c] = []
                data[c].append(x)
            else:
                data.append(values)
    return data

def KNN(dData, aTest, K=3):
    """
    Implementación del algoritmo KNN.
    
    Args:
    - dData: Diccionario de datos de entrenamiento.
    - aTest: Punto de prueba a clasificar.
    - K: Número de vecinos más cercanos.
    
    Returns:
    - Clase predicha para el punto de prueba.
    """
    aD = []
    for c in dData:
        for f in dData[c]:
            dE = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, aTest)))  # Distancia Euclidiana
            aD.append((dE, c))
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

def classify_and_display(dData, aTestData, K, true_labels=None):
    """
    Clasificar puntos de prueba, mostrar resultados y calcular el accuracy.
    
    Args:
    - dData: Diccionario de datos de entrenamiento.
    - aTestData: Lista de características de los puntos de prueba.
    - K: Número de vecinos más cercanos.
    - true_labels: Lista de etiquetas reales de los puntos de prueba (opcional).
    
    Returns:
    - Lista de clases predichas.
    """
    predictions = []
    print(f"\nClasificación para K={K}:")
    for i, test_point in enumerate(aTestData, start=1):
        predicted_class = KNN(dData, test_point, K)
        predictions.append(predicted_class)
        print(f"Punto de prueba {i}: {test_point} -> Clase predicha: {dLabel[predicted_class]}")
    
    # Calcular y mostrar accuracy si tenemos etiquetas reales
    if true_labels:
        accuracy = calculate_accuracy(predictions, true_labels)
        print(f"Accuracy para K={K}: {accuracy:.2f}%")
    
    return predictions

# Archivos de datos
training_file = "data_training.txt"
test_file = "data_test.txt"

# Cargar datos
dD = load_data(training_file, has_class=True)  # Datos de entrenamiento
aT = load_data(test_file, has_class=False)  # Datos de prueba (sin etiquetas)

# Simulando etiquetas reales de test (si se conocen), por ejemplo, usando las etiquetas de entrenamiento
# Si no tienes las etiquetas reales de los datos de prueba, debes proporcionar un archivo con esas etiquetas
# Para simplificar, aquí se simula usando una lista de etiquetas ficticias para test
# Suponiendo que tienes una función para obtener las etiquetas reales de los datos de prueba.

# Para fines del ejemplo, utilizamos las clases correspondientes a las etiquetas de prueba
# Simularemos que el dataset tiene 3 clases (0, 1, 2), y asignamos aleatoriamente a los puntos de prueba
true_labels = [0, 1, 2, 1, 0]  # Esto debe ser reemplazado con las etiquetas reales de los datos de prueba

# Probar para valores de K (2, 3, 4, 5)
for k in range(2, 6):
    predictions = classify_and_display(dD, aT, k, true_labels)
