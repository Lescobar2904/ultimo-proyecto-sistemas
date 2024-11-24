import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Etiquetas para las clases
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

# Función para cargar datos
def load_data(file_path, has_labels=True):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Saltar la fila de encabezados
    if has_labels:
        X = data[:, :-1]  # Características
        y = data[:, -1].astype(int)  # Etiquetas
        return X, y
    else:
        return data  # Solo características

# 1. Cargar los datos de entrenamiento
X, y = load_data("data_training.txt", has_labels=True)

# Dividir el conjunto de entrenamiento en 80% entrenamiento y 20% validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Cargar los datos de prueba (sin etiquetas)
X_test = load_data("data_test.txt", has_labels=False)

# 3. Probar clasificación con K = 2, 3, 4, 5
k_values = [2, 3, 4, 5]

for k in k_values:
    print(f"\nProbando con k = {k}")
    # Crear el modelo k-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenar el modelo
    knn.fit(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de validación
    y_val_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Exactitud en validación (Accuracy): {accuracy:.2f}")
    
    # Reporte de clasificación
    print("Reporte de Clasificación en Validación:")
    print(classification_report(y_val, y_val_pred, target_names=[dLabel[i] for i in range(3)]))
    
    # Predecir las etiquetas para el conjunto de prueba
    y_test_pred = knn.predict(X_test)
    print(f"Predicciones para el conjunto de prueba con k = {k}:")
    print([dLabel[label] for label in y_test_pred])
