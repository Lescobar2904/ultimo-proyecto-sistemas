import math as mt

# Etiquetas para las clases dentro de un diccionario
dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}



# funcion del algoritmo KNN
def KNN(dData, aTest, K=3):
    # arreglo 
    aD = []

    for c in dData:
        #-----------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------
        # Convertir los datos a listas de flotantes para calcular la distancia
        # map() combierte la linea del txt en valores flotantes
        # list lo convierte en una lista
        dFeatures = list(map(float, c[:-1]))  # Características del conjunto de entrenamiento
        testFeatures = list(map(float, aTest))  # Características del dato de prueba
        #-----------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------
        # Cálculo de la distancia euclidiana
        # formula = √(∑(Yi- Xj)^2)
        dE = mt.sqrt(sum((dF - tF) ** 2 for dF, tF in zip(dFeatures, testFeatures)))
        # Almacenar la distancia y la etiqueta
        aD.append((dE, c[-1]))
        #-----------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------
    
    # Ordenar por distancia y seleccionar los K vecinos más cercanos
    # La funcion sorted ordena los valores de aD 
    # y ademas deja solo los primeros K elementos (predeterminado 3)
    aD = sorted(aD, key=lambda x: x[0])[:K]
    # Contar la frecuencia de cada clase en los K vecinos
    clase_contadores = {}
    for _, clase in aD:
        clase_contadores[clase] = clase_contadores.get(clase, 0) + 1

    # Determinar la clase más frecuente
    prediccion = max(clase_contadores, key=clase_contadores.get)
    return prediccion

def archivo(nombre_archivo, tiene_etiqueta=True):
    arreglo = []
    try:
        with open(nombre_archivo, 'r') as f:
            # Omitir la primera línea de encabezado
            next(f)
            for linea in f:
                datos = linea.strip().split(',')
                if tiene_etiqueta:
                    # Último valor es la etiqueta
                    arreglo.append(datos[:-1] + [int(datos[-1])])
                else:
                    # No tiene etiqueta, usar solo las características
                    arreglo.append(datos)
    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no existe.")
    except Exception as e:
        print(f"Se produjo un error: {e}")
    return arreglo

# Cargar los datos de entrenamiento y prueba
dD = archivo('data_training.txt')
aT = archivo('data_test.txt', tiene_etiqueta=False)

# Ejecutar KNN para cada punto de prueba
for aTest in aT:
    prediccion = KNN(dD, aTest)
    print(f"Test: {aTest} -> Predicción: {dLabel[int(prediccion)]}")
