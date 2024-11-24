import math

dLabel = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'}

def KNN(dData, aTest, K=3):
    aD = []
    for c in dData:
        for f in dData[c]:
            dE = math.sqrt('Calculo Distancia Euclidiana')
            aD.append((dE, c))
    aD = sorted(aD[:k])
    ...
    for d in aD:
        ...
        ...  # -> Completar
    ...
    return

# Obs:
# 1.- Cargue en dD <-- data_training.txt
# 2.- Cargue en aT <-- data_test.txt
# 3.- Probar clasificación con K = 2, 3, 4 y 5.

for aTest in aT:
    print(KNN(dD, aTest, nK))
