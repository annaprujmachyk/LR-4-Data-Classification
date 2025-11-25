
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

print("Вхідні дані:")
print(input_data)

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\\nБінаризовані дані:")
print(data_binarized)

print("\\nДО виключення середнього:")
print("Середнє =", input_data.mean(axis=0))
print("Стандартне відхилення =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\\nПІСЛЯ виключення середнього:")
print("Середнє =", data_scaled.mean(axis=0))
print("Стандартне відхилення =", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\\nМасштабовані дані (MinMax):")
print(data_scaled_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\\nL1 нормалізовані дані:")
print(data_normalized_l1)
print("\\nL2 нормалізовані дані:")
print(data_normalized_l2)

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

print("\\nВідображення міток:")
for i, item in enumerate(encoder.classes_):
    print(f"{item} -> {i}")

test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print(f"\\nМітки: {test_labels}")
print(f"Закодовані значення: {list(encoded_values)}")

encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print(f"\\nЗакодовані значення: {encoded_values}")
print(f"Декодовані мітки: {list(decoded_list)}")
