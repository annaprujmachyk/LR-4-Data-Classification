
import numpy as np
from sklearn import preprocessing

input_data = np.array([[1.3, 3.9, 6.2],
                       [4.9, 2.2, -4.3],
                       [-2.2, 6.5, 4.1],
                       [-5.2, -3.4, -5.2]])

print("Вхідні дані (варіант №7):")
print(input_data)

data_binarized = preprocessing.Binarizer(threshold=2.0).transform(input_data)
print("\\nБінаризовані дані:")
print(data_binarized)

data_scaled = preprocessing.scale(input_data)
print("\\nДані після виключення середнього:")
print(data_scaled)

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
