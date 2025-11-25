
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utilities import visualize_classifier

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5],
              [6, 5], [5.6, 5], [3.3, 0.4],
              [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier.fit(X, y)

print("Модель навчена!")

visualize_classifier(classifier, X, y)
plt.title('Логістична регресія - Межі класифікації')
plt.show()

test_points = np.array([[4.5, 6.0], [3.0, 2.0], [1.5, 4.0]])
predictions = classifier.predict(test_points)
print("Передбачення для тестових точок:", predictions)
