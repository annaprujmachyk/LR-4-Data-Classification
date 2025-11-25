
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print(f"Accuracy (всі дані): {accuracy:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)
accuracy_new = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print(f"Accuracy (тестові дані): {accuracy_new:.2f}%")

accuracy_cv = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print(f"Accuracy (CV): {100 * accuracy_cv.mean():.2f}%")
