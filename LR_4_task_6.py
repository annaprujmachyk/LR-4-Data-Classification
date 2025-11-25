
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
print(f"Розмірність даних: {X.shape}")
print(f"Кількість класів: {len(np.unique(y))}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Тренувальна вибірка: {X_train.shape[0]} samples")
print(f"Тестова вибірка: {X_test.shape[0]} samples")

svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Точність SVM: {accuracy_svm:.4f}")
print(f"Точність Naive Bayes: {accuracy_nb:.4f}")

print("\nЗвіт класифікації для SVM:")
print(classification_report(y_test, y_pred_svm))
print("\nЗвіт класифікації для Naive Bayes:")
print(classification_report(y_test, y_pred_nb))

svm_cv_scores = cross_val_score(svm_classifier, X, y, cv=5, scoring='accuracy')
nb_cv_scores = cross_val_score(nb_classifier, X, y, cv=5, scoring='accuracy')
print(f"SVM крос-валідація: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})")
print(f"Naive Bayes крос-валідація: {nb_cv_scores.mean():.4f} (+/- {nb_cv_scores.std() * 2:.4f})")

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
visualize_classifier(svm_classifier, X_test, y_test)
plt.title('SVM Classifier')
plt.subplot(1, 2, 2)
visualize_classifier(nb_classifier, X_test, y_test)
plt.title('Naive Bayes Classifier')
plt.tight_layout()
plt.show()

print("Матриця помилок SVM:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nМатриця помилок Naive Bayes:")
print(confusion_matrix(y_test, y_pred_nb))

print("\n=== ПОРІВНЯЛЬНИЙ АНАЛІЗ ===")
print(f"SVM точність: {accuracy_svm:.4f}")
print(f"Naive Bayes точність: {accuracy_nb:.4f}")
if accuracy_svm > accuracy_nb:
    print("SVM показує кращу точність")
elif accuracy_nb > accuracy_svm:
    print("Naive Bayes показує кращу точність")
else:
    print("Обидві моделі показують однакову точність")
