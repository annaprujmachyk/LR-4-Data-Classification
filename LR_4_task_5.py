
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score


df = pd.read_csv('data_metrics.csv')
print("Перші 5 рядків даних:")
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
print("\nДані з прогнозованими мітками:")
print(df.head())

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def my_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

print("\nМатриця помилок для RF:")
print("TP:", find_TP(df.actual_label.values, df.predicted_RF.values))
print("FN:", find_FN(df.actual_label.values, df.predicted_RF.values))
print("FP:", find_FP(df.actual_label.values, df.predicted_RF.values))
print("TN:", find_TN(df.actual_label.values, df.predicted_RF.values))

print("\nВласна матриця помилок для RF:")
print(my_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

print("\nБібліотечна матриця помилок для RF:")
print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))

def my_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

print("\nAccuracy RF: %.3f" % (my_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print("Accuracy LR: %.3f" % (my_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

def my_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

print("\nRecall RF: %.3f" % (my_recall_score(df.actual_label.values, df.predicted_RF.values)))
print("Recall LR: %.3f" % (my_recall_score(df.actual_label.values, df.predicted_LR.values)))

def my_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

print("\nPrecision RF: %.3f" % (my_precision_score(df.actual_label.values, df.predicted_RF.values)))
print("Precision LR: %.3f" % (my_precision_score(df.actual_label.values, df.predicted_LR.values)))

def my_f1_score(y_true, y_pred):
    recall = my_recall_score(y_true, y_pred)
    precision = my_precision_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

print("\nF1 RF: %.3f" % (my_f1_score(df.actual_label.values, df.predicted_RF.values)))
print("F1 LR: %.3f" % (my_f1_score(df.actual_label.values, df.predicted_LR.values)))

assert abs(my_f1_score(df.actual_label.values, df.predicted_RF.values) - f1_score(df.actual_label.values, df.predicted_RF.values)) < 0.001, 'my_f1_score failed on RF'
assert abs(my_f1_score(df.actual_label.values, df.predicted_LR.values) - f1_score(df.actual_label.values, df.predicted_LR.values)) < 0.001, 'my_f1_score failed on LR'
print("\nПеревірка f1_score пройдена успішно!")

print("\nФайл LR_4_task_5.py виконано успішно!")
