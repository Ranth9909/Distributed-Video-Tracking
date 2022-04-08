from statistics import mode
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import load_model
from Collect_Train_data import *
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay

#res = y_test - This data has category labels for your test data, these labels will be used to test the accuracy between actual and predicted categories
#test_input = X_test - which will not be used in the training phase and will be used to make predictions to test the accuracy of the model.
#y_hat = y_train -  dependent variable which needs to be predicted by this model

test_input, test_val, y_true = test_data.as_numpy_iterator().next()  
y_hat = siamese_model.predict([test_input, test_val])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
#Plot ROC curve with roc_auc score
fpr, tpr, _ = metrics.roc_curve(y_true,  y_hat)
roc = metrics.roc_auc_score(y_true,  y_hat)
roc_disp = RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc, estimator_name =  "Roc Curve")
roc_disp.plot(ax=ax1)
_ = roc_disp.ax_.set_title("ROC Curve")
#Plot Precision Recall Curve with Average precision
prec, recall, _ = metrics.precision_recall_curve(y_true, y_hat)
ap = metrics.average_precision_score(y_true, y_hat) 
pr_disp = PrecisionRecallDisplay(precision=prec, recall=recall, average_precision = ap, estimator_name = "Average precision")
pr_disp.plot(ax=ax2)
_ = pr_disp.ax_.set_title("Precision Recall Curve")
#Combining the display objects into a single plot
plt.show()

