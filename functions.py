import numpy as np
import glob
import os
import random
from time import time
from sklearn import svm
import argparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from skimage import io
from string import digits
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy import interp

def load_images_from_folder(folder):
    images = []
    classes = []
    classIndexes = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
        cl = os.path.splitext(os.path.basename(filename))[0]
        cl = cl.translate(None, digits)
        if cl not in classes:
            classes.append(cl)
    return images, classes

def load_classes_from_folder(folder, names):
    classes = []
    for filename in os.listdir(folder):
        cl = os.path.splitext(os.path.basename(filename))[0]
        cl = cl.translate(None, digits)
        clIndex = names.index(cl)
        namesIndex = [0] * len(names)
        namesIndex[clIndex] = 1
        classes.append(namesIndex)
    return classes

def compute_pca(X_train, y_train, comparator = LinearRegression()):
    n_components = min(X_train.shape[0], X_train.shape[1])
    pipe = Pipeline([('pca', PCA(n_components, svd_solver='randomized')),
                     ('linear', comparator)])
    pipe.fit(X_train, y_train)
    return pipe

def compute_cv_pca(X_train, y_train, X_test, cross_validation):
    scores = []
    comparator = LinearRegression()
    for i in range(cross_validation):
        clf = compute_pca(X_train, y_train, comparator)
        predictions = clf.predict(X_test);
        scores.append(predictions)
    scores = np.array(scores)
    mean = np.mean(scores, axis=0)

    return mean

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def pred_approx_val(arr, treshold):
    array_np = np.copy(arr)
    low_val_indices = arr < treshold
    high_val_indices = arr >= treshold
    array_np[low_val_indices] = 0
    array_np[high_val_indices] = 1
    return array_np

def perf_measure(actual, score, treshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    size = len(score)
    for i in range(size):
        predArr = pred_approx_val(score[i], treshold)
        for j in range(len(score[i])):
            if(predArr[j] != actual[i][j] and predArr[j] == 1):
                FP+=1
            if(predArr[j] == actual[i][j] == 1):
                TP+=1
            if(predArr[j] != actual[i][j] and predArr[j] == 0):
                FN+=1
            if(predArr[j] == actual[i][j] == 0):
                TN+=1
    return TP, FP, TN, FN

def calc_far_frr(FP, FN, totalP, totalN):
    FAR = FP/float(totalP) # False Accept Rate in percentage
    FRR = FN/float(totalN) # False Reject Rate in percentage
    return FAR, FRR

def prepare_graph_far_frr(actual, score, totalP, totalN):
    step = 1
    far = dict()
    frr = dict()

    for i in xrange(0, 100, step):
        _, FP, _, FN = perf_measure(actual, score, i/float(100))
        far[i], frr[i] = calc_far_frr(FP, FN, totalP, totalN)
    return far, frr

# Parameters FAR and FRR in dict() format
def plot_far_frr(far, frr):

    axisVal = np.arange(0,1.00,0.01);

    # PLOT FAR FRR
    plt.figure()
    lw = 2
    plt.plot(far.values(), axisVal, label='False Accept Rate', color='blue', lw=lw)
    plt.plot(axisVal, frr.values(), label='False Reject Rate', color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Treshold')
    plt.ylabel('Errors')
    plt.title('FAR and FRR')
    plt.legend(loc="lower right")

def compute_roc(n_classes, y_score, y_test):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return tpr, fpr, roc_auc

def plot_roc(tpr, fpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='micro-average ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
                    color='blue',linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")