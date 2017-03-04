import numpy as np
import glob
import random
from time import time
from sklearn import svm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt

from functions import load_images_from_folder
from functions import load_classes_from_folder
from functions import compute_pca
from functions import unison_shuffled_copies
from functions import pred_approx_val
from functions import perf_measure
from functions import calc_far_frr
from functions import prepare_graph_far_frr
from functions import plot_far_frr
from functions import compute_roc
from functions import plot_roc

parser = argparse.ArgumentParser(description='Calculating biometric system FAR, FRR and ROC')
parser.add_argument('--test_size', type=float, default=0.3,
                    help='Choose the test size in 0-1 float range (default: 0.3)')
parser.add_argument('--cross_val', type=int, default=4,
                    help='Choose the cross validation k-folds, in int (default: 4)')

args = parser.parse_args()

# Global variables
test_percent = args.test_size
folder_name = "./faces"
cross_validation = args.cross_val

# Load data from folder
imageDir = folder_name
X, names = load_images_from_folder(imageDir)
X = np.array(X)
n_samples, h, w = X.shape
y = load_classes_from_folder(imageDir, names)
y = np.array(y)
n_classes = y.shape[1]

# Shuffle array
X, y = unison_shuffled_copies(X, y)

# Change to lilnear vectors
linearData=[]
for i in range(0, len(X)):
    linearData.append(X[i].ravel())

linearData = np.array(linearData)
X = linearData

# Shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent,
                                                    random_state=0)
n_features = X.shape[1]

print("--------------------")
print("Total dataset size:")
print("Number of samples in dataset: %d" % n_samples)
print("Number of featurs (dimension of images): %d" % n_features)
print("Number of classes in dataset: %d" % n_classes)
print("--------------------")
print("Total train size: %d samples, %d%%" % (len(X_train), (1 - test_percent) * 100))
print("Total test size: %d samples, %d%%" % (len(X_test), test_percent * 100))
print("Cross validation %d-fold" % cross_validation)
print("--------------------")

if cross_validation < 2:
    print "ERROR: cross-validation argument must be greater than 1"
    quit()

print "Computing PCA and Linear regression"
t0 = time()
model = compute_pca(X_train, y_train, LinearRegression())
print("done in %0.3fs" % (time() - t0))

print "Computing prediction with cross validation"
t0 = time()
y_score = cross_val_predict(model, X_test, y_test, cv=cross_validation) # make n-fold cross validation
y_score = y_score.clip(0) # clipping negative numbers to zero
test_size = len(y_score)
total_comparisions = test_size * n_classes;
print("done in %0.3fs" % (time() - t0))

# Compute FAR and FRR for graph
print "Computing FAR and FRR for graph"
t0 = time()
far, frr = prepare_graph_far_frr(y_test, y_score, total_comparisions - test_size, test_size)
print("done in %0.3fs" % (time() - t0))

# Compute ROC curve and ROC area for each class
print "Computing ROC curve for graph"
t0 = time()
tpr, fpr, roc_auc = compute_roc(n_classes, y_score, y_test)
print("done in %0.3fs" % (time() - t0))

print "Plotting graphs..."

# Plot FAR and FRR
plot_far_frr(far, frr)

# Plot ROC
plot_roc(tpr, fpr, roc_auc)
plt.show()