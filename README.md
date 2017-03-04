# Biometric System Classification using PCA
Biometric system classification using PCA and computing its FAR (False Accept Rate), FRR (False Reject Rate) and ROC (Receiver Operating Characteristics) using scikit-learn and Python.

- This project was made as a school assignment.
- The project uses pre-processed dataset of images in 16x15px dimension in folder /faces

Required install:
```
python
scikit-learn
python-matplotlib
```

To run with defaults 4-fold cross validation and 30% test data

```
python main.py
```

Arguments

Choose the test size in 0-1 float range (default: 0.3)

--test_size TEST_SIZE

Choose the cross validation k-folds, in int (default:4)

--cross_val CROSS_VAL

The result of FAR, FRR and ROC are two plots, one is showing FAR and FRR and the other one depicts ROC
