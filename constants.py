regression_dict = dict()
regression_dict["SVR"] = "svr"
regression_dict["Gaussian Process Regressor"] = "gaussian_process_regressor"
regression_dict["Kernel Ridge"] = "kernel_ridge"
regression_dict["Random Forest Regressor"] = "random_forest_regressor"
regression_dict["Decision Tree Regressor"] = "decision_tree_regressor"
regression_dict["Linear Regressor"] = "linear_regressor"
regression_dict["Neural Network Regressor"] = "nnr"
regression_dict["QGPR: Quantum Gaussian Process Regressor"] = "qgpr"
regression_dict["QSVR: Quantum Support Vector Regressor"] = "qsvr"
regression_dict["QKRR: Quantum Kernel Ridge Regressor"] = "qkrr"
regression_dict["QNNR: Quantum Neural Network Regressor"] = "qnnr"
regression_dict["QRCR"] = "qrcr"


classification_dict = dict()
classification_dict["SVC"] = "svc"
classification_dict["Gaussian Process Classifier"] = "gaussian_process_classifier"
classification_dict["Ridge Classifier"] = "ridge_classifier"
classification_dict["Random Forest Classifier"] = "random_forest_classifier"
classification_dict["Decision Tree Classifier"] = "decision_tree_classifier"
classification_dict["Perceptron"] = "perceptron"
classification_dict["Logistic Regression Classifier"] = "logistic_regression_classifier"
classification_dict["QGPC: Quantum Gaussian Process Classifier"] = "qgpc"
classification_dict["QSVC: Quantum Support Vector Classifier"] = "qsvc"
classification_dict["QNNC: Quantum Neural Network Classifier"] = "qnnc"
classification_dict["QRCC"] = "qrcc"

rescaling_dict = dict()
rescaling_dict["Standard Scaling"] = "standard_scaling"
rescaling_dict["Normalization"] = "normalization"
rescaling_dict["MinMaxScaling"] = "min_max_scaling"
rescaling_dict["no Rescaling"] = "no-op"

encoding_dict = dict()
encoding_dict["Categorical"] = "categorical"
encoding_dict["One-Hot Encoding"] = "one-hot"
encoding_dict["no Encoding"] = "no-op"

downsampling_dict = dict()
downsampling_dict["Resampling"] = "resampling"
downsampling_dict["no Downsampling"] = "no-op"

dim_reduction_dict = dict()
dim_reduction_dict["PCA"] = "pca"
dim_reduction_dict["TSNE"] = "tsne"
dim_reduction_dict["UMAP"] = "umap"
dim_reduction_dict["Autoencoder"] = "autoencoder"
dim_reduction_dict["no Dimension Reduction"] = "no-op"

imputation_dict = dict()
imputation_dict["Constant"] = "constant"
imputation_dict["Mean"] = "mean"
imputation_dict["Drop"] = "drop"
imputation_dict["no Imputation"] = "no-op"