"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    mean = np.nanmean(X,axis=0)
    return np.where(np.isnan(X), np.where(np.isnan(mean),0,mean),X)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X,25,axis=0)
    q3 = np.percentile(X,75,axis=0)
    iqr = q3-q1
    lower = q1-k*iqr
    upper = q3+k*iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X_clipped = np.clip(X,lower ,upper)
    return X_clipped

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator/(denominator+eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X = np.hstack((X,np.expand_dims(col,axis=1)))
    return X

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    return (labels[:,None]==np.unique(labels)).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    std = np.where(std==0,1.0,std)
    return mean,std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    return np.hstack((np.ones((X.shape[0],1)),X))

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed)
    return rng.permutation(range(n_samples))

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = indices.shape[0]
    train_ratio,val_ratio = int(train_ratio*n),int(val_ratio*n)
    return indices[:train_ratio],indices[train_ratio:train_ratio+val_ratio],indices[train_ratio+val_ratio:]

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices],y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(A,b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X@theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_pred-y_true))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(np.mean((y_pred-y_true)**2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    if ss_tot == 0:
        return 0
    return 1 - (np.sum((y_pred-y_true)**2)/ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = (y_true - y_pred).astype(np.float64)
    ans = {}
    ans["mean"] = np.mean(r)
    ans["std"] = np.std(r)
    ans["median_abs"] = np.median(np.abs(r))
    return ans

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    mean = np.nanmean(X,axis=0)
    mean = np.where(np.isnan(mean),0,mean)
    X = np.where(np.isnan(X),mean,X)

    q1 = np.percentile(X,25,axis=0)
    q3 = np.percentile(X,75,axis=0)
    iqr = q3-q1

    lower = q1-iqr_k*iqr
    upper = q3+iqr_k*iqr
    X = np.clip(X,lower,upper)
    return X

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    ratio = X_num[:,ratio_num_idx]/X_num[:,ratio_den_idx]
    ans = np.hstack((X_num,np.expand_dims(ratio,axis=1)))
    if cat_labels is not None:
        labels = np.asarray(cat_labels)
        ans = np.hstack((ans,(labels[:,None]==np.unique(labels)).astype(float)))
    return ans

# Step 21 - make_train_val_test
import numpy as np

def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    samples = X.shape[0]
    
    # Use legacy random seed and permutation to match expected test outputs
    np.random.seed(seed)
    indices = np.random.permutation(samples)
    
    X = X[indices]
    y = y[indices]
    
    n_train = int(train_ratio * samples)
    n_val = int(val_ratio * samples)
    
    return {
        "X_train": X[:n_train],
        "X_val":   X[n_train : n_train + n_val],
        "X_test":  X[n_train + n_val :],
        "y_train": y[:n_train],
        "y_val":   y[n_train : n_train + n_val],
        "y_test":  y[n_train + n_val :],
    }

# Step 22 - standardize_and_add_bias
import numpy as np

def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepends a column of ones to a feature matrix."""
    return np.c_[np.ones(X.shape[0]), X]

def standardize_and_add_bias(splits: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Fits standardizer on train set, normalizes all splits, and prepends bias column.
    """
    X_train = splits["X_train"]
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # ONLY replace 0-std with 1 to avoid division by zero.
    # Do NOT use np.maximum(std, 1.0), as it alters valid std values < 1.0.
    std = np.where(std == 0, 1.0, std)
    
    std_splits = {}
    for key, val in splits.items():
        if key.startswith("X_"):
            std_splits[key] = add_bias((val - mean) / std)
        else:
            std_splits[key] = val
            
    return std_splits, mean, std

# Step 23 - evaluate_predictions
import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepends a column of ones to a feature matrix."""
    return np.c_[np.ones(X.shape[0]), X]


def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    ans = {}
    error = y_true-y_pred
    mean = np.mean(y_true)
    ans["mae"] = np.mean(np.abs(error))
    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true-mean)**2)
    ans["r2"] = 1 - (ss_res/ss_tot if ss_tot != 0 else 1)
    ans["residual_summary"] = {
        "mean": np.mean(error),
        "std": np.std(error),
        "median_abs": np.median(np.abs(error))
    }
    ans["rmse"] = np.sqrt(np.mean((error**2)))
    return ans


def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...

    # CLEAN THE DATA
    X_mean = np.nan_to_num(np.nanmean(X, axis=0))
    X = np.where(np.isnan(X), X_mean, X)
    y_mean = np.nan_to_num(np.nanmean(y))
    y = np.where(np.isnan(y), y_mean, y)

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3-q1
    lower = q1-iqr_k*iqr
    upper = q3+iqr_k*iqr
    X = np.clip(X, lower, upper)

    # FEATURIZE
    ratio = X[:, ratio_num_idx]/X[:, ratio_den_idx]
    X = np.hstack((X, ratio.reshape(-1, 1)))
    if cat_labels is not None:
        ohe = (cat_labels[:, None] == np.unique(cat_labels)).astype(float)
        X = np.hstack((X, ohe))

    # SPLITTING DATAS
    no_samples = X.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(no_samples)

    X = X[indices]
    y = y[indices]

    n_train = int(train_ratio * no_samples)
    n_val = int(val_ratio * no_samples)

    splits = {
        "X_train": X[:n_train],
        "X_val":   X[n_train: n_train + n_val],
        "X_test":  X[n_train + n_val:],
        "y_train": y[:n_train],
        "y_val":   y[n_train: n_train + n_val],
        "y_test":  y[n_train + n_val:],
    }

    X_train = splits["X_train"]
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train_std = np.where(X_train_std == 0.0, 1.0, X_train_std)

    std_splits = {}
    for key, val in splits.items():
        if key.startswith("X_"):
            std_splits[key] = add_bias((val - X_train_mean) / X_train_std)
        else:
            std_splits[key] = val

    # FIT AND PREDICT
    X_train = std_splits["X_train"]
    y_train = std_splits["y_train"]
    A = X_train.T@X_train
    b = X_train.T@y_train
    weights = np.linalg.lstsq(A, b, rcond=None)[0]

    X_test = std_splits["X_test"]
    X_val = std_splits["X_val"]
    y_test = std_splits["y_test"]
    y_val = std_splits["y_val"]

    # EVALUATING THE PREDICTION
    y_val_pred = X_val @ weights
    y_test_pred = X_test @ weights

    val_metrics = evaluate_predictions(y_val, y_val_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    result = {"theta": weights, "y_test": y_test,
              "y_test_pred": y_test_pred, "val_metrics": val_metrics, "test_metrics": test_metrics}

    return result

# Step 24 - house_price_pipeline
import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepends a column of ones to a feature matrix."""
    return np.c_[np.ones(X.shape[0]), X]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates regression metrics given true and predicted values."""
    error = y_true - y_pred
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...

    # CLEAN THE DATA
    X_mean = np.nan_to_num(np.nanmean(X, axis=0))
    X = np.where(np.isnan(X), X_mean, X)
    y_mean = np.nan_to_num(np.nanmean(y))
    y = np.where(np.isnan(y), y_mean, y)

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3-q1
    lower = q1-iqr_k*iqr
    upper = q3+iqr_k*iqr
    X = np.clip(X, lower, upper)

    # FEATURIZE
    ratio = X[:, ratio_num_idx]/X[:, ratio_den_idx]
    X = np.hstack((X, ratio.reshape(-1, 1)))
    if cat_labels is not None:
        ohe = (cat_labels[:, None] == np.unique(cat_labels)).astype(float)
        X = np.hstack((X, ohe))

    # SPLITTING DATAS
    no_samples = X.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(no_samples)

    X = X[indices]
    y = y[indices]

    n_train = int(train_ratio * no_samples)
    n_val = int(val_ratio * no_samples)

    splits = {
        "X_train": X[:n_train],
        "X_val":   X[n_train: n_train + n_val],
        "X_test":  X[n_train + n_val:],
        "y_train": y[:n_train],
        "y_val":   y[n_train: n_train + n_val],
        "y_test":  y[n_train + n_val:],
    }

    X_train = splits["X_train"]
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train_std = np.where(X_train_std == 0.0, 1.0, X_train_std)

    std_splits = {}
    for key, val in splits.items():
        if key.startswith("X_"):
            std_splits[key] = add_bias((val - X_train_mean) / X_train_std)
        else:
            std_splits[key] = val

    # FIT AND PREDICT
    X_train = std_splits["X_train"]
    y_train = std_splits["y_train"]
    A = X_train.T@X_train
    b = X_train.T@y_train
    weights = np.linalg.lstsq(A, b, rcond=None)[0]

    X_test = std_splits["X_test"]
    X_val = std_splits["X_val"]
    y_test = std_splits["y_test"]
    y_val = std_splits["y_val"]

    # EVALUATING THE PREDICTION
    y_val_pred = X_val @ weights
    y_test_pred = X_test @ weights

    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    result = {"theta": weights, "y_test": y_test,
              "y_test_pred": y_test_pred, "val_metrics": val_metrics, "test_metrics": test_metrics}

    return result

