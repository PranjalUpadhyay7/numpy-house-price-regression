"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import pandas as pd
def impute_nan_with_mean(X):

    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X=pd.DataFrame(X)
    X=X.fillna(X.mean()).fillna(0)
    X=X.to_numpy()
    return X
    # TODO: Replace every NaN with that column's nan-aware mean...
    # pass

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1= np.percentile(X,25,axis=0)
    q3= np.percentile(X,75,axis=0)
    IQR= q3-q1
    lower= q1-k*IQR
    upper=q3+ k*IQR
    return [lower, upper]
    # pass

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    # pass
    return np.clip(X,lower,upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    # pass
    return numerator/(denominator +eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    # pass
    return np.column_stack([X,col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    # pass
    label = np.asarray(labels)
    c=np.unique(label)
    oneHot=label[:,None] ==c[None,:]
    return oneHot.astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    # pass
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    std=np.where(std==0,1.0,std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    # pass
    return (X-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    # pass
    zero=np.ones([X.shape[0],1])
    return np.column_stack([zero,X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    # pass
    index=np.arange(n_samples)
    return np.random.RandomState(seed).permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    # pass
    N=len(indices)
    train=int(N*train_ratio)
    val=train + int(N*val_ratio)
    return [indices[:train], indices[train:val], indices[val:]]

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    # pass
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    # pass

    A= np.matmul(X.T,X)
    b=np.matmul(X.T,y)
    return np.linalg.solve(A,b)

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    # pass
    if len(X.shape)==2 and X.shape[1]== len(theta):
        return np.matmul(X,theta)

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    # pass
    return np.mean(np.abs(y_true-y_pred))

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
    # pass
    return np.sqrt(np.mean(np.square(y_pred-y_true)))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    # pass
    ss_res=np.sum(np.square(y_true-y_pred))
    mean=np.mean(y_true)
    
    ss_tot=np.sum(np.square(y_true-mean))
    if ss_tot==0:
        return 0.0
    return 1-(ss_res/ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    # pass
    r=y_true-y_pred
    return {"mean":np.mean(r),
    "std": np.std(r),
    "median_abs": np.median(np.abs(r))}

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
    # pass
    X=impute_nan_with_mean(X)
    lower,upper=compute_iqr_bounds(X,k=iqr_k)
    return clip_columns(X,lower,upper)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    # pass
    numerator= X_num[:,ratio_num_idx]
    denominator=X_num[:,ratio_den_idx]
    ratio=make_ratio_feature(numerator,denominator)
    X_num=append_column(X_num,ratio)
    if cat_labels is not None:
        cat=one_hot_encode(cat_labels)
        X_num=np.hstack((X_num,cat))
    return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    # pass
    n=len(X)
    index=make_shuffled_indices(n,seed)
    part_ind=partition_indices(index,train_ratio,val_ratio)
    X_train,y_train=subset_xy(X,y,part_ind[0])
    X_val,y_val=subset_xy(X,y,part_ind[1])
    X_test,y_test=subset_xy(X,y,part_ind[2])
    return {'X_train':X_train,'y_train': y_train,'X_val':X_val,'y_val':y_val,"X_test":X_test,"y_test":y_test}

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

