import numpy as np

def sample_fixed_class_prior(X_train, y_train, X_test, y_test, p1_train=0.1, p1_test=0.5, random_state=42):
    """
    Modifies train and test sets to enforce given class priors.

    Parameters:
      X_train : array-like or DataFrame
          Training feature matrix.
      y_train : array-like or Series
          Training target vector.
      X_test : array-like or DataFrame
          Testing feature matrix.
      y_test : array-like or Series
          Testing target vector.
      p1_train : float, optional (default=0.1)
          Desired proportion of the positive class (y==1) in the training set.
      p1_test : float, optional (default=0.5)
          Desired proportion of the positive class (y==1) in the test set.
      random_state : int, optional (default=42)
          Random seed for reproducibility.

    Returns:
      X_train_samp, y_train_samp, X_test_samp, y_test_samp : Modified train and test datasets.
    """
    np.random.seed(random_state)

    # Function to sample with fixed class prior
    def resample_fixed_prior(X_set, y_set, p1_target):
        # Converting y_set to numpy array if it's a pandas object
        if hasattr(y_set, "to_numpy"):
            y_array = y_set.to_numpy()
        else:
            y_array = np.array(y_set)

        pos_idx = np.where(y_array == 1)[0]
        neg_idx = np.where(y_array == 0)[0]

        # Computing required number of positive and negative samples
        total_samples = len(y_array)
        target_pos_samples = int(total_samples * p1_target)
        target_neg_samples = total_samples - target_pos_samples

        # Sampling positives and negatives accordingly
        if len(pos_idx) > target_pos_samples:
            sampled_pos_idx = np.random.choice(pos_idx, size=target_pos_samples, replace=False)
        else:
            sampled_pos_idx = pos_idx  # Use all if not enough

        if len(neg_idx) > target_neg_samples:
            sampled_neg_idx = np.random.choice(neg_idx, size=target_neg_samples, replace=False)
        else:
            sampled_neg_idx = neg_idx  # Use all if not enough

        # Combining and sort indices
        sel = np.sort(np.concatenate((sampled_pos_idx, sampled_neg_idx)))

        # Handling X_set differently depending on whether it's Pandas or NumPy
        if hasattr(X_set, "iloc"):
            # Pandas DataFrame or Series
            X_resampled = X_set.iloc[sel, :].copy()
        else:
            # NumPy array
            X_resampled = X_set[sel, :].copy()

        # Handling y_set similarly
        if hasattr(y_set, "iloc"):
            y_resampled = y_set.iloc[sel].copy()
        else:
            y_resampled = y_array[sel].copy()

        return X_resampled, y_resampled

    # Modifying class distributions in both train and test
    X_train_samp, y_train_samp = resample_fixed_prior(X_train, y_train, p1_train)
    X_test_samp, y_test_samp = resample_fixed_prior(X_test, y_test, p1_test)

    return X_train_samp, y_train_samp, X_test_samp, y_test_samp
