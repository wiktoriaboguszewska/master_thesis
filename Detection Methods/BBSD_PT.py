import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from dnn import dnn
from proportion_test import run_proportion_test

def BBSD_PT(X_train, y_train, X_test, clf_name='logreg', alpha=0.05):
    """
    Proportion Test for distribution shift with many possible classificators.

    :param X_train: Training set features.
    :param y_train: Training set labels.
    :param X_test: Test set features.
    :param clf_name: Classifier name ('logreg', 'dnn', 'rf', 'svc', 'gb').
    :param alpha: Significance level for hypothesis testing.
    :return: p_value from the two-sample proportion test.
    """
    # Splitting training set into 3 subsets: S1 and S2
    X_S1, X_S2, y_S1, y_S2 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Selecting classifier
    if clf_name == 'logreg':
        model = LogisticRegression(solver='liblinear')
    elif clf_name == 'dnn':
        model = dnn()
    elif clf_name == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif clf_name == 'svc':
        model = SVC(probability=True, random_state=42)
    elif clf_name == 'gb':
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported classifier: {clf_name}")

    # Training the model on S1
    model.fit(X_S1, y_S1)

    # Prediction of probabilities for S2 and for X_test
    preds_S2 = model.predict_proba(X_S2)[:, 1]
    preds_test = model.predict_proba(X_test)[:, 1]

    # Predicion vectors (binary)
    y_pred_S2 = (preds_S2 > 0.5).astype(int)
    y_pred_test = (preds_test > 0.5).astype(int)

    # Computing proportions
    p_hat = np.mean(y_pred_S2)
    q_hat = np.mean(y_pred_test)

    # Computing test statistic for two-sample proportion test
    n_s = len(y_pred_S2)
    n_t = len(y_pred_test)

    # Running the proportion test
    z_stat, p_value = run_proportion_test(p_hat, q_hat, n_s, n_t, alpha=alpha)

    print(f"Z-statistic: {z_stat:.4f}, P-value: {p_value:.4f}")
    if p_value < alpha:
        print("Reject H0: Distribution shift detected.")
    else:
        print("No reason to reject H0: No significant shift detected.")

    return p_value