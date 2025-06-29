from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from dnn import dnn

def BBSD_KS(X_train, y_train, X_test, clf_name='logreg', alpha=0.05):
    """
    Kolmogorov-Smirnov Test for distribution shift with many possible classificators.

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

    # Statistical test KS between predictions for S2 and for X_test
    ks_stat, p_value = ks_2samp(preds_S2, preds_test)
    print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < alpha:
        print("Reject H0: Distribution shift detected.")
    else:
        print("No reason to reject H0: No significant shift detected.")

    return p_value


