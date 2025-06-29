import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from dnn import dnn

def threshold_calibration(X_train, Y_train, X_test, clf_name='logreg'):
    """
    Threshold Calibration under Label Shift

    Parameters
    ----------
    X_train   : array-like, shape (n_train, d)
    Y_train   : array-like, shape (n_train,)   {0,1}
    X_test    : array-like, shape (n_test, d)  (unlabeled)
    clf_name  : 'logreg' | 'dnn' | 'rf'
                'logreg' -> LogisticRegression
                'dnn'    -> simple feed-forward net `dnn`
                'rf'     -> RandomForestClassifier

    Returns
    -------
    q_x      : ndarray, shape (n_test, 2)
               recalibrated posteriors q(y|x) on the test set
    q1_est   : float
               estimated target prior P*(y=1)
    threshold: float
               equivalent decision threshold on original p(y=1|x)
    """
    # 1) fitting an *unweighted* probabilistic classifier on the training set
    if clf_name == 'logreg':
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        )
        clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
        clf.fit(X_train, Y_train)
        p_train = clf.predict_proba(X_train)[:, 1]
        p_test  = clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'dnn':
        net = dnn()
        net.fit(X_train, Y_train, weights0=[0])
        p_train = net.predict_proba(X_train)[:, 1]
        p_test  = net.predict_proba(X_test)[:, 1]

    elif clf_name == 'rf':
        base = RandomForestClassifier(n_estimators=100, random_state=42)
        clf  = CalibratedClassifierCV(base, method='isotonic', cv=5)
        clf.fit(X_train, Y_train)
        p_train = clf.predict_proba(X_train)[:, 1]
        p_test  = clf.predict_proba(X_test)[:, 1]

    else:
        raise ValueError("clf_name must be 'logreg', 'dnn', or 'rf'")

    # 2) compute class-conditional means on train
    A = np.mean(p_train[Y_train == 1])  # E[p|y=1]
    B = np.mean(p_train[Y_train == 0])  # E[p|y=0]
    Q = np.mean(p_test)                # E_test[p]

    # 3) estimate new positive prior q1
    eps = 1e-8
    q1 = (Q - B) / (A - B + eps)
    q1 = np.clip(q1, 0.0, 1.0)

    # 4) recalibrate test probabilities via prior-ratio formula
    p1 = np.mean(Y_train)
    r1 = (q1 / p1) * p_test
    r0 = ((1 - q1) / (1 - p1)) * (1 - p_test)
    q1_x = r1 / (r1 + r0 + eps)

    # 5) assemble full posterior and compute threshold
    q_x = np.vstack([1 - q1_x, q1_x]).T

    thresh = ((1 - q1) * p1) / ((1 - p1) * q1 + (1 - q1) * p1 + eps)

    return q_x, q1, thresh



