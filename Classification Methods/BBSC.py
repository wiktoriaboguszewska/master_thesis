import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from dnn import dnn

def bbsc(X_train, Y_train, X_test, clf_name='logreg'):
    """
    Black-Box Shift Correction (Saerens-Latinne re-weighting)

    Parameters
    ----------
    X_train : (n_train, d) ndarray
    Y_train : (n_train,) ndarray  {0,1}
    X_test  : (n_test , d) ndarray
    clf_name: ''  -> LogisticRegression
              'dnn' -> simple feed-forward net `dnn`
              'rf'  -> RandomForestClassifier

    Returns
    -------
    q_x : (n_test, 2) ndarray – corrected posteriors P*(y|x) for test set
    """

    # ------------------------------------------------------------------
    # 1) fit an *unweighted* probabilistic classifier on the training set
    # ------------------------------------------------------------------
    if clf_name == 'logreg':
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        )
        base_clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
        base_clf.fit(X_train, Y_train)
        yhat_train = base_clf.predict_proba(X_train)[:, 1]
        yhat_test  = base_clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'dnn':
        base_clf = dnn()
        base_clf.fit(X_train, Y_train, weights0=[0])
        yhat_train = base_clf.predict_proba(X_train)[:, 1]
        yhat_test  = base_clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'rf':
        base = RandomForestClassifier(n_estimators=100, random_state=42)
        base_clf = CalibratedClassifierCV(base, method='isotonic', cv=5)
        base_clf.fit(X_train, Y_train)
        yhat_train = base_clf.predict_proba(X_train)[:, 1]
        yhat_test  = base_clf.predict_proba(X_test)[:, 1]

    else:
        raise ValueError("clf_name must be 'logreg', 'dnn', or 'rf'")

    # ------------------------------------------------------------------
    # 2) Black-box shift estimation (Saerens-Latinne-Decaestecker)
    #    Compute A,B (mean scores conditioned on true label)
    #    Estimate new prior q1 on test set from average score Q
    # ------------------------------------------------------------------
    A  = np.mean(yhat_train[Y_train == 1])  # E[p|y=1]
    B  = np.mean(yhat_train[Y_train == 0])  # E[p|y=0]
    Q  = np.mean(yhat_test)                 # average on *test*

    eps = 1e-4
    q1 = (Q - B) / (A - B + eps)            # new prior P*(y=1)
    q1 = np.clip(q1, 0.0, 1.0)              # guard numerics
    q0 = 1.0 - q1

    p1 = np.mean(Y_train)                   # training prior
    p0 = 1.0 - p1
    if p1 == 0: p1 = eps
    if p0 == 0: p0 = eps

    w1 = q1 / p1                            # sample-weight multiplier
    w0 = q0 / p0

    # ------------------------------------------------------------------
    # 3) Build per-example weights and train *weighted* classifier
    # ------------------------------------------------------------------
    weights = np.where(Y_train == 1, w1, w0)

    if clf_name == 'logreg':
        final = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        )
        final_clf = CalibratedClassifierCV(final, method='isotonic', cv=5)
        final_clf.fit(X_train, Y_train, sample_weight=weights)
        q_x = final_clf.predict_proba(X_test)          # (n_test, 2)

    elif clf_name == 'dnn':
        final_clf = dnn()
        final_clf.fit(X_train, Y_train, weights0=weights)
        q_x = final_clf.predict_proba(X_test)

    elif clf_name == 'rf':
        final = RandomForestClassifier(n_estimators=100, random_state=0)
        final_clf = CalibratedClassifierCV(final, method='isotonic', cv=5)
        final_clf.fit(X_train, Y_train, sample_weight=weights)
        q_x = final_clf.predict_proba(X_test)

    else:
        # should never happen – guarded above
        raise RuntimeError("Unknown clf_name")

    return q_x
