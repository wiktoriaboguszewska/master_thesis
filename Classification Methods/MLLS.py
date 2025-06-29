import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from dnn import dnn

def em(X_train, Y_train, X_test, clf_name='logreg', epochs=200, q_init=0.5, tol=None):
    """
    Expectation–Maximisation for label-shift **with binary classes**.

    Parameters
    ----------
    X_train : (n_train, d) ndarray
    Y_train : (n_train,) ndarray  {0,1}
    X_test  : (n_test , d) ndarray   (unlabelled)
    clf_name: '' | 'dnn' | 'rf'
    epochs  : int   – max EM iterations (ignored if tol is supplied)
    q_init  : float – initial test prior
    tol     : float or None – stop if |q_new - q_old| < tol

    Returns
    -------
    q_post  : (n_test, 2) ndarray – corrected posteriors on X_test
    q_prior : float              – estimated test prior P*(y=1)
    """

    # ------------------------------------------------------------------
    # 1) fit a *probabilistic* classifier on training data
    # ------------------------------------------------------------------
    if clf_name == 'logreg':
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        )
        model = CalibratedClassifierCV(base, method='isotonic', cv=5)
        model.fit(X_train, Y_train)
        p_1_x = model.predict_proba(X_test)[:, 1]   # p(y=1 | x) on test

    elif clf_name == 'dnn':
        net = dnn()
        net.fit(X_train, Y_train, weights0=[0])      # mimic original API
        p_1_x = net.predict_proba(X_test)[:, 1]

    elif clf_name == 'rf':
        base = RandomForestClassifier(n_estimators=100, random_state=42)
        model = CalibratedClassifierCV(base, method='isotonic', cv=5)
        model.fit(X_train, Y_train)
        p_1_x = model.predict_proba(X_test)[:, 1]

    else:
        raise ValueError("clf_name must be 'logreg' or 'dnn' or 'rf'")

    # empirical training prior
    p_1 = np.mean(Y_train)

    # ------------------------------------------------------------------
    # 2) EM iterations
    # ------------------------------------------------------------------
    q_1 = q_init
    for epoch in tqdm(range(epochs), disable=(epochs==1)):
        # ---- E-step ----
        if epoch == 0:
            q_1_x = p_1_x.copy()     # first iteration: use classifier outputs
        else:
            num   = p_1_x * (q_1 / p_1)
            denom = num + (1 - p_1_x) * ((1 - q_1) / (1 - p_1))
            q_1_x = num / denom      # corrected P*(y=1 | x)

        # ---- M-step ----
        q_new = np.mean(q_1_x)

        # convergence?
        if tol is not None and abs(q_new - q_1) < tol:
            q_1 = q_new
            break
        q_1 = q_new

    # ------------------------------------------------------------------
    # 3) pack output posteriors
    # ------------------------------------------------------------------
    q_post = np.empty((X_test.shape[0], 2), dtype=float)
    q_post[:, 1] = q_1_x
    q_post[:, 0] = 1.0 - q_1_x
    return q_post, q_1
