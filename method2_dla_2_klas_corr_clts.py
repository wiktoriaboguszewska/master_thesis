import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import ks_2samp
from random import choices
from scipy.stats import uniform, randint
import time
from dnn_new import dnn

##############################################################################
# (1) EM-algorithm do estymacji rozkładu q(y)
##############################################################################

def em_algorithm(posterior_probs, q_init, p_train, max_iter=100, tol=1e-5):
    """
    Algorytm EM do estymacji rozkładu q(y).

    :param posterior_probs: Prawdopodobieństwa posteriori p(y|x).
    :param q_init: Początkowy rozkład q(y).
    :param p_train: Rozkład klas w zbiorze treningowym.
    :param max_iter: Maksymalna liczba iteracji.
    :param tol: Tolerancja błędu do zatrzymania iteracji.
    :return: Estymowany rozkład q(y).
    """
    q = q_init.copy()
    eps = 1e-10     # Dodanie epsilona, żeby uniknąć dzielenia przez 0
    for _ in range(max_iter):
        # Krok E: Obliczenie q_t(y | x)
        q_y_given_x = (q[1] / (p_train[1] + eps)) * posterior_probs / (
                (q[1] / (p_train[1] + eps)) * posterior_probs + (q[0] / (p_train[0] + eps)) * (1 - posterior_probs) + eps
        )

        # Krok M: Aktualizacja q(y)
        q_new = np.array([1 - np.mean(q_y_given_x), np.mean(q_y_given_x)])

        # Sprawdzenie zbieżności
        if np.linalg.norm(q_new - q) < tol:
            break
        q = q_new
    return q

##############################################################################
# (2) Klasa do "temperature scaling" dla DNN
##############################################################################
class TemperatureScaler:
    """
    Zakładamy, że dnn.predict_logits(X) zwraca logity (np. log(p/(1-p)))
    lub że mamy sposób wyciągnięcia surowych wyjść, które można przeskalować.
    """
    def __init__(self):
        self.temperature_ = 1.0

    def fit(self, logits_val, y_val):
        """ Dopasowujemy parametr temperature t > 0, np. minimalizując logloss. """
        import numpy as np
        from scipy.optimize import minimize

        def objective_func(t):
            t = max(t, 1e-3)  # zabezp
            p = 1.0 / (1.0 + np.exp(-logits_val / t))
            eps = 1e-15
            p = np.clip(p, eps, 1 - eps)
            logloss = - np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
            return logloss

        res = minimize(objective_func, x0=self.temperature_, bounds=((1e-3, 100),))
        self.temperature_ = res.x[0]

    def predict_proba(self, logits):
        """Zwracamy [p0, p1] = [1 - p, p] po korekcie temperature."""
        p = 1.0 / (1.0 + np.exp(-logits / self.temperature_))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.column_stack([1 - p, p])

##############################################################################
# (3) Docelowa funkcja "method2_em_kl_dnn"
##############################################################################

def method2_em_kl_dnn(X_train, y_train, X_test, clf_name='logreg', alpha=0.05, n_resamples=2000, random_search_iter=10):
    """
    Implementacja metody 2 z algorytmem EM i statystyką KL z obsługą wielu klasyfikatorów.

    :param X_train: Dane cech zbioru treningowego.
    :param y_train: Etykiety zbioru treningowego.
    :param X_test: Dane cech zbioru testowego.
    :param clf_name: Nazwa klasyfikatora ('logreg', 'dnn', 'rf', 'svc', 'knn', 'gb').
    :param alpha: Poziom istotności dla testu hipotezy.
    :param n_resamples: Liczba próbek bootstrapowych.
    """
    # Podział zbioru treningowego na S1 i S2
    X_S1, X_S2, y_S1, y_S2 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Rozkład p_train
    p_train = np.bincount(y_S1) / len(y_S1)

    ############################################################################
    # Wybór i trenowanie klasyfikatora + KALIBRACJA
    ############################################################################
    if clf_name == 'logreg':
        base_clf = LogisticRegression(solver='liblinear')
        clf = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')
        # Trenowanie klasyfikatora na zbiorze S1
        clf.fit(X_S1, y_S1)
        # Posterior test
        posterior_probs_test = clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'knn':
        base_clf = KNeighborsClassifier()
        clf = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')
        clf.fit(X_S1, y_S1)
        posterior_probs_test = clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'rf':
        base_clf = RandomForestClassifier(random_state=42)
        param_dist = {
            'base_estimator__n_estimators': randint(50, 301),
            'base_estimator__max_depth': [3, 6, 10, None],
            'method': ['sigmoid', 'isotonic'],
        }
        # Tworzymy CalibratedClassifierCV (z defaultowymi parametrami)
        cal_clf = CalibratedClassifierCV(base_clf, cv=3)
        rnd = RandomizedSearchCV(
            estimator=cal_clf,
            param_distributions=param_dist,
            n_iter=random_search_iter,
            cv=3,
            scoring='roc_auc',
            random_state=42
        )
        rnd.fit(X_S1, y_S1)
        clf = rnd.best_estimator_
        print("RF best params:", rnd.best_params_)
        # Posterior
        posterior_probs_test = clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'gb':
        base_clf = GradientBoostingClassifier(random_state=42)
        param_dist = {
            'base_estimator__n_estimators': randint(50, 301),
            'base_estimator__max_depth': [3, 6, 10],
            'method': ['sigmoid', 'isotonic']
        }
        cal_clf = CalibratedClassifierCV(base_clf, cv=3)
        rnd = RandomizedSearchCV(
            estimator=cal_clf,
            param_distributions=param_dist,
            n_iter=random_search_iter,
            cv=3,
            scoring='roc_auc',
            random_state=42
        )
        rnd.fit(X_S1, y_S1)
        clf = rnd.best_estimator_
        print("GB best params:", rnd.best_params_)
        posterior_probs_test = clf.predict_proba(X_test)[:, 1]

    elif clf_name == 'dnn':
        dnn_model = dnn()
        dnn_model.fit(X_S1, y_S1)
        # dopasowujemy temperature scaling na S2
        logits_s2 = dnn_model.predict_logits(X_S2)
        ts = TemperatureScaler()
        ts.fit(logits_s2, y_S2)
        # przetwarzamy X_test
        logits_test = dnn_model.predict_logits(X_test)
        posterior_probs_test = ts.predict_proba(logits_test)[:, 1]

    else:
        raise ValueError(f"Nieobsługiwany klasyfikator: {clf_name}")

    # Inicjalizacja rozkładu q(y)
    q_init = np.array([0.5, 0.5])

    # Uruchomienie algorytmu EM na zbiorze S2
    q_new = em_algorithm(posterior_probs_test, q_init, p_train)
    print("Nowe estymowane q(y):", q_new)

    # Statystyka KL
    eps = 1e-10
    kl_div = np.sum(p_train * np.log((p_train + eps) / (q_new + eps)))
    print(f"Statystyka KL: {kl_div:.4f}")

    # Bootstrapowanie na S2
    kl_values = []
    # do ponownego uzyskania posterioru na S2:
    # bo oryg. kod brał X_S2, liczył posterior i bootstrappował
    # lepiej jest zrobić once -> posterior_probs_S2 i w EM
    #  zachować spójną metodę
    n_resamples = n_resamples if n_resamples else 1000
    X_S2_len = len(X_S2)
    # obliczamy once posterior S2
    if clf_name != 'dnn':
        # klascz
        # bierzemy ponownie clf i predict_proba
        # ALE musimy pamiętać, że S2 = X_S2, y_S2
        posterior_probs_s2 = clf.predict_proba(X_S2)[:, 1]
    else:
        # dnn
        logits_s2_again = dnn_model.predict_logits(X_S2)
        posterior_probs_s2 = ts.predict_proba(logits_s2_again)[:, 1]

    for _ in range(n_resamples):
        indices = choices(np.arange(X_S2_len), k=X_S2_len)
        # bootstrap
        X_S2_bootstrap = X_S2[indices, :]
        # musimy policzyć posterior bootstrap
        # albo prościej: weźmy posterior_probs_s2[indices]
        post_s2_boot = posterior_probs_s2[indices]
        q_bootstrap = em_algorithm(post_s2_boot, q_init, p_train)
        kl_bootstrap = np.sum(p_train * np.log((p_train + eps) / (q_bootstrap + eps)))
        kl_values.append(kl_bootstrap)

    # Obliczenie p-value
    kl_values = np.array(kl_values)
    p_value = np.mean(np.array(kl_values) >= kl_div)
    print(f"P-value: {p_value:.4f}")
    if p_value < alpha:
        print("Odrzucamy hipotezę H0: Wystąpiło przesunięcie rozkładu.")
    else:
        print("Brak podstaw do odrzucenia H0: Rozkład jest zgodny.")

    return p_value