import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from dnn_new import dnn

def method1_ks_test_dnn(X_train, y_train, X_test, clf_name='logreg'):
    """
    Test Kolmogorova-Smirnova dla przesunięcia rozkładu z obsługą wielu klasyfikatorów.

    :param X_train: Dane cech zbioru treningowego.
    :param y_train: Etykiety zbioru treningowego.
    :param X_test: Dane cech zbioru testowego.
    :param clf_name: Nazwa klasyfikatora ('logreg', 'dnn', 'rf', 'svc', 'knn', 'gb').
    """
    # Podział zbioru treningowego na dwa podzbiory: S1 i S2
    X_S1, X_S2, y_S1, y_S2 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Wybór klasyfikatora
    if clf_name == 'logreg':
        model = LogisticRegression(solver='liblinear')
    elif clf_name == 'dnn':
        model = dnn()
    elif clf_name == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif clf_name == 'svc':
        model = SVC(probability=True, random_state=42)
    elif clf_name == 'knn':
        model = KNeighborsClassifier()
    elif clf_name == 'gb':
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Nieobsługiwany klasyfikator: {clf_name}")

    # Trenowanie modelu
    model.fit(X_S1, y_S1)

    # Predykcja prawdopodobieństw dla S2 i X_test
    preds_S2 = model.predict_proba(X_S2)[:, 1]
    preds_test = model.predict_proba(X_test)[:, 1]

    # Test statystyczny KS między predykcjami dla S2 i X_test
    ks_stat, p_value = ks_2samp(preds_S2, preds_test)
    print(f"Statystyka KS: {ks_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Odrzucamy hipotezę H0: Wykryto przesunięcie rozkładu.")
    else:
        print("Brak podstaw do odrzucenia H0: Rozkłady są zgodne.")

    return p_value


