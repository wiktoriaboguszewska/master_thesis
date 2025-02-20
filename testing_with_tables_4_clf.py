import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import datasets
from method1_dnn_dla_2_klas import method1_ks_test_dnn
from method2_dnn_dla_2_klas import method2_em_kl_dnn
from creating_distribution_shift import sample_fixed_class_prior


# ----------------------------------------------------------------
# Modified experiment functions that loop over classifiers
# ----------------------------------------------------------------
def run_experiment_no_shift(X, y,
                            methods=['method1', 'method2'],
                            classifiers=['logreg', 'dnn', 'rf', 'svc', 'knn', 'gb'],
                            n_trials=100,
                            output_file_prefix="no_shift_results"):
    """
    Run experiments without distribution shift over n_trials.
    For each trial, and for each combination of method and classifier,
    record the p-value and running time.
    Results are saved in files with names using the given prefix.
    """
    results = {}
    times = {}
    # Initialize dictionary keys for each combination: e.g. "method1_logreg"
    for clf in classifiers:
        for method in methods:
            key = f"{method}_{clf}"
            results[key] = []
            times[key] = []

    for trial in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=trial
        )

        for clf in classifiers:
            for method in methods:
                start_time = time.time()
                if method == 'method1':
                    # Call method1 with current classifier
                    p_value = method1_ks_test_dnn(X_train, y_train, X_test, clf_name=clf)
                elif method == 'method2':
                    # Call method2 with current classifier
                    p_value = method2_em_kl_dnn(X_train, y_train, X_test, clf_name=clf)
                elapsed_time = time.time() - start_time
                key = f"{method}_{clf}"
                results[key].append(p_value)
                times[key].append(elapsed_time)

    # Save results DataFrames to files
    df_results = pd.DataFrame(results)
    df_times = pd.DataFrame(times)
    results_file = f"{output_file_prefix}.txt"
    times_file = f"{output_file_prefix}_times.txt"
    df_results.to_csv(results_file, sep="\t", index=False)
    df_times.to_csv(times_file, sep="\t", index=False)
    print(f"Results saved to files: {results_file} and {times_file}")


def run_experiment_with_shift(X, y,
                              p_shift=0.1,
                              methods=['method1', 'method2'],
                              classifiers=['logreg', 'dnn', 'rf', 'svc', 'knn', 'gb'],
                              n_trials=100,
                              output_file_prefix="shift_results"):
    """
    Run experiments with a distribution shift (p1 = p_shift) over n_trials.
    For each trial, sample a shifted training set and then loop over
    all classifier and method combinations.
    """
    results = {}
    times = {}
    for clf in classifiers:
        for method in methods:
            key = f"{method}_{clf}"
            results[key] = []
            times[key] = []

    for trial in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=trial
        )
        # Create a shifted training set
        X_train_shifted, y_train_shifted = sample_fixed_class_prior(X_train, y_train, p1=p_shift)

        for clf in classifiers:
            for method in methods:
                start_time = time.time()
                if method == 'method1':
                    p_value = method1_ks_test_dnn(X_train_shifted, y_train_shifted, X_test, clf_name=clf)
                elif method == 'method2':
                    p_value = method2_em_kl_dnn(X_train_shifted, y_train_shifted, X_test, clf_name=clf)
                elapsed_time = time.time() - start_time
                key = f"{method}_{clf}"
                results[key].append(p_value)
                times[key].append(elapsed_time)

    df_results = pd.DataFrame(results)
    df_times = pd.DataFrame(times)
    results_file = f"{output_file_prefix}.txt"
    times_file = f"{output_file_prefix}_times.txt"
    df_results.to_csv(results_file, sep="\t", index=False)
    df_times.to_csv(times_file, sep="\t", index=False)
    print(f"Results saved to files: {results_file} and {times_file}")


# ----------------------------------------------------------------
# Modified summary function: overall and per classifier final results
# ----------------------------------------------------------------
def compute_errors_and_power(file_no_shift="no_shift_results.txt",
                             file_shift="shift_results.txt",
                             file_no_shift_time="no_shift_results_times.txt",
                             file_shift_time="shift_results_times.txt",
                             alpha=0.05):
    """
    Reads the saved no-shift and shift result files and computes:
      - Type I error (false positive rate) using the no-shift p-values,
      - Type II error (false negative rate) and power (true positive rate)
        using the shift p-values,
      - Average running time.

    The results are computed for each (method, classifier) combination.
    Finally, one overall result file is saved, and separate result files
    are saved for each classifier.
    """
    df_no_shift = pd.read_csv(file_no_shift, sep="\t")
    df_shift = pd.read_csv(file_shift, sep="\t")
    df_no_shift_time = pd.read_csv(file_no_shift_time, sep="\t")
    df_shift_time = pd.read_csv(file_shift_time, sep="\t")

    overall_results = {}
    # Calculate metrics for each column (each combination, e.g. "method1_logreg")
    for col in df_no_shift.columns:
        error_I = np.mean(df_no_shift[col] < alpha)  # false positive rate
        error_II = np.mean(df_shift[col] >= alpha)  # false negative rate
        power = np.mean(df_shift[col] < alpha)  # detection power
        avg_time = (df_no_shift_time[col].mean() + df_shift_time[col].mean()) / 2
        overall_results[col] = {
            "error_I": error_I,
            "error_II": error_II,
            "power": power,
            "avg_time": avg_time
        }

    # Save overall results to one file
    df_overall = pd.DataFrame(overall_results).T
    overall_filename = "final_results_overall.txt"
    df_overall.to_csv(overall_filename, sep="\t")
    print(f"Overall final results saved to {overall_filename}")

    # Now, split the results by classifier.
    # Assuming column names are in the format "method_classifier"
    classifier_results = {}
    for col in df_no_shift.columns:
        try:
            method, clf = col.split('_')
        except ValueError:
            # In case the column naming is unexpected
            continue
        if clf not in classifier_results:
            classifier_results[clf] = {}
        classifier_results[clf][col] = overall_results[col]

    # Save separate files for each classifier
    for clf, res in classifier_results.items():
        df_clf = pd.DataFrame(res).T
        clf_filename = f"final_results_{clf}.txt"
        df_clf.to_csv(clf_filename, sep="\t")
        print(f"Final results for classifier '{clf}' saved to {clf_filename}")


# ----------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Ładujemy przykładowe dane
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Uruchamiamy eksperyment bez przesunięcia rozkładu
    run_experiment_no_shift(
        X, y,
        classifiers=['knn', 'rf', 'gb', 'dnn'],  # <--- tylko te klasyfikatory
        n_trials=100,
        output_file_prefix="no_shift_results"
    )

    # Uruchamiamy eksperyment z przesunięciem rozkładu
    run_experiment_with_shift(
        X, y,
        p_shift=0.1,
        classifiers=['knn', 'rf', 'gb', 'dnn'],  # <--- tylko te klasyfikatory
        n_trials=100,
        output_file_prefix="shift_results"
    )

    # Obliczamy i zapisujemy wyniki końcowe
    compute_errors_and_power(
        file_no_shift="no_shift_results.txt",
        file_shift="shift_results.txt",
        file_no_shift_time="no_shift_results_times.txt",
        file_shift_time="shift_results_times.txt",
        alpha=0.05
    )

