import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, matthews_corrcoef,
    precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Feature_Dataset import FEATURE_NAMES, MolecularDataset, check_split_overlap


TRAIN_PATH = r"Data\train.csv"
VALID_PATH = r"Data\test.csv"
TEST_PATH = r"Data\test.csv"
OUTPUT_PATH = r"Data\classification_results.csv"
LABEL_THRESHOLD = 0.5
POSITIVE_IF_HIGHER = True
RANDOM_STATE = 42


def binary_labels(labels, threshold, positive_if_higher):
    return ((labels >= threshold) if positive_if_higher else (labels <= threshold)).astype(int)


def metrics(labels, predictions):
    return {
        "Acc": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def save_results(records, output_path, complete=False):
    results = pd.DataFrame(records)
    results["Complete_run"] = complete
    results["Selected_by_validation"] = False
    if complete:
        results.loc[results["Valid_MCC"].idxmax(), "Selected_by_validation"] = True
    results = results.sort_values("Valid_MCC", ascending=False, kind="stable")
    results.to_csv(output_path, index=False, float_format="%.6f")
    return results


def main(train_path=TRAIN_PATH, valid_path=VALID_PATH, test_path=TEST_PATH,
         output_path=OUTPUT_PATH, threshold=LABEL_THRESHOLD,
         positive_if_higher=POSITIVE_IF_HIGHER):
    paths = [Path(path).resolve() for path in (train_path, valid_path, test_path)]
    if len(set(paths)) != 3:
        raise ValueError("Training, validation, and test paths must be different files.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    train, valid, test = [MolecularDataset(path) for path in paths]
    check_split_overlap(train, valid, test)
    y_train, y_valid, y_test = [
        binary_labels(data.labels, threshold, positive_if_higher)
        for data in (train, valid, test)
    ]
    for name, labels in zip(("train", "valid", "test"), (y_train, y_valid, y_test)):
        if np.unique(labels).size != 2:
            raise ValueError(f"{name} must contain both classes after thresholding.")

    neighbors = sorted({min(3, len(train)), min(7, len(train))})
    models = {
        "SVM": (SVC(kernel="rbf", cache_size=512),
                {"model__C": [1.0], "model__gamma": ["scale"]}),
        "RF": (RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
               {"model__max_depth": [None, 5], "model__min_samples_leaf": [1]}),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
                       {"model__max_depth": [None, 15], "model__min_samples_leaf": [1, 2]}),
        "LogisticRegression": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                               {"model__C": [0.1, 1.0, 10.0]}),
        "KNN": (KNeighborsClassifier(n_jobs=-1),
                {"model__n_neighbors": neighbors, "model__weights": ["uniform", "distance"]}),
    }
    split = PredefinedSplit(np.r_[np.full(len(train), -1), np.zeros(len(valid), dtype=int)])
    y_development = np.concatenate([y_train, y_valid])
    records = []

    for feature in FEATURE_NAMES:
        X_train, X_valid, X_test = [data.X(feature) for data in (train, valid, test)]
        X_development = np.vstack([X_train, X_valid])
        for name, (estimator, grid) in models.items():
            print(f"Training {feature} / {name}", flush=True)
            scaler = "passthrough" if name in {"RF", "ExtraTrees"} else StandardScaler()
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", scaler),
                ("model", clone(estimator)),
            ])
            search = GridSearchCV(
                pipeline, grid, scoring=make_scorer(matthews_corrcoef),
                cv=split, refit=False, n_jobs=1, error_score="raise",
            )
            search.fit(X_development, y_development)
            best_model = clone(pipeline).set_params(**search.best_params_)
            best_model.fit(X_train, y_train)
            valid_metrics = metrics(y_valid, best_model.predict(X_valid))
            test_metrics = metrics(y_test, best_model.predict(X_test))
            records.append({
                "Feature": feature, "Model": name, "N_features": X_train.shape[1],
                "N_train": len(train), "N_valid": len(valid), "N_test": len(test),
                "Label_threshold": threshold, "Positive_if_higher": positive_if_higher,
                "Best_params": json.dumps(search.best_params_, sort_keys=True),
                **{f"Valid_{key}": value for key, value in valid_metrics.items()},
                **test_metrics,
                "Excluded_descriptors": "Ipc;AvgIpc" if feature == "descriptors" else "",
                "RDKit_version": rdBase.rdkitVersion, "Sklearn_version": sklearn.__version__,
            })
            save_results(records, output_path)
            print(
                f"Saved {len(records)}/25: {feature} / {name}; "
                f"Valid MCC: {valid_metrics['MCC']:.4f}", flush=True,
            )

    results = save_results(records, output_path, complete=True)
    print(f"Completed all {len(results)} combinations: {output_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
