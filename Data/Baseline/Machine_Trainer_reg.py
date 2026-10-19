import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from Feature_Dataset import FEATURE_NAMES, MolecularDataset, check_split_overlap



TRAIN_PATH = r"Data\train.csv"
VALID_PATH = r"Data\test.csv"
TEST_PATH = r"Data\test.csv"
OUTPUT_PATH = r"Data\regression_results.csv"
RANDOM_STATE = 42


def metrics(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    return {
        "MAE": mean_absolute_error(labels, predictions),
        "R2": r2_score(labels, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }


def main(train_path=TRAIN_PATH, valid_path=VALID_PATH, test_path=TEST_PATH,
         output_path=OUTPUT_PATH):
    train, valid, test = [MolecularDataset(path) for path in (train_path, valid_path, test_path)]
    check_split_overlap(train, valid, test)
    if min(len(train), len(valid), len(test)) < 2:
        raise ValueError("Each regression partition must contain at least two records.")
    y_train, y_valid, y_test = [data.labels for data in (train, valid, test)]
    neighbors = sorted({min(3, len(train)), min(7, len(train))})
    models = {
        "SVM": (SVR(kernel="rbf", cache_size=512),
                {"model__C": [1.0, 5.0], "model__epsilon": [0.1, 1.0]}),
        "RF": (RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
               {"model__max_depth": [None, 5, 15], "model__min_samples_leaf": [1, 10, 100]}),
        "ExtraTrees": (ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
                       {"model__max_depth": [None, 15], "model__min_samples_leaf": [1, 2]}),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        "KNN": (KNeighborsRegressor(n_jobs=-1),
                {"model__n_neighbors": neighbors, "model__weights": ["uniform", "distance"]}),
    }
    split = PredefinedSplit(np.r_[np.full(len(train), -1), np.zeros(len(valid), dtype=int)])
    y_development = np.concatenate([y_train, y_valid])
    records = []

    for feature in FEATURE_NAMES:
        X_train, X_valid, X_test = [data.X(feature) for data in (train, valid, test)]
        X_development = np.vstack([X_train, X_valid])
        for name, (estimator, grid) in models.items():
            scaler = "passthrough" if name in {"RF", "ExtraTrees"} else StandardScaler()
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", scaler),
                ("model", clone(estimator)),
            ])
            search = GridSearchCV(
                pipeline, grid, scoring="neg_mean_absolute_error",
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
                "Best_params": json.dumps(search.best_params_, sort_keys=True),
                **{f"Valid_{key}": value for key, value in valid_metrics.items()},
                **test_metrics,
                "RDKit_version": rdBase.rdkitVersion, "Sklearn_version": sklearn.__version__,
            })
            print(f"{feature:12s} {name:20s} Valid MAE: {valid_metrics['MAE']:.4f}", flush=True)

    results = pd.DataFrame(records)
    results["Selected_by_validation"] = False
    results.loc[results["Valid_MAE"].idxmin(), "Selected_by_validation"] = True
    results = results.sort_values("Valid_MAE", ascending=True, kind="stable")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved {len(results)} combinations to {output_path}")
    return results


if __name__ == "__main__":
    main()
