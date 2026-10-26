
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from features_engineering import features_engineering


def main():
    all_stocks_5yr = features_engineering()

    # Séparation temporelle : test à partir de 2017
    train = all_stocks_5yr[
        all_stocks_5yr["date"] < "2017-01-01"
    ].copy()

    test = all_stocks_5yr[
        all_stocks_5yr["date"] >= "2017-01-01"
    ].copy()

    features = [
        "RSI_14",
        "MACD",
        "Bollinger_Bandwidth"
    ]

    X_train = train[features]
    y_train = train["target"]

    # Dates uniques, triées chronologiquement
    dates = pd.Index(
        train["date"].drop_duplicates().sort_values()
    )

    # Conserver plus de deux ans pour le premier entraînement
    first_date = dates[0]
    first_validation_date = first_date + pd.DateOffset(years=2)

    # Dates disponibles pour la validation
    validation_dates = dates[
        dates >= first_validation_date
    ]

    # Diviser la période restante en 10 blocs
    date_chunks = np.array_split(
        validation_dates.to_numpy(), 10
    )

    cv_splits = []

    for val_dates in date_chunks:
        if len(val_dates) == 0:
            continue

        # L'entraînement utilise uniquement les dates antérieures
        fold_train_dates = dates[
            dates < val_dates[0]
        ]

        train_idx = np.flatnonzero(
            train["date"].isin(fold_train_dates).to_numpy()
        )

        val_idx = np.flatnonzero(
            train["date"].isin(val_dates).to_numpy()
        )

        cv_splits.append((train_idx, val_idx))

    if len(cv_splits) != 10:
        raise ValueError(
            "Impossible de construire 10 folds temporels."
        )

    # Pipeline du modèle
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    # Hyperparamètres à tester
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splits,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure AUC moyenne :", grid_search.best_score_)


if __name__ == "__main__":
    main()
