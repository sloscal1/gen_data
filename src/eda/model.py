from collections import defaultdict
from typing import Dict, List, Tuple

import optuna
import optuna.integration
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import xgboost as xgb
from sklearn.metrics import (  # type: ignore
    auc,
    average_precision_score,
    precision_recall_curve,
)

num_boost_round = 10
default_fraud_rate = 0.0015
n_trials = 10
hyperparameters = {
    "eta": ("suggest_float", 0.01, 0.3, True),
    "max_depth": ("suggest_int", 3, 10, False),
    "min_child_weight": ("suggest_float", 1e-3, 10, True),
    "subsample": ("suggest_float", 0.5, 1.0, False),
    "colsample_bytree": ("suggest_float", 0.5, 1.0, False),
    "lambda": ("suggest_float", 1e-3, 10, True),
    "alpha": ("suggest_float", 1e-3, 10, True),
}
unused_for_training = [
    "fraud_label",
    "transaction_timestamp",
    "card_id",
    "transaction_id",
    "customer_id",
    "fraud_on_card",
    "compromised_date",
    "first_fraud",
]


data = pd.DataFrame()
target_encoded: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)


def generate_temporal_folds(
    data: pd.DataFrame, target: str, n_splits: int = 5
) -> List[Tuple[int, int]]:
    # Sort data by transaction_timestamp
    data = data.sort_values(by="transaction_timestamp")

    # Create temporal folds
    fold_size = len(data) // (n_splits + 1)  # There is always a holdout set
    folds = []
    for i in range(n_splits):
        end_idx = (i + 1) * fold_size
        folds.append((i, end_idx))

    not_encoded = {
        "card_id",
        "transaction_id",
        "customer_id",
        "transaction_timestamp",
        target,
    }

    if not target_encoded:
        categoricals = data.select_dtypes(include=["category"]).columns.tolist()
        categoricals = list(set(categoricals) - not_encoded)

        for fold, stop in folds:
            fold_data = data.iloc[:stop][categoricals + [target]].copy()
            for col in categoricals:
                encoding_map = (
                    fold_data.groupby(col, observed=True)[target].mean().to_dict()
                )
                target_encoded[fold][col] = encoding_map

    return folds


def apply_target_encoding(
    data: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    """
    Apply target encoding to the DataFrame for a specific fold.

    Args:
        data (pd.DataFrame): The input DataFrame containing transaction data.
        fold (int): The fold number for which to apply target encoding.
        target (str): The target variable for encoding.

    Returns:
        pd.DataFrame: The DataFrame with target encoding applied.
    """
    if fold not in target_encoded:
        raise ValueError("Fold not found in target_encoded.")

    data = data.copy()
    for col, encoding_map in target_encoded[fold].items():
        data[col] = data[col].astype("str").map(encoding_map).fillna(default_fraud_rate)

    return data


# Define objective function for Optuna
def objective(trial: optuna.Trial) -> float:
    param = {
        field: getattr(trial, hyperparameters[field][0])(
            field,
            hyperparameters[field][1],
            hyperparameters[field][2],
            log=hyperparameters[field][3],
        )
        for field in hyperparameters
    }
    param = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "booster": "gbtree",
        **param,
    }

    # Sort data by transaction_timestamp for temporal holdout
    data_sorted = prepare_data(data)
    folds = generate_temporal_folds(data_sorted, "fraud_label")

    # Split into train and test based on temporal order
    aucpr = []
    for fold, stop_index in folds:
        train_data = data_sorted.iloc[:stop_index]
        test_data = data_sorted.iloc[stop_index:]
        X_train, y_train = (
            train_data.drop(columns=unused_for_training),
            train_data["fraud_label"],
        )
        X_test, y_test = (
            test_data.drop(
                columns=unused_for_training,
            ),
            test_data["fraud_label"],
        )
        X_train = apply_target_encoding(X_train, fold)
        X_test = apply_target_encoding(X_test, fold)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # pruning = optuna.integration.XGBoostPruningCallback(trial, "validation-aucpr")

        model = xgb.train(
            param,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False,
            # callbacks=[pruning],
        )
        preds = model.predict(dtest)
        aucpr.append(average_precision_score(y_test, preds))
    return sum(aucpr) / len(aucpr)


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the input DataFrame by performing the following operations:
    1. Converts specified categorical columns to the 'category' data type.
    2. Converts the 'transaction_timestamp' column to datetime format.
    3. Sorts the DataFrame by the 'transaction_timestamp' column.

    Args:
        data (pd.DataFrame): The input DataFrame containing transaction data.
            It must include the following columns:
            - "card_id"
            - "transaction_id"
            - "merchant_id"
            - "merchant_name"
            - "customer_id"
            - "location"
            - "transaction_type"
            - "transaction_timestamp"

    Returns:
        pd.DataFrame: The processed DataFrame with updated data types and sorted rows.
    """
    # Convert categorical columns to category dtype
    categorical_cols = [
        "card_id",
        "transaction_id",
        "merchant_id",
        "merchant_name",
        "customer_id",
        "location",
        "transaction_type",
    ]
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    # Convert transaction_timestamp to datetime
    data["transaction_timestamp"] = pd.to_datetime(data["transaction_timestamp"])

    # Sort data by transaction_timestamp
    data = data.sort_values(by="transaction_timestamp")

    return data


def main() -> None:
    # Load the dataset
    global data
    data = pd.read_csv("synthetic_transactions.csv")

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
    )

    # Best parameters
    print("Best parameters:", study.best_params)

    # Train final model with best parameters and plot AUCPR
    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "aucpr"

    # Sort data by transaction_timestamp for temporal holdout
    data_sorted = prepare_data(data)
    folds = generate_temporal_folds(data_sorted, "fraud_label", n_splits=5)

    train_data = data_sorted.iloc[: folds[-1][1]]
    test_data = data_sorted.iloc[folds[-1][1] :]
    X_train, y_train = (
        train_data.drop(
            columns=unused_for_training,
        ),
        train_data["fraud_label"],
    )
    X_test, y_test = (
        test_data.drop(
            columns=unused_for_training,
        ),
        test_data["fraud_label"],
    )
    X_train = apply_target_encoding(X_train, folds[-1][0])
    X_test = apply_target_encoding(X_test, folds[-1][0])
    print(X_test.columns)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
        # callbacks=[pruning],
    )
    preds = model.predict(dtest)

    precision, recall, _ = precision_recall_curve(y_test, preds)
    aucpr = auc(recall, precision)
    precision_list = [precision]
    recall_list = [recall]
    aucpr_list = [aucpr]

    # Plot PR curve
    fig = go.Figure()

    for precision, recall in zip(precision_list, recall_list):
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", opacity=0.3))

    baseline_aucpr = y_test.sum() / len(y_test)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline_aucpr] * 2,
            mode="lines",
            name="Baseline",
            line=dict(dash="dash"),
        )
    )

    mean_aucpr = sum(aucpr_list) / len(aucpr_list)
    fig.update_layout(
        title=f"Precision-Recall Curve (Mean AUCPR: {mean_aucpr:.4f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
    )
    fig.show()

    fig = optuna.visualization.plot_slice(study, params=hyperparameters)
    fig.show()


if __name__ == "__main__":
    main()
