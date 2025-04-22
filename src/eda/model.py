import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import mlflow.xgboost
import optuna
import optuna.integration
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import shap  # type: ignore
import xgboost as xgb
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import (  # type: ignore
    auc,
    average_precision_score,
    precision_recall_curve,
)

import mlflow

results_dir = pathlib.Path("results")
parent_id: str = ""

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
    "approved",
    "final_fraud",
    "transaction_timestamp",
    "card_id",
    "transaction_id",
    "customer_id",
    "fraud_on_card",
    "compromised_date",
    "first_fraud",
    "fraud",
]
target = "final_fraud"
mlflow_callback = MLflowCallback(metric_name="aucpr")


data = pd.DataFrame()
target_encoded: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)


def generate_temporal_folds(
    data: pd.DataFrame, target: str, n_splits: int = 5
) -> List[Tuple[int, int]]:
    # Sort data by transaction_timestamp
    data = data.sort_values(by="transaction_timestamp")

    # Create temporal folds
    fold_size = len(data) // (n_splits + 1)  # There is always a holdout set
    folds: List[Tuple[int, int]] = []
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
    for fold, train_end in folds:
        print(
            f"Fold {fold}: {len(data.iloc[:train_end])} train transactions, "
            f"{data.iloc[:train_end][target].sum()} train fraud transactions "
            f"{len(data.iloc[train_end:])} test transactions, "
            f"{data.iloc[train_end:][target].sum()} test fraud transactions"
        )

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
        "random_state": 1337,
        **param,
    }

    # Sort data by transaction_timestamp for temporal holdout
    data_sorted = prepare_data(data)
    folds = generate_temporal_folds(data_sorted, target)

    # Start MLflow run
    with mlflow.start_run(parent_run_id=parent_id, nested=True):
        mlflow.log_params(param)
        aucpr = []
        for fold, stop_index in folds:
            train_data = data_sorted.iloc[:stop_index]
            test_data = data_sorted.iloc[stop_index:]
            X_train, y_train = (
                train_data.drop(columns=unused_for_training, errors="ignore"),
                train_data[target],
            )
            X_test, y_test = (
                test_data.drop(
                    columns=unused_for_training,
                    errors="ignore",
                ),
                test_data[target],
            )
            X_train = apply_target_encoding(X_train, fold)
            X_test = apply_target_encoding(X_test, fold)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            pruning = optuna.integration.XGBoostPruningCallback(trial, "train-aucpr")

            model = xgb.train(
                param,
                dtrain=dtrain,
                maximize=True,
                num_boost_round=num_boost_round,
                callbacks=[pruning],
                evals=[(dtrain, "train")],
            )

            preds = model.predict(dtest)
            aucpr.append(average_precision_score(y_test, preds))

        # Retrieve the study from the trial
        current_value = sum(aucpr) / len(aucpr)
        mlflow.log_metric("aucpr", current_value)
        if len(trial.study.get_trials()) == 1 or current_value > trial.study.best_value:
            model.save_model(results_dir / "xgb.json")
    return current_value


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
            - "zip_code"
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
        "zip_code",
        "transaction_type",
    ]
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    # Convert transaction_timestamp to datetime
    data["transaction_timestamp"] = pd.to_datetime(data["transaction_timestamp"])

    # Sort data by transaction_timestamp
    data = data.sort_values(by="transaction_timestamp")

    return data


def plot_aucpr(
    precision_list: List[float],
    recall_list: List[float],
    aucpr_list: List[float],
    y_test: pd.Series,
) -> go.Figure:
    # Plot PR curve
    fig = go.Figure()

    for precision, recall, line_color, name in zip(
        precision_list,
        recall_list,
        ["red"] + ["black"] * (len(precision_list) - 1),
        ["True Label", "Observed Label"],
    ):
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                opacity=0.3,
                line=dict(color=line_color),
                name=name,
            )
        )

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
    return fig


def plot_shap_values(model, X_test: pd.DataFrame, class_names: List[str]) -> plt.Figure:
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False, class_names=class_names)
    plt.title("SHAP Summary Plot")
    plt.xlabel("SHAP Feature Importance Value")
    plt.ylabel("Feature Name")
    return plt.gcf()


def train_model(imbalance: float, fraud_noise: float, non_fraud_noise: float) -> None:
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    name_suffix = f"_{imbalance:0.2f}_{fraud_noise:0.2f}_{non_fraud_noise:0.2f}"
    # Load the dataset
    global data
    data = pd.read_csv(
        "generated_data/"
        f"transactions_imbalance_{imbalance:0.2f}"
        f"_fraud_noise_{fraud_noise:0.2f}"
        f"_non_fraud_noise_{non_fraud_noise:0.2f}.csv"
    )

    # Start MLflow experiment
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run() as parent_run:
        global parent_id
        parent_id = parent_run.info.run_id
        mlflow.log_param("imbalance", imbalance)
        mlflow.log_param("fraud_noise", fraud_noise)
        mlflow.log_param("non_fraud_noise", non_fraud_noise)

        # Check if the study file exists
        study_path = results_dir / f"study{name_suffix}.pkl"
        storage_name = f"sqlite:///{study_path}.db"
        study = optuna.create_study(
            study_name=f"study{name_suffix}", storage=storage_name, load_if_exists=True
        )
        study.optimize(
            objective,
            n_trials=n_trials,
        )
        # Save the study to a file
        study.trials_dataframe().to_csv(study_path, index=False)
        if (results_dir / "xgb.json").exists():
            (results_dir / "xgb.json").rename(results_dir / f"xgb{name_suffix}.json")

        # Log Optuna study results
        mlflow.log_artifact(str(study_path))

        # Sort data by transaction_timestamp for temporal holdout
        data_sorted = prepare_data(data)
        folds = generate_temporal_folds(data_sorted, target, n_splits=5)

        test_data = data_sorted.iloc[folds[-1][1] :]
        X_test, y_test = (
            test_data.drop(columns=unused_for_training, errors="ignore"),
            test_data[target],
        )
        X_test = apply_target_encoding(X_test, folds[-1][0])
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.Booster()
        model.load_model(results_dir / f"xgb{name_suffix}.json")
        # Save and log the model
        mlflow.xgboost.log_model(
            model, artifact_path="model.json", input_example=X_test
        )
        preds = model.predict(dtest)

        precision_list, recall_list, aucpr_list = [], [], []
        for baseline in [test_data["fraud"], y_test]:
            precision, recall, _ = precision_recall_curve(baseline, preds)
            aucpr = auc(recall, precision)
            precision_list.append(precision)
            recall_list.append(recall)
            aucpr_list.append(aucpr)

        # Log final metrics
        mean_aucpr = sum(aucpr_list) / len(aucpr_list)
        mlflow.log_metric("final_mean_aucpr", mean_aucpr)
        mlflow.log_figure(
            plot_shap_values(model, X_test, ["non-fraud", "fraud"]),
            f"shap_summary{name_suffix}.png",
        )
        mlflow.log_figure(
            plot_aucpr(precision_list, recall_list, aucpr_list, y_test), "aucpr.png"
        )
        mlflow.log_figure(
            optuna.visualization.plot_slice(study, params=hyperparameters),
            "optuna_slice.png",
        )
