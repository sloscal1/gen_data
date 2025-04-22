import pathlib
from typing import Dict, List

import numpy as np  # type: ignore
import pandas as pd
import panel as pn
import plotly.graph_objects as go  # type: ignore

from eda.data_generation import generate_fraud_data
from eda.model import train_model

# Enable Panel extensions
pn.extension()

generated_data = None
generated_data_dir = pathlib.Path("generated_data")


# Adjustable parameters with editable values
fraud_noise = pn.widgets.EditableFloatSlider(
    name="Fraud Label Noise (%)",
    fixed_start=0.0,
    fixed_end=50.0,
    step=0.01,
    value=0.0,
)
non_fraud_noise = pn.widgets.EditableFloatSlider(
    name="Non-Fraud Label Noise (%)",
    fixed_start=0.0,
    fixed_end=50.0,
    step=0.01,
    value=0.0,
)
class_imbalance = pn.widgets.EditableFloatSlider(
    name="Class Imbalance (%)",
    fixed_start=0.0,
    fixed_end=50.0,
    step=0.01,
    value=50.0,
)

# Store generated values for the scatter plot
scatter_data: Dict[str, List] = {
    "fraud_noise": [],
    "non_fraud_noise": [],
    "class_imbalance": [],
    "fraud_ratio": [],
}

# Button to trigger model training
train_button = pn.widgets.Button(name="Train Model", button_type="success")

# Table to display generated data
data_table = pn.widgets.DataFrame(
    value=pd.DataFrame(
        columns=["Fraud Noise (%)", "Non-Fraud Noise (%)", "Class Imbalance (%)"]
    ),
    autosize_mode="fit_columns",
    height=800,
    show_index=False,
)


# Callback function for the train button
def on_train_click(event):
    del event  # Unused in this function
    fraud_noise_rate = fraud_noise.value
    non_fraud_noise_rate = non_fraud_noise.value
    fraud_rate = class_imbalance.value
    selected_row = data_table.selection
    if selected_row:
        selected_data = data_table.value.iloc[selected_row[0]]
        print(selected_data)
        fraud_noise_rate = selected_data["Fraud Noise (%)"]
        non_fraud_noise_rate = selected_data["Non-Fraud Noise (%)"]
        fraud_rate = selected_data["Class Imbalance (%)"]

    if not pathlib.Path(
        generated_data_dir
        / f"transactions_imbalance_{fraud_rate:0.2f}_fraud_noise_{fraud_noise_rate:0.2f}_non_fraud_noise_{non_fraud_noise_rate:0.2f}.csv"
    ).exists():
        print("No data generated yet. Please generate data first.")
        return

    # Train the model with the current slider values
    train_model(fraud_rate, fraud_noise_rate, non_fraud_noise_rate)
    print("Model training completed.")


# Link the train button to the callback
train_button.on_click(on_train_click)

# Button to trigger fraud transaction generation
generate_button = pn.widgets.Button(
    name="Generate Fraud Transactions", button_type="primary"
)


# Callback function for the button
def on_generate_click(event):
    del event  # Unused in this function
    pathlib.Path(generated_data_dir).mkdir(parents=True, exist_ok=True)
    fraud_noise_value = fraud_noise.value
    non_fraud_noise_value = non_fraud_noise.value
    class_imbalance_value = class_imbalance.value

    global generated_data
    generated_data = generate_fraud_data(
        config_path="./config.yaml",
        fraud_noise_rate=fraud_noise_value / 100,
        non_fraud_noise_rate=non_fraud_noise_value / 100,
        fraud_rate=class_imbalance_value / 100,
    )
    generated_data.to_csv(
        generated_data_dir
        / f"transactions_imbalance_{class_imbalance_value:0.2f}_fraud_noise_{fraud_noise_value:0.2f}_non_fraud_noise_{non_fraud_noise_value:0.2f}.csv",
        index=False,
        header=True,
    )

    # Add a new row to the table
    new_row = {
        "Fraud Noise (%)": fraud_noise_value,
        "Non-Fraud Noise (%)": non_fraud_noise_value,
        "Class Imbalance (%)": class_imbalance_value,
    }
    data_table.value = pd.concat(
        [data_table.value, pd.DataFrame([new_row])], ignore_index=True
    ).drop_duplicates()
    data_table.selection = [len(data_table.value) - 1]  # Select the last row


# Function to simulate data and plot
def update_plot(
    fraud_noise: float, non_fraud_noise: float, class_imbalance: float
) -> go.Figure:
    fraud_noise /= 100
    non_fraud_noise /= 100
    class_imbalance /= 100
    # Simulate data
    classes = [1, 0]
    total_samples = 10_000
    imbalance_ratio = [class_imbalance, 1 - class_imbalance]
    samples_per_class = (np.array(imbalance_ratio) * total_samples).astype(int)
    samples_per_class[1] -= (
        samples_per_class.sum() - total_samples
    )  # Ensure rounding in above line doesn't change dimensions
    labels = np.concatenate(
        [
            np.full(samples_per_class[0], 1),
            np.full(samples_per_class[1], 0),
        ]
    )

    # Add label noise
    noisy_fraud_labels = np.concatenate(
        [
            np.random.rand(samples_per_class[0]) < fraud_noise,
            np.full(samples_per_class[1], False),
        ]
    )
    noisy_non_fraud_labels = np.concatenate(
        [
            np.full(samples_per_class[0], False),
            np.random.rand(samples_per_class[1]) < non_fraud_noise,
        ]
    )
    labels[noisy_fraud_labels] = 0
    labels[noisy_non_fraud_labels] = 1

    # Count occurrences of each class
    class_counts = {cls: (labels == cls).sum() for cls in classes}
    str_class_counts = {"Fraud": class_counts[1], "Non-Fraud": class_counts[0]}
    fraud_ratio = str_class_counts["Fraud"] / str_class_counts["Non-Fraud"]

    # Update scatter data
    scatter_data["fraud_noise"].append(fraud_noise)
    scatter_data["non_fraud_noise"].append(non_fraud_noise)
    scatter_data["class_imbalance"].append(class_imbalance)
    scatter_data["fraud_ratio"].append(fraud_ratio)

    # Create Plotly bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(str_class_counts.keys()),
                y=list(str_class_counts.values()),
                marker=dict(color="skyblue"),
            )
        ],
    )
    fig.update_layout(
        title="Class Distribution with Noise",
        xaxis_title="Class",
        yaxis_title="Count",
        bargap=0.2,
    )

    return fig


def non_fraud_scatter_plot(non_fraud_noise: float, class_imbalance: float) -> go.Figure:
    del non_fraud_noise, class_imbalance  # Unused in this function
    fig = go.Figure(
        data=[
            go.Scatter(
                x=scatter_data["class_imbalance"],
                y=scatter_data["non_fraud_noise"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=scatter_data["fraud_ratio"],
                    colorscale="Viridis",
                    colorbar=dict(title="Fraud/Non-Fraud Ratio"),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Non-Fraud Label Noise vs Class Imbalance",
        xaxis_title="Class Imbalance",
        yaxis_title="Non-Fraud Label Noise",
    )
    return fig


# Function to generate scatter plot
def fraud_scatter_plot(fraud_noise: float, class_imbalance: float) -> go.Figure:
    del fraud_noise, class_imbalance  # Unused in this function
    fig = go.Figure(
        data=[
            go.Scatter(
                x=scatter_data["class_imbalance"],
                y=scatter_data["fraud_noise"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=scatter_data["fraud_ratio"],
                    colorscale="Viridis",
                    colorbar=dict(title="Fraud/Non-Fraud Ratio"),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Fraud Label Noise vs Class Imbalance",
        xaxis_title="Class Imbalance",
        yaxis_title="Fraud Label Noise",
    )
    return fig


if __name__ == "__main__":
    np.random.seed(1337)  # Might set this in each random call
    # Populate the data table with existing CSV files in the generated_data directory
    if generated_data_dir.exists():
        csv_files = list(generated_data_dir.glob("*.csv"))
        rows = []
        for file in csv_files:
            try:
                # Extract parameters from the file name
                parts = file.stem.split("_")
                fraud_noise_value = float(parts[5])
                non_fraud_noise_value = float(parts[9])
                class_imbalance_value = float(parts[2])
                rows.append(
                    {
                        "Fraud Noise (%)": fraud_noise_value,
                        "Non-Fraud Noise (%)": non_fraud_noise_value,
                        "Class Imbalance (%)": class_imbalance_value,
                    }
                )
            except (IndexError, ValueError):
                print(f"Skipping file with unexpected format: {file.name}")
        if rows:
            data_table.value = pd.DataFrame(rows).drop_duplicates()

    # Link the button to the callback
    generate_button.on_click(on_generate_click)

    # Interactive Panel
    dashboard = pn.Row(
        pn.Column(
            pn.pane.Markdown("# Data Perturbation Dashboard"),
            fraud_noise,
            non_fraud_noise,
            class_imbalance,
            generate_button,
            train_button,
            pn.bind(
                update_plot,
                fraud_noise=fraud_noise,
                non_fraud_noise=non_fraud_noise,
                class_imbalance=class_imbalance,
            ),
        ),
        pn.Column(
            data_table,
        ),
    )

    # Serve the dashboard
    print("serving")
    dashboard.servable()
    print("up")
    pn.serve(dashboard, start=True)
