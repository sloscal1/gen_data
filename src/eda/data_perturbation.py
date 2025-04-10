from typing import Dict, List

import numpy as np  # type: ignore
import panel as pn
import plotly.graph_objects as go  # type: ignore

# Enable Panel extensions
pn.extension()

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
    # Interactive Panel
    dashboard = pn.Row(
        pn.Column(
            pn.pane.Markdown("# Data Perturbation Dashboard"),
            fraud_noise,
            non_fraud_noise,
            class_imbalance,
            pn.bind(
                update_plot,
                fraud_noise=fraud_noise,
                non_fraud_noise=non_fraud_noise,
                class_imbalance=class_imbalance,
            ),
        ),
        pn.Column(
            pn.bind(
                fraud_scatter_plot,
                fraud_noise=fraud_noise,
                class_imbalance=class_imbalance,
            ),
            pn.bind(
                non_fraud_scatter_plot,
                non_fraud_noise=non_fraud_noise,
                class_imbalance=class_imbalance,
            ),
        ),
    )

    # Serve the dashboard
    dashboard.servable()
    pn.serve(dashboard, start=True)
