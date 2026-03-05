import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix


class Visualizer:
    """
    Responsibility: Visualizing and saving training and validation results.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_training_history(self, history: List[Dict[str, Any]]):
        """
        Plots Training and Validation Loss curves based on the log history.
        Goal: Visualize if the model is learning (decreasing loss) or overfitting (diverging curves).
        """
        train_loss = []
        eval_loss = []
        train_steps = []
        eval_steps = []

        # Extract data from the Hugging Face Trainer log history
        for entry in history:
            if "loss" in entry and "step" in entry:
                train_loss.append(entry["loss"])
                train_steps.append(entry["step"])
            elif "eval_loss" in entry and "step" in entry:
                eval_loss.append(entry["eval_loss"])
                eval_steps.append(entry["step"])

        if not train_loss:
            print("[Visualizer] Warning: No training loss data found to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_loss, label="Training Loss", color="#1f77b4")

        if eval_loss:
            plt.plot(eval_steps, eval_loss, label="Validation Loss", color="#ff7f0e", linestyle="--",
                     marker="o")

        plt.title("Training and Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()

        save_path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(save_path)
        plt.close()
        print(f"   [Visualizer] Loss curve saved to: {save_path}")

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], labels: List[str]):
        """
        Plots the confusion matrix.
        Goal: Identify which entities are being confused with others by the model.
        """
        # normalize='true' helps compare ratios across categories with different sample sizes
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            cbar=False
        )

        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"   [Visualizer] Confusion matrix saved to: {save_path}")

    def plot_entity_performance(self, metrics: Dict[str, Any]):
        """
        Bar chart showing the F1-score performance for each entity type.
        """
        # Data processing: Extract entity-level metrics from the seqeval output
        entities = []
        f1_scores = []

        # seqeval output contains overall metrics and per-entity dictionaries
        for key, value in metrics.items():
            if isinstance(value, dict) and "f1" in value:
                entities.append(key)
                f1_scores.append(value["f1"])

        if not entities:
            print("[Visualizer] No entity-level metrics found.")
            return

        # Use Pandas DataFrame for easy plotting
        df = pd.DataFrame({"Entity": entities, "F1 Score": f1_scores})

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x="Entity", y="F1 Score", data=df, palette="viridis", hue="Entity", legend=False)
        plt.ylim(0, 1.05)
        plt.title("F1 Score per Entity Type")

        # Annotate values on top of the bars
        for i, v in enumerate(f1_scores):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

        save_path = os.path.join(self.output_dir, "entity_performance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"   [Visualizer] Entity performance chart saved to: {save_path}")