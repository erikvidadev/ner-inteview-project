import os
import numpy as np
from typing import List

# Optimize Apple Silicon / GPU memory management
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from src.utils import DeviceManager
from src.data_handler import DataHandler
from src.model_factory import ModelFactory
from src.evaluator import Evaluator
from src.trainer import ModelTrainer
from src.predictor import Predictor
from src.visualizer import Visualizer
from transformers import DataCollatorForTokenClassification

CONFIG = {
    "model_name": "distilbert/distilbert-base-cased",
    "train_path": "data/eng.train",
    "valid_path": "data/eng.testa",
    "output_dir": "./outputs",
    "vis_dir": "./plots",
    "batch_size": 4,
    "epochs": 1
}

def main() -> None:
    # 1. Data Processing
    print("\n[1/5] Loading and processing dataset...")
    data_loader = DataHandler(
        model_name=CONFIG["model_name"],
        train_path=CONFIG["train_path"],
        valid_path=CONFIG["valid_path"]
    )
    data_loader.load_dataset()
    dataset = data_loader.tokenize_and_align_labels()
    label_names: List[str] = data_loader.label_list

    # 2. Model Configuration
    print(f"\n[2/5] Initializing model: {CONFIG['model_name']}...")
    device = DeviceManager.get_optimal_device()

    model = ModelFactory.create(
        model_name=CONFIG["model_name"],
        num_labels=len(label_names)
    )

    model.config.id2label = {i: label for i, label in enumerate(label_names)}
    model.config.label2id = {label: i for i, label in enumerate(label_names)}
    model.to(device)

    # 3. Training Setup
    evaluator = Evaluator(label_names)
    data_collator = DataCollatorForTokenClassification(tokenizer=data_loader.tokenizer)

    trainer_wrapper = ModelTrainer(
        model=model,
        tokenizer=data_loader.tokenizer,
        evaluator=evaluator,
        output_dir=CONFIG["output_dir"],
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"]
    )

    train_subset = dataset["train"].shuffle(seed=42)  #.select(range(50))
    eval_subset = dataset["validation"]   #.select(range(20))

    # 4. Training Execution
    print(f"\n[3/5] Starting training on device: {device}...")
    trainer_wrapper.train(
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        data_collator=data_collator
    )

    # 5. Visualization and Diagnostics
    print("\n[4/5] Generating visualizations...")
    viz = Visualizer(output_dir=CONFIG["vis_dir"])

    if trainer_wrapper.trainer.state.log_history:
        viz.plot_training_history(trainer_wrapper.trainer.state.log_history)

    print("   -> Running validation predictions for diagnostics...")
    pred_output = trainer_wrapper.trainer.predict(eval_subset)

    print("\n--- Calculated Entity-Level Metrics ---")
    for metric_name, value in pred_output.metrics.items():
        if isinstance(value, dict) or "f1" in metric_name:
            print(f" > {metric_name}: {value}")
    print("--------------------------------------\n")

    # Filter labels and predictions for Confusion Matrix (removing -100 masks)
    predictions = np.argmax(pred_output.predictions, axis=2)
    labels = pred_output.label_ids

    y_true_filtered = []
    y_pred_filtered = []

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                y_true_filtered.append(label_names[labels[i][j]])
                y_pred_filtered.append(label_names[predictions[i][j]])

    viz.plot_confusion_matrix(y_true_filtered, y_pred_filtered, label_names)
    viz.plot_entity_performance(pred_output.metrics)

    # 6. Inference Test
    print("\n[5/5] Running inference tests...")
    predictor = Predictor(model, data_loader.tokenizer)

    test_sentences: List[str] = [
        "Apple was founded by Steve Jobs in California.",
        "Budapest is the capital of Hungary.",
        "Elon Musk bought Twitter."
    ]

    for text in test_sentences:
        result = predictor.predict(text)
        print(f"\nInput: {text}")
        for entity in result:
            print(f"  - {entity['entity_group']}: {entity['word']} ({entity['score']:.4f})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCritical Error: {e}")