import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from typing import List

from src.utils import DeviceManager
from src.data_handler import DataHandler
from src.model_factory import ModelFactory
from src.evaluator import Evaluator
from src.trainer import ModelTrainer
from src.predictor import Predictor
from transformers import DataCollatorForTokenClassification


def main() -> None:
    data_loader = DataHandler(
        model_name="distilbert/distilbert-base-cased",
        train_path="data/eng.train",
        valid_path="data/eng.testa"
    )

    data_loader.load_dataset()
    dataset = data_loader.tokenize_and_align_labels()
    label_names: List[str] = data_loader.label_list

    # Model Factory + Label Mapping
    device = DeviceManager.get_optimal_device()
    model = ModelFactory.create(model_name="distilbert/distilbert-base-cased", num_labels=len(label_names))

    # Injecting label mappings into model configuration
    model.config.id2label = {i: label for i, label in enumerate(label_names)}
    model.config.label2id = {label: i for i, label in enumerate(label_names)}
    model.to(device)

    # Metrics and Collator
    evaluator = Evaluator(label_names)
    data_collator = DataCollatorForTokenClassification(tokenizer=data_loader.tokenizer)

    # Training (ModelTrainer)
    trainer_wrapper = ModelTrainer(
        model=model,
        tokenizer=data_loader.tokenizer,
        evaluator=evaluator,
        output_dir="./outputs",
        batch_size=4,
        epochs=1
    )

    train_subset = dataset["train"].shuffle(seed=42)
    eval_subset = dataset["validation"]

    print(f"\n--- Starting Training ---")
    print(f"Device: {device} | Training file: {"data/eng.train"}")

    trainer_wrapper.train(
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        data_collator=data_collator
    )


    print("\n--- Test Predictions ---")
    predictor = Predictor(model, data_loader.tokenizer)

    test_sentences: List[str] = [
        "Apple was founded by Steve Jobs in California.",
        "Budapest is the capital of Hungary.",
        "Elon Musk bought Twitter and renamed it to X."
    ]

    for text in test_sentences:
        result = predictor.predict(text)
        print(f"\nInput: {text}\nOutput: {result}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical Error: {e}")