import os
import shutil
from typing import Any, Optional
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import TrainOutput
from src.utils import DeviceManager


class ModelTrainer:
    """Encapsulates the training logic for a NER model using centralized device management."""

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            evaluator: Any,
            output_dir: str = "./outputs",
            batch_size: int = 4,
            epochs: int = 1,
    ) -> None:

        self.device = DeviceManager.get_optimal_device()
        self.model: PreTrainedModel = model.to(self.device)
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.evaluator: Any = evaluator
        self.output_dir: str = os.path.abspath(output_dir)
        self.trainer: Optional[Trainer] = None

        if os.path.exists(self.output_dir):
            print(f"--- Cleaning up previous training data in: {self.output_dir} ---")
            shutil.rmtree(self.output_dir)

        self.training_args: TrainingArguments = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            save_total_limit=1,
            report_to="none",
            fp16=True if self.device.type == "cuda" else False,
            dataloader_pin_memory=True if self.device.type == "cuda" else False,
        )

    def train(
            self,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            data_collator: Any,
    ):

        DeviceManager.cleanup_memory(self.device)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.evaluator.compute_metrics,
            data_collator=data_collator,
        )

        try:
            train_result: TrainOutput = self.trainer.train()
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"--- Training finished. Model saved to: {self.output_dir} ---")
            return train_result

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n[Error] {self.device.type.upper()} out of memory. Reduce batch size.")
            raise exc

        finally:
            DeviceManager.cleanup_memory(self.device)