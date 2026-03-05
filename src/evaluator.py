from typing import List, Tuple, Dict, Any

import evaluate
import numpy as np
from numpy.typing import NDArray


class Evaluator:
    """Computes entity-level NER evaluation metrics using seqeval."""

    def __init__(self, label_names: List[str]) -> None:
        self.label_names: List[str] = label_names
        self.metric = evaluate.load("seqeval")

    def compute_metrics(
        self,
        eval_pred: Tuple[NDArray[np.float_], NDArray[np.int_]],
    ) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions: NDArray[np.int_] = np.argmax(logits, axis=2)

        true_labels = [
            [self.label_names[l] for l in lab if l != -100]
            for lab in labels
        ]

        true_predictions = [
            [self.label_names[p] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]

        results: Dict[str, Any] = self.metric.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        final_metrics = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

        for key, value in results.items():
            if isinstance(value, dict):
                final_metrics[key] = value

        return final_metrics