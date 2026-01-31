from transformers import AutoModelForTokenClassification, PreTrainedModel


class ModelFactory:
    """Factory for creating token classification models for NER."""

    @staticmethod
    def create(model_name: str, num_labels: int) -> PreTrainedModel:
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
