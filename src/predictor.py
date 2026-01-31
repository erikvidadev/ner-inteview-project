from typing import List, Dict, Any

from transformers import (
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class Predictor:
    """Runs NER inference on input text using a Hugging Face pipeline."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.pipeline = pipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        return self.pipeline(text)
