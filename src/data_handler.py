import os
from typing import List, Tuple, Dict, Optional
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class DataHandler:
    """Class for loading and tokenizing local NER datasets."""

    def __init__(
            self,
            model_name: str,
            train_path: str,
            valid_path: str
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        self.train_path = train_path
        self.valid_path = valid_path
        self.dataset: Optional[DatasetDict] = None
        self.label_list: List[str] = []
        self.label2id: Dict[str, int] = {}

    def _parse_conll(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sentences, labels = [], []
        current_words, current_tags = [], []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if current_words:
                        sentences.append(current_words)
                        labels.append(current_tags)
                        current_words, current_tags = [], []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]
                    current_words.append(word)
                    current_tags.append(tag)
                    self.label_list.append(tag)

        return sentences, labels

    def load_dataset(self) -> DatasetDict:
        train_sentences, train_tags = self._parse_conll(self.train_path)
        valid_sentences, valid_tags = self._parse_conll(self.valid_path)

        self.label_list = sorted(set(self.label_list))
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}

        self.dataset = DatasetDict({
            "train": Dataset.from_dict({
                "tokens": train_sentences,
                "ner_tags": [[self.label2id[t] for t in tags] for tags in train_tags]
            }),
            "validation": Dataset.from_dict({
                "tokens": valid_sentences,
                "ner_tags": [[self.label2id[t] for t in tags] for tags in valid_tags]
            }),
        })

        return self.dataset

    def tokenize_and_align_labels(self) -> DatasetDict:
        def tokenize_fn(examples):
            tokenized = self.tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            aligned_labels = []
            for i, ner_tags in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_ids, prev_word_id = [], None
                for word_id in word_ids:
                    if word_id is None or word_id == prev_word_id:
                        label_ids.append(-100)
                    else:
                        label_ids.append(ner_tags[word_id])
                    prev_word_id = word_id
                aligned_labels.append(label_ids)
            tokenized["labels"] = aligned_labels
            return tokenized

        return self.dataset.map(tokenize_fn, batched=True)