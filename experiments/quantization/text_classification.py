import logging
from typing import Any, Tuple

from experiments.base_experiment import BaseExperiment, ExperimentResult
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, GPT2Model

logger = logging.getLogger(__name__)


class TextClassificationExperiment(BaseExperiment):
    """Experiment for text classification tasks."""

    def __init__(
        self,
        dataset: str,
        model_name: str,
        device: str = "cuda",
        data_root: str = "./data",
        output_dir: str = "./experiments/results",
        batch_size: int = 16,
    ):
        super().__init__(
            task="text_classification",
            dataset=dataset,
            model_name=model_name,
            device=device,
            data_root=data_root,
            output_dir=output_dir,
        )
        self.batch_size = batch_size
        self.tokenizer = None

    def load_model(self) -> nn.Module:
        """Load text classification model."""

        model_name_lower = self.model_name.lower()

        # Try to load from transformers
        try:
            if "bert" in model_name_lower:
                model_name = "bert-base-uncased"
                base_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "encoder"

            elif "roberta" in model_name_lower:
                model_name = "roberta-base"
                base_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "encoder"

            elif "distilbert" in model_name_lower:
                model_name = "distilbert-base-uncased"
                base_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "encoder"

            elif "albert" in model_name_lower:
                model_name = "albert-base-v2"
                base_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "encoder"

            elif "t5" in model_name_lower:
                model_name = "t5-small"
                base_model = T5EncoderModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "t5"

            elif "gpt" in model_name_lower or "gpt2" in model_name_lower:
                model_name = "gpt2"
                base_model = GPT2Model.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "decoder"

            else:
                model_name = "bert-base-uncased"
                base_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_type = "encoder"

            # Add classification head
            num_classes = self._get_num_classes()
            model = TextClassifier(base_model, num_classes, model_type=model_type)

            return model
        except Exception as e:
            logger.info(f"Could not load from transformers: {e}")
            # Fallback to simple model
            return self._create_simple_text_model()

    def _create_simple_text_model(self) -> nn.Module:
        """Create a simple text classification model."""

        class SimpleTextClassifier(nn.Module):
            def __init__(self, vocab_size=10000, embed_dim=128, num_classes=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(
                    embed_dim, 256, batch_first=True, bidirectional=True
                )
                self.fc = nn.Linear(512, num_classes)

            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                lstm_out, _ = self.lstm(x)
                # Use last hidden state
                pooled = lstm_out[:, -1, :]
                return self.fc(pooled)

        num_classes = self._get_num_classes()
        return SimpleTextClassifier(num_classes=num_classes)

    def _get_num_classes(self) -> int:
        """Get number of classes for dataset."""

        dataset_map = {
            "imdb": 2,
            "ag_news": 4,
            "yelp": 5,
            "amazon": 5,
            "sst2": 2,
            "mrpc": 2,
            "qqp": 2,
            "qnli": 2,
            "mnli": 3,
            "mnli-m": 3,
            "mnli-mm": 3,
            "cola": 2,
            "rte": 2,
            "wnli": 2,
            "sts-b": 1,
        }
        return dataset_map.get(self.dataset.lower(), 2)

    def load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load text classification dataset."""

        dataset_name = self.dataset.lower()

        if dataset_name == "imdb":
            return self._load_imdb()
        elif dataset_name == "ag_news":
            return self._load_ag_news()
        elif dataset_name == "yelp":
            return self._load_yelp()
        elif dataset_name == "amazon":
            return self._load_amazon()
        elif dataset_name == "sst2":
            return self._load_sst2()
        elif dataset_name == "mrpc":
            return self._load_mrpc()
        elif dataset_name == "qqp":
            return self._load_qqp()
        elif dataset_name == "qnli":
            return self._load_qnli()
        elif dataset_name in ["mnli", "mnli-m", "mnli-mm"]:
            return self._load_mnli()
        elif dataset_name == "cola":
            return self._load_cola()
        elif dataset_name == "rte":
            return self._load_rte()
        elif dataset_name == "wnli":
            return self._load_wnli()
        elif dataset_name == "sts-b":
            return self._load_sts_b()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_imdb(self) -> Tuple[DataLoader, DataLoader]:
        """Load IMDB dataset."""

        from torch.utils.data import TensorDataset

        # Dummy data - in practice would load actual IMDB dataset
        train_texts = torch.randint(0, 10000, (1000, 512))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 512))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_ag_news(self) -> Tuple[DataLoader, DataLoader]:
        """Load AG News dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 256))
        train_labels = torch.randint(0, 4, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 256))
        test_labels = torch.randint(0, 4, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_yelp(self) -> Tuple[DataLoader, DataLoader]:
        """Load Yelp dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 256))
        train_labels = torch.randint(0, 5, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 256))
        test_labels = torch.randint(0, 5, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_amazon(self) -> Tuple[DataLoader, DataLoader]:
        """Load Amazon dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 256))
        train_labels = torch.randint(0, 5, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 256))
        test_labels = torch.randint(0, 5, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_sst2(self) -> Tuple[DataLoader, DataLoader]:
        """Load SST-2 dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_mrpc(self) -> Tuple[DataLoader, DataLoader]:
        """Load MRPC dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_qqp(self) -> Tuple[DataLoader, DataLoader]:
        """Load QQP (Quora Question Pairs) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_qnli(self) -> Tuple[DataLoader, DataLoader]:
        """Load QNLI (Question-answering NLI) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_mnli(self) -> Tuple[DataLoader, DataLoader]:
        """Load MNLI (Multi-Genre NLI) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(
            0, 3, (1000,)
        )
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 3, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_cola(self) -> Tuple[DataLoader, DataLoader]:
        """Load CoLA (Corpus of Linguistic Acceptability) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_rte(self) -> Tuple[DataLoader, DataLoader]:
        """Load RTE (Recognizing Textual Entailment) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_wnli(self) -> Tuple[DataLoader, DataLoader]:
        """Load WNLI (Winograd NLI) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.randint(0, 2, (1000,))
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.randint(0, 2, (200,))
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_sts_b(self) -> Tuple[DataLoader, DataLoader]:
        """Load STS-B (Semantic Textual Similarity) dataset."""

        from torch.utils.data import TensorDataset

        train_texts = torch.randint(0, 10000, (1000, 128))
        train_labels = torch.rand(1000) * 5.0
        train_set = TensorDataset(train_texts, train_labels)

        test_texts = torch.randint(0, 10000, (200, 128))
        test_labels = torch.rand(200) * 5.0
        test_set = TensorDataset(test_texts, test_labels)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate classification accuracy."""

        model.eval()
        model.to(self.device)

        correct = 0
        total = 0
        batch_idx = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    input_ids, labels = batch[0], batch[1]
                else:
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                if isinstance(batch, dict) and "attention_mask" in batch:
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)

                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                elif isinstance(outputs, dict):
                    outputs = (
                        outputs["logits"] if "logits" in outputs else outputs["pred"]
                    )

                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_total = labels.size(0)
                total += batch_total
                correct += batch_correct

                if hasattr(self, 'writer') and self.writer is not None:
                    batch_accuracy = 100.0 * batch_correct / batch_total
                    self.writer.add_scalar('evaluation/batch_accuracy', batch_accuracy, batch_idx)
                    batch_idx += 1

        accuracy = 100.0 * correct / total
        return accuracy


class TextClassifier(nn.Module):
    """Wrapper for transformer models with classification head."""

    def __init__(self, base_model: nn.Module, num_classes: int, model_type: str = "encoder"):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        
        if self.model_type == "t5":
            last_hidden_state = outputs.last_hidden_state
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (last_hidden_state * attention_mask_expanded).sum(dim=1)
                sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = last_hidden_state.mean(dim=1)

        elif self.model_type == "decoder":
            last_hidden_state = outputs.last_hidden_state
            if attention_mask is not None:
                seq_lengths = (attention_mask.sum(dim=1) - 1).long()
                batch_size = last_hidden_state.size(0)
                pooled_output = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), 
                    seq_lengths
                ]
            else:
                pooled_output = last_hidden_state[:, -1, :]

        else:
            pooled_output = (
                outputs.pooler_output
                if hasattr(outputs, "pooler_output")
                else outputs.last_hidden_state[:, 0]
            )
        
        return self.classifier(pooled_output)
