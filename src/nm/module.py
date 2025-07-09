"""
Module containing the ToxicModel LightningModule for multilabel text classification.

Implements a PyTorch LightningModule for toxic comment classification using a HuggingFace model and tokenizer.
"""

from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn.functional import binary_cross_entropy, sigmoid
from torchmetrics.classification import MultilabelAccuracy

from nm.config import ModelConfig
from nm.datamodule import JigsawDataModule
from nm.utils import get_model_and_tokenizer


class ToxicModel(pl.LightningModule):
    """A custom LightningModule for multilabel toxic comment classification.

    Args:
        model_name: The name of the model and accompanying tokenizer.
        num_classes: Number of output classes.
        max_token_len: Maximum length of tokenized input sequences.
        learning_rate: Learning rate for the optimizer.
        cache_dir: Directory to cache the model and tokenizer.
    """
    def __init__(
        self,
        model_name: str = ModelConfig.model_name,
        num_classes: int = ModelConfig.num_classes,
        max_token_len: int = ModelConfig.max_token_len,
        learning_rate: float = ModelConfig.learning_rate,
        cache_dir: str | None = ModelConfig.cache_dir,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_token_len = max_token_len
        self.learning_rate = learning_rate
        self.cache_dir = cache_dir

        self.model, self.tokenizer = get_model_and_tokenizer(
            model_name, num_classes, cache_dir
        )
        self.accuracy = MultilabelAccuracy(num_labels=num_classes)

    def forward(self, x: str | list[str]) -> Tensor:
        """Performs a forward pass through the model.

        Args:
            x: Input text or list of texts to classify.

        Returns:
            Tensor: Sigmoid-activated logits for each class.
        """
        x = self.tokenizer(
            x,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        y_hat = self.model(**x)
        y_hat = y_hat["logits"]
        return sigmoid(y_hat)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Runs a single training step.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training

        Args:
            batch: A tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Dict with key 'loss' containing the training loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Runs a single validation step.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation

        Args:
            batch: A tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Dict with keys 'loss' and 'acc' for validation loss and accuracy.
        """
        x, y = batch
        y_hat = self(x)
        loss = binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Runs a single test step.

        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#testing

        Args:
            batch: A tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Dict with keys 'loss' and 'acc' for test loss and accuracy.
        """
        x, y = batch
        y_hat = self(x)
        loss = binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # TODO: Implement predict_step


if __name__ == "__main__":
    dm = JigsawDataModule()
    model = ToxicModel()
    model.eval()
    with torch.no_grad():
        text = ["I hate you."]
        result = model(text)
        print(result)
