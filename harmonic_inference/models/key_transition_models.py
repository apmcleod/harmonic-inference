"""Models that output the probability of a key change occurring on a given input."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.chord import get_chord_vector_length
from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.datasets import KeyTransitionDataset
from harmonic_inference.data.key import get_key_change_vector_length


class KeyTransitionModel(pl.LightningModule, ABC):
    """
    The base class for all Key Transition Models which model when a key change will occur.
    """

    def __init__(
        self,
        input_chord_pitch_type: PitchType,
        input_key_pitch_type: PitchType,
        use_inversions: bool,
        reduction: Dict[ChordType, ChordType],
        learning_rate: float,
        input_mask: List[int],
    ):
        """
        Create a new base KeySequenceModel with the given output and input data types.

        Parameters
        ----------
        input_chord_pitch_type : PitchType
            The pitch representation used in the input chord data.
        input_key_pitch_type : PitchType
            The pitch representation used in the input key data.
        use_inversions : bool
            True to use inversions in the input vectors. False otherwise.
        reduction : Dict[ChordType, ChordType]
            The reduction to use for chord types.
        learning_rate : float
            The learning rate.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        """
        super().__init__()
        self.INPUT_CHORD_PITCH_TYPE = input_chord_pitch_type
        self.INPUT_KEY_PITCH_TYPE = input_key_pitch_type

        self.reduction = reduction
        self.use_inversions = use_inversions

        self.lr = learning_rate

        self.input_mask = input_mask

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """
        Get a kwargs dict that can be used to create a dataset for this model with
        the correct parameters.

        Returns
        -------
        dataset_kwargs : Dict[str, Any]
            A keyword args dict that can be used to create a dataset for this model with
            the correct parameters.
        """
        return {
            "reduction": self.reduction,
            "use_inversions": self.use_inversions,
            "input_mask": self.input_mask,
        }

    def get_data_from_batch(self, batch):
        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"]
        max_length = max(input_lengths)
        inputs = inputs[:, :max_length]

        targets = batch["targets"].float()[:, :max_length]

        mask = (targets != -100).bool()

        return inputs, input_lengths, targets, mask

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

        outputs, _ = self(inputs, input_lengths)

        flat_mask = mask.reshape(-1)
        outputs = outputs.reshape(-1)[flat_mask]
        targets = targets.reshape(-1)[flat_mask]

        loss = F.binary_cross_entropy(outputs, targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

        outputs, _ = self(inputs, input_lengths)

        flat_mask = mask.reshape(-1)
        outputs = outputs.reshape(-1)[flat_mask]
        targets = targets.reshape(-1)[flat_mask]

        loss = F.binary_cross_entropy(outputs, targets)

        acc = 100 * (outputs.round().long() == targets).sum().float() / len(outputs)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def evaluate(self, dataset: KeyTransitionDataset):
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        total = 0
        total_acc = 0
        total_loss = 0

        for batch in tqdm(dl, desc="Evaluating KTM"):
            inputs, input_lengths, targets, mask = self.get_data_from_batch(batch)

            outputs, _ = self(inputs, input_lengths)

            flat_mask = mask.reshape(-1)
            outputs = outputs.reshape(-1)[flat_mask]
            targets = targets.reshape(-1)[flat_mask]

            batch_count = len(outputs)
            loss = F.binary_cross_entropy(outputs, targets)
            acc = 100 * (outputs.round().long() == targets).sum().float() / len(outputs)

            total += batch_count
            total_acc += acc * batch_count
            total_loss += loss * batch_count

        return {
            "acc": (total_acc / total).item(),
            "loss": (total_loss / total).item(),
        }

    @abstractmethod
    def init_hidden(self, batch_size: int) -> Tuple[Variable, ...]:
        """
        Get initial hidden layers for this model.

        Parameters
        ----------
        batch_size : int
            The batch size to initialize hidden layers for.

        Returns
        -------
        hidden : Tuple[Variable, ...]
            A tuple of initialized hidden layers.
        """
        raise NotImplementedError()

    def run_one_step(self, batch):
        inputs = batch["inputs"].float()
        hidden = (
            torch.transpose(batch["hidden_states"][0], 0, 1),
            torch.transpose(batch["hidden_states"][1], 0, 1),
        )

        return self(inputs, torch.ones(len(inputs)), hidden=hidden)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


class SimpleKeyTransitionModel(KeyTransitionModel):
    """
    The most simple key transition model, with layers:
        1. Linear embedding layer
        2. LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer
    """

    def __init__(
        self,
        input_chord_pitch_type: PitchType,
        input_key_pitch_type: PitchType,
        embed_dim: int = 128,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        use_inversions: bool = True,
        reduction: Dict[ChordType, ChordType] = None,
        input_mask: List[int] = None,
    ):
        """
        Create a new simple key transition model.

        Parameters
        ----------
        input_chord_pitch_type : PitchType
            The pitch representation used in the input chord data.
        input_key_pitch_type : PitchType
            The pitch representation used in the input key data.
        embed_dim : int
            The size of the input embedding.
        lstm_layers : int
            The number of bi-directional LSTM layers.
        lstm_hidden_dim : int
            The size of the LSTM's hidden dimension.
        hidden_dim : int
            The size of the hidden dimension between the 2 consecutive linear layers.
        dropout : float
            The dropout proportion.
        learning_rate : float
            The learning rate.
        use_inversions : bool
            True to use inversions in the input vectors. False otherwise.
        reduction : Dict[ChordType, ChordType]
            The reduction to use for chord types.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        """
        super().__init__(
            input_chord_pitch_type,
            input_key_pitch_type,
            use_inversions,
            reduction,
            learning_rate,
            input_mask,
        )
        self.save_hyperparameters()

        # Input derived from input type
        self.input_dim = (
            get_chord_vector_length(
                input_chord_pitch_type,
                one_hot=False,
                relative=True,
                use_inversions=use_inversions,
                pad=True,
                reduction=reduction,
            )
            + get_key_change_vector_length(input_key_pitch_type, one_hot=False)
            + 1
        )

        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.input_dim, self.embed_dim)

        # LSTM hidden layer and depth
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=False,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.dropout1 = nn.Dropout(self.dropout)

    def init_hidden(self, batch_size: int) -> Tuple[Variable, Variable]:
        """
        Initialize the LSTM's hidden layer for a given batch size.

        Parameters
        ----------
        batch_size : int
            The batch size.
        """
        return (
            Variable(
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device)
            ),
            Variable(
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device)
            ),
        )

    def forward(self, inputs, lengths, hidden=None):
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1).cpu()
        h_0, c_0 = self.init_hidden(batch_size) if hidden is None else hidden

        embedded = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed, (h_0, c_0))
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = torch.squeeze(self.fc2(relu2), -1)

        return torch.sigmoid(output), hidden_out
