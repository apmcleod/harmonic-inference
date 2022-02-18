"""Models that post-process chord classifications and assign each to a key."""
from abc import ABC, abstractmethod
from math import exp
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.chord import get_chord_vector_length
from harmonic_inference.data.data_types import ChordType, PitchType
from harmonic_inference.data.datasets import KeyPostProcessorDataset
from harmonic_inference.utils.harmonic_utils import get_key_label_list


class KeyPostProcessorModel(pl.LightningModule, ABC):
    """
    The base type for all Key Post-Processor Models, which take as input sequences of chord
    classifications, and assign each a key.
    """

    def __init__(
        self,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType],
        use_inversions: bool,
        learning_rate: float,
        scheduled_sampling: bool,
        sigmoid_k: float,
        transposition_range: Union[List[int], Tuple[int, int]],
        input_mask: List[int],
    ):
        """
        Create a new base KeyPostProcessorModel with the given input and output formats.

        Parameters
        ----------
        input_pitch : PitchType
            What pitch type the model is expecting for chord roots.
        output_pitch : PitchType
            The pitch type to use for output tonics of this model.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the input chord types.
        use_inversions : bool
            Whether inversions are used in the chord inputs.
        learning_rate : float
            The learning rate.
        scheduled_sampling : bool
            Whether to use scheduled sampling or not.
        sigmoid_k : float
            The value to use for k, in calculation of the clean probability during
            scheduled sampling.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each chord and target key of the
            dataset. Each __getitem__ call will return every possible transposition in this
            (min, max) range, inclusive on each side. The transpositions are measured in
            whatever PitchType is used in the dataset.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        """
        super().__init__()
        self.INPUT_PITCH = input_pitch
        self.OUTPUT_PITCH = output_pitch

        self.reduction = reduction
        self.use_inversions = use_inversions
        self.transposition_range = transposition_range

        self.lr = learning_rate
        self.scheduled_sampling = scheduled_sampling
        self.sigmoid_k = sigmoid_k

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
            "transposition_range": self.transposition_range,
            "input_mask": self.input_mask,
        }

    def load_scheduled_sampling_inputs(self, batch: Dict[str, Any]) -> None:
        """
        Load scheduled sampling data *in place* in the given batch dictionary.
        The resulting input will be placed in batch["input"], overwriting whatever
        is already there. If the batch dictionary has no "scheduled_sampling_data"
        key, the "input" is returned unchanged.

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch data, mapping strings to tensrs.
        """

        def get_scheduled_sampling_prob(epoch_num: int, sigmoid_k: float = 10.0) -> float:
            """
            Get the probability of using the clean, ground truth input given an epoch number
            and the saturation epoch number

            Parameters
            ----------
            epoch_num : int
                The current epoch number.
            sigmoid_k : float
                The value of k to use in the function clean_prob = k / (k + exp(epoch_num / k)).

            Returns
            -------
            clean_prob : float
                The probability of using the clean, ground truth input for each data point.
            """
            return sigmoid_k / (sigmoid_k + exp(epoch_num / sigmoid_k))

        clean_prob = get_scheduled_sampling_prob(self.current_epoch, sigmoid_k=self.sigmoid_k)
        random_sample = torch.bernoulli(
            torch.full((torch.sum(batch["input_lengths"].clip(0)).item(),), 1 - clean_prob)
        ).type(torch.bool)

        start = 0
        for i, length in enumerate(batch["input_lengths"]):
            if length <= 0:
                continue
            sample = random_sample[start : start + length]
            if torch.sum(sample).item() != 0:
                batch["inputs"][i][:length][sample] = batch["scheduled_sampling_data"][i][:length][
                    sample
                ]

            start += length

    def get_data_from_batch(self, batch, sched):
        if sched:
            self.load_scheduled_sampling_inputs(batch)

        inputs = batch["inputs"].float()
        input_lengths = batch["input_lengths"].long()

        longest = max(input_lengths)
        inputs = inputs[:, :longest]

        targets = None
        if "targets" in batch:
            targets = batch["targets"].long()
            targets = targets[:, :longest]

            for i, length in enumerate(input_lengths):
                targets[i, length:] = -100

        return inputs, input_lengths, targets

    def get_output(self, batch: Dict[str, Any]):
        inputs, input_lengths, _ = self.get_data_from_batch(batch, False)

        return self(inputs, input_lengths)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        inputs, input_lengths, targets = self.get_data_from_batch(batch, self.scheduled_sampling)

        outputs = self(inputs, input_lengths)

        loss = F.nll_loss(outputs.permute(0, 2, 1), targets, ignore_index=-100)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        inputs, input_lengths, targets = self.get_data_from_batch(batch, self.scheduled_sampling)

        outputs = self(inputs, input_lengths)

        loss = F.nll_loss(outputs.permute(0, 2, 1), targets, ignore_index=-100)

        targets = targets.reshape(-1)
        mask = targets != -100
        outputs = outputs.argmax(-1).reshape(-1)[mask]
        targets = targets[mask]
        acc = 100 * (outputs == targets).sum().float() / len(targets)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def evaluate(self, dataset: KeyPostProcessorDataset):
        data_loader = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        total = 0
        total_loss = 0
        total_acc = 0

        for batch in tqdm(data_loader, desc="Evaluating KPPM"):
            inputs, input_lengths, targets = self.get_data_from_batch(batch, False)

            outputs = self(inputs, input_lengths)

            loss = F.nll_loss(outputs.permute(0, 2, 1), targets, ignore_index=-100)

            targets = targets.reshape(-1)
            mask = targets != -100
            outputs = outputs.argmax(-1).reshape(-1)[mask]
            targets = targets[mask]
            batch_count = len(targets)
            acc = 100 * (outputs == targets).sum().float() / len(targets)

            total += batch_count
            total_loss += loss * batch_count
            total_acc += acc * batch_count

        return {
            "acc": (total_acc / total).item(),
            "loss": (total_loss / total).item(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

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


class SimpleKeyPostProcessorModel(KeyPostProcessorModel):
    """
    The simplest key post-processing model, with layers:
    1. Linear embedding layer
    2. Bi-LSTM
    3. Linear layer
    4. Dropout
    5. Linear layer
    """

    def __init__(
        self,
        input_pitch_type: PitchType,
        output_pitch_type: PitchType,
        transposition_range: Union[List[int], Tuple[int, int]] = (0, 0),
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        embed_dim: int = 64,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        scheduled_sampling: bool = False,
        sigmoid_k: float = 10,
        input_mask: List[int] = None,
    ):
        """
        Create a new simple chord sequence model.

        Parameters
        ----------
        input_pitch_type : PitchType
            The type of pitch representation for the input chord root pitches.
        output_pitch_type : PitchType
            The type of pitch representation for the target chord root pitches.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each note and chord of the
            dataset. Each __getitem__ call will return every possible transposition in this
            (min, max) range, inclusive on each side. The transpositions are measured in
            whatever PitchType is used in the dataset.
        reduction : Dict[ChordType, ChordType]
            The reduction used for input vector chord types.
        use_inversions : bool
            True to take inversions into account for the input. False to ignore them.
            False to ignore them.
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
        scheduled_sampling : bool
            Whether to use scheduled sampling or not.
        sigmoid_k : float
            The value to use for k in the caluclation of the clean probability during
            scheduled sampling.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        """
        super().__init__(
            input_pitch_type,
            output_pitch_type,
            reduction,
            use_inversions,
            learning_rate,
            scheduled_sampling,
            sigmoid_k,
            transposition_range,
            input_mask,
        )
        self.save_hyperparameters()

        # Input and output derived from input type and use_inversions
        self.input_dim = get_chord_vector_length(
            input_pitch_type,
            one_hot=False,
            relative=False,
            use_inversions=use_inversions,
            pad=False,
            reduction=reduction,
        )

        self.output_dim = len(get_key_label_list(output_pitch_type, relative=False))

        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.input_dim, self.embed_dim)

        # LSTM hidden layer and depth
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
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
                torch.zeros(
                    2 * self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device
                )
            ),
            Variable(
                torch.zeros(
                    2 * self.lstm_layers, batch_size, self.lstm_hidden_dim, device=self.device
                )
            ),
        )

    def forward(self, inputs, lengths):
        # pylint: disable=arguments-differ
        batch_size = inputs.shape[0]
        lengths = torch.clamp(lengths, min=1).cpu()
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, _ = self.lstm(packed, (h_0, c_0))
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        relu1 = F.relu(lstm_out)
        drop1 = self.dropout1(relu1)
        fc1 = self.fc1(drop1)
        relu2 = F.relu(fc1)
        output = self.fc2(relu2)

        return F.log_softmax(output, dim=2)
