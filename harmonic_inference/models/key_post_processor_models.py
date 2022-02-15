"""Models that post-process chord classifications and assign each to a key."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.autograd import Variable

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

    def get_output(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def evaluate(self, dataset: KeyPostProcessorDataset):
        raise NotImplementedError()

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


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
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
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
