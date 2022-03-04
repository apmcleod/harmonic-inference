"""Models that generate probability distributions over chord classifications of a given input."""
from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from h5py import File
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from harmonic_inference.data.chord import get_chord_vector_length
from harmonic_inference.data.data_types import ChordType, PieceType, PitchType
from harmonic_inference.data.datasets import ChordClassificationDataset
from harmonic_inference.data.note import get_note_vector_length
from harmonic_inference.utils.harmonic_constants import NUM_PITCHES


class ChordClassifierModel(pl.LightningModule, ABC):
    """
    The base type for all Chord Classifier Models, which take as input sets of frames from Pieces,
    and output chord probabilities for them.
    """

    def __init__(
        self,
        input_type: PieceType,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType],
        use_inversions: bool,
        learning_rate: float,
        transposition_range: Union[List[int], Tuple[int, int]],
        input_mask: List[int],
    ):
        """
        Create a new base ChordClassifierModel with the given input and output formats.

        Parameters
        ----------
        input_type : PieceType
            The type of piece that the input data is coming from.
        input_pitch : PitchType
            What pitch type the model is expecting for notes.
        output_pitch : PitchType
            The pitch type to use for outputs of this model.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the output chord types.
        use_inversions : bool
            Whether to use different inversions as different chords in the output.
        learning_rate : float
            The learning rate.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each note and chord of the
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
        self.INPUT_TYPE = input_type
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

    def add_settings_to_h5_file(self, h5_file: File) -> None:
        """
        Add this model's hyperparameter settings to the given h5 file. This is specifically used
        when this model has generated scheduled sampling data.

        Parameters
        ----------
        h5_file : File
            An h5 file, which should already be open for writing.
        """
        if self.reduction is not None:
            h5_file.create_dataset(
                "reduction",
                data=np.array([x.value, y.value] for x, y in self.reduction.items()),
                compression="gzip",
            )

        h5_file.create_dataset(
            "use_inversions", data=np.array([self.use_inversions]), compression="gzip"
        )
        h5_file.create_dataset(
            "pitch_type", data=np.array([self.OUTPUT_PITCH.value]), compression="gzip"
        )

    def get_output(self, batch):
        notes = batch["inputs"].float()
        notes_lengths = batch["input_lengths"]

        outputs = self(notes, notes_lengths)

        return F.softmax(outputs, dim=-1)

    def training_step(self, batch, batch_idx):
        notes = batch["inputs"].float()
        notes_lengths = batch["input_lengths"]
        targets = batch["targets"].long()

        outputs = self(notes, notes_lengths)
        loss = F.cross_entropy(outputs, targets, ignore_index=-1)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        notes = batch["inputs"].float()
        notes_lengths = batch["input_lengths"]
        targets = batch["targets"].long()

        outputs = self(notes, notes_lengths)

        mask = targets != -1
        outputs = outputs[mask]
        targets = targets[mask]

        if len(targets) > 0:
            acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)
            loss = F.cross_entropy(outputs, targets, ignore_index=-1)

            self.log("val_loss", loss)
            self.log("val_acc", acc)

    def evaluate(self, dataset: ChordClassificationDataset):
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        total = 0
        total_loss = 0
        total_acc = 0

        for batch in tqdm(dl, desc="Evaluating CCM"):
            notes = batch["inputs"].float()
            notes_lengths = batch["input_lengths"]
            targets = batch["targets"].long()

            batch_count = len(targets)
            outputs = self(notes, notes_lengths)
            loss = F.cross_entropy(outputs, targets)
            acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)

            total += batch_count
            total_loss += loss * batch_count
            total_acc += acc * batch_count

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


class SimpleChordClassifier(ChordClassifierModel):
    """
    The most simple chord classifier, with layers:
        1. Linear layer (embedding)
        2. Bi-LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer
    """

    def __init__(
        self,
        input_type: PieceType,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        transposition_range: Union[List[int], Tuple[int, int]] = (0, 0),
        embed_dim: int = 128,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        input_mask: List[int] = None,
    ):
        """
        Create a new SimpleChordClassifier.

        Parameters
        ----------
        input_type : PieceType
            The type of piece that the input data is coming from.
        input_pitch : PitchType
            What pitch type the model is expecting for notes.
        output_pitch : PitchType
            The pitch type to use for outputs of this model. Used to derive the output length.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the output chord types.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each note and chord of the
            dataset. Each __getitem__ call will return every possible transposition in this
            (min, max) range, inclusive on each side. The transpositions are measured in
            whatever PitchType is used in the dataset.
        use_inversions : bool
            Whether to use different inversions as different chords in the output. Used to
            derive the output length.
        embed_dim : int
            The size of the initial embedding layer.
        lstm_layers : int
            The number of Bi-LSTM layers to use.
        lstm_hidden_dim : int
            The size of each LSTM layer's hidden vector.
        hidden_dim : int
            The size of the output vector of the first linear layer.
        dropout : float
            The dropout proportion of the first linear layer's output.
        learning_rate : float
            The learning rate.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        """
        super().__init__(
            input_type,
            input_pitch,
            output_pitch,
            reduction,
            use_inversions,
            learning_rate,
            transposition_range,
            input_mask,
        )
        self.save_hyperparameters()

        # Input and output derived from pitch_type and use_inversions
        self.input_dim = get_note_vector_length(input_pitch)
        self.num_classes = get_chord_vector_length(
            output_pitch,
            one_hot=True,
            relative=False,
            use_inversions=use_inversions,
            pad=False,
            reduction=reduction,
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
            bidirectional=True,
            batch_first=True,
        )

        # Linear layers post-LSTM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 because bi-directional
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
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

    def forward(self, notes, lengths):
        # pylint: disable=arguments-differ
        batch_size = notes.shape[0]
        lengths = torch.clamp(lengths, min=1).cpu()
        h_0, c_0 = self.init_hidden(batch_size)

        embed = F.relu(self.embed(notes))

        packed_notes = pack_padded_sequence(embed, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed_notes, (h_0, c_0))
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)

        # Get lengths in proper format
        lstm_out_lengths_tensor = (
            lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand((-1, 1, lstm_out_forward.shape[2]))
        ).to(self.device)
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze(dim=1)
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)

        relu1 = F.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = F.relu(fc1)
        drop1 = self.dropout1(relu2)
        output = self.fc2(drop1)

        return output


class MultiTargetChordClassifier(ChordClassifierModel):
    """
    A simple chord classifier, with intermediate targets, using layers:
        1. Linear layer (embedding)
        2. Bi-LSTM
        3. Linear layer
        4. Dropout
        5. Linear layer with intermediate targets
        6. Linear output layer
    """

    def __init__(
        self,
        input_type: PieceType,
        input_pitch: PitchType,
        output_pitch: PitchType,
        reduction: Dict[ChordType, ChordType] = None,
        use_inversions: bool = True,
        transposition_range: Union[List[int], Tuple[int, int]] = (0, 0),
        embed_dim: int = 128,
        lstm_layers: int = 1,
        lstm_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        input_mask: List[int] = None,
    ):
        """
        Create a new MultiTargetChordClassifier.

        Parameters
        ----------
        input_type : PieceType
            The type of piece that the input data is coming from.
        input_pitch : PitchType
            What pitch type the model is expecting for notes.
        output_pitch : PitchType
            The pitch type to use for outputs of this model. Used to derive the output length.
        reduction : Dict[ChordType, ChordType]
            The reduction used for the output chord types.
        transposition_range : Union[List[int], Tuple[int, int]]
            Minimum and maximum bounds by which to transpose each note and chord of the
            dataset. Each __getitem__ call will return every possible transposition in this
            (min, max) range, inclusive on each side. The transpositions are measured in
            whatever PitchType is used in the dataset.
        use_inversions : bool
            Whether to use different inversions as different chords in the output. Used to
            derive the output length.
        embed_dim : int
            The size of the initial embedding layer.
        lstm_layers : int
            The number of Bi-LSTM layers to use.
        lstm_hidden_dim : int
            The size of each LSTM layer's hidden vector.
        hidden_dim : int
            The size of the output vector of the first linear layer.
        dropout : float
            The dropout proportion of the first linear layer's output.
        learning_rate : float
            The learning rate.
        input_mask : List[int]
            A binary input mask which is 1 in every location where each input vector
            should be left unchanged, and 0 elsewhere where the input vectors should
            be masked to 0. Essentially, if given, each input vector is multiplied
            by this mask in the Dataset code.
        """
        super().__init__(
            input_type,
            input_pitch,
            output_pitch,
            reduction,
            use_inversions,
            learning_rate,
            transposition_range,
            input_mask,
        )
        self.save_hyperparameters()

        # Input and output derived from pitch_type and use_inversions
        self.input_dim = get_note_vector_length(input_pitch)
        self.num_classes = get_chord_vector_length(
            output_pitch,
            one_hot=True,
            relative=False,
            use_inversions=use_inversions,
            pad=False,
            reduction=reduction,
        )

        # Intermediate target dim
        self.int_target_dim = 3 * NUM_PITCHES[output_pitch]  # root, bass, and pitch presence

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
        self.fc1 = nn.Linear(2 * self.lstm_hidden_dim, self.hidden_dim)  # 2 because bi-directional

        self.int_target_fc = nn.Linear(self.hidden_dim, self.int_target_dim)

        self.fc2 = nn.Linear(self.int_target_dim, self.num_classes)
        self.dropout1 = nn.Dropout(self.dropout)

    def get_output(self, batch):
        notes = batch["inputs"].float()
        notes_lengths = batch["input_lengths"]

        outputs, _ = self(notes, notes_lengths)

        return F.softmax(outputs, dim=-1)

    def training_step(self, batch, batch_idx):
        notes = batch["inputs"].float()
        notes_lengths = batch["input_lengths"]

        root_targets = batch["intermediate_targets"]["root"]
        bass_targets = batch["intermediate_targets"]["bass"]
        pitch_targets = batch["intermediate_targets"]["pitches"].float()
        targets = batch["targets"].long()

        outputs, (root, bass, presence) = self(notes, notes_lengths)
        output_loss = F.cross_entropy(outputs, targets, ignore_index=-1)

        root_loss = F.nll_loss(root, root_targets, ignore_index=-1)
        bass_loss = F.nll_loss(bass, bass_targets, ignore_index=-1)

        mask = pitch_targets != -1
        presence_loss = F.binary_cross_entropy(presence[mask], pitch_targets[mask])

        loss = output_loss + root_loss + bass_loss + presence_loss

        self.log("train_output_loss", output_loss)
        self.log("train_root_loss", root_loss)
        self.log("train_bass_loss", bass_loss)
        self.log("train_presence_loss", presence_loss)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        notes = batch["inputs"].float()
        notes_lengths = batch["input_lengths"]

        root_targets = batch["intermediate_targets"]["root"]
        bass_targets = batch["intermediate_targets"]["bass"]
        pitch_targets = batch["intermediate_targets"]["pitches"].float()
        targets = batch["targets"].long()

        outputs, (root, bass, presence) = self(notes, notes_lengths)

        mask = targets != -1
        outputs = outputs[mask]
        targets = targets[mask]

        if len(targets) > 0:
            output_acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)
            output_loss = F.cross_entropy(outputs, targets, ignore_index=-1)

            root_loss = F.nll_loss(root, root_targets, ignore_index=-1)
            bass_loss = F.nll_loss(bass, bass_targets, ignore_index=-1)

            mask = root_targets != -1
            presence_loss = F.binary_cross_entropy(presence[mask], pitch_targets[mask])

            loss = output_loss + root_loss + bass_loss + presence_loss

            root_acc = (
                100 * (root[mask].argmax(-1) == root_targets[mask]).sum().float() / len(targets)
            )
            bass_acc = (
                100 * (bass[mask].argmax(-1) == bass_targets[mask]).sum().float() / len(targets)
            )
            presence_acc = (
                100
                * (presence[mask].round() == pitch_targets[mask]).sum().float()
                / len(pitch_targets[mask].flatten())
            )

            self.log("val_output_acc", output_acc)
            self.log("val_root_acc", root_acc)
            self.log("val_bass_acc", bass_acc)
            self.log("val_presence_acc", presence_acc)

            self.log("val_output_loss", output_loss)
            self.log("val_root_loss", root_loss)
            self.log("val_bass_loss", bass_loss)
            self.log("val_presence_loss", presence_loss)
            self.log("val_loss", loss)

    def evaluate(self, dataset: ChordClassificationDataset):
        dl = DataLoader(dataset, batch_size=dataset.valid_batch_size)

        total = 0
        total_loss = 0
        total_acc = 0

        for batch in tqdm(dl, desc="Evaluating CCM"):
            notes = batch["inputs"].float()
            notes_lengths = batch["input_lengths"]
            targets = batch["targets"].long()

            batch_count = len(targets)
            outputs, (root, bass, presence) = self(notes, notes_lengths)
            loss = F.cross_entropy(outputs, targets)
            acc = 100 * (outputs.argmax(-1) == targets).sum().float() / len(targets)

            total += batch_count
            total_loss += loss * batch_count
            total_acc += acc * batch_count

        return {
            "acc": (total_acc / total).item(),
            "loss": (total_loss / total).item(),
        }

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

    def forward(self, notes, lengths):
        # pylint: disable=arguments-differ
        batch_size = notes.shape[0]
        lengths = torch.clamp(lengths, min=1).cpu()
        h_0, c_0 = self.init_hidden(batch_size)

        embed = F.relu(self.embed(notes))

        packed_notes = pack_padded_sequence(embed, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, (_, _) = self.lstm(packed_notes, (h_0, c_0))
        lstm_out_unpacked, lstm_out_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Reshape lstm outs
        lstm_out_forward, lstm_out_backward = torch.chunk(lstm_out_unpacked, 2, 2)

        # Get lengths in proper format
        lstm_out_lengths_tensor = (
            lstm_out_lengths.unsqueeze(1).unsqueeze(2).expand((-1, 1, lstm_out_forward.shape[2]))
        ).to(self.device)
        last_forward = torch.gather(lstm_out_forward, 1, lstm_out_lengths_tensor - 1).squeeze()
        last_backward = lstm_out_backward[:, 0, :]
        lstm_out = torch.cat((last_forward, last_backward), 1)

        relu1 = F.relu(lstm_out)
        fc1 = self.fc1(relu1)
        relu2 = F.relu(fc1)
        drop1 = self.dropout1(relu2)

        int_targets_raw = self.int_target_fc(drop1)  # Batch x int_targets_dim
        # Split values into root, bass, presence
        root, bass, presence = torch.split(int_targets_raw, self.int_target_dim // 3, dim=-1)
        root = F.softmax(root, dim=1)
        bass = F.softmax(bass, dim=1)
        presence = torch.sigmoid(presence)  # Multi-hot

        int_targets_encoded = torch.cat([root, bass, presence], dim=-1)
        output = self.fc2(int_targets_encoded)

        return output, (root, bass, presence)


class TranspositionInvariantCNNClassifier(nn.Module):
    """
    A transposition invariant CNN takes as input some (batch x num_input_channels x
    pitch_vector_length) matrix and classifies it in a transpositional invariant way.

    The last dimension should go along some representation of "pitches" such that a circular
    convolution along this dimension will represent transpositions of the input representation.
    The output channels of the convolutional layer are then fed into identical copies of the same
    feed-forward network.

    Parameters
    ----------
    num_chord_types : int
        The number of chord types for the network to output per root.

    num_hidden : int
        The number of hidden layers to use.

    hidden_size : int
        The number of nodes in the input layer and each hidden layer.

    batch_norm : boolean
        True to include batch normalization after the activation function of
        the input layer and each hidden layer.

    dropout : float
        The percentage of nodes in the input layer and each hidden layer to
        dropout. This is applied after activation (and before batch normalization
        if batch_norm is True, although it is not recommended to use both).
    """

    def __init__(
        self,
        num_chord_types,
        num_input_channels=1,
        pitch_vector_length=12,
        num_conv_channels=10,
        num_hidden=1,
        hidden_size=100,
        batch_norm=False,
        dropout=0.0,
    ):
        super().__init__()

        # Convolutional layer
        self.num_input_channels = num_input_channels
        self.pitch_vector_length = pitch_vector_length
        self.num_conv_channels = num_conv_channels

        self.conv = nn.Conv1d(
            self.num_input_channels,
            self.num_conv_channels,
            self.pitch_vector_length,
            padding=self.pitch_vector_length,
            padding_mode="circular",
        )

        # Parallel linear layers
        self.num_chord_types = num_chord_types
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.input = nn.Linear(num_conv_channels, hidden_size)
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(num_hidden)]
        )
        self.output = nn.Linear(hidden_size, num_chord_types)

        if batch_norm:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_size) for i in range(num_hidden + 1)]
            )
        else:
            self.batch_norms = nn.ModuleList([None] * (num_hidden + 1))

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_hidden + 1)])

    def forward(self, data):
        # Conv layer
        conv = F.relu(self.conv(data.unsqueeze(1)))

        # Parallel linear layers
        parallel_in = conv.reshape(conv.shape[0] * 12, -1)

        # Input layer
        parallel = self.dropouts[0](F.relu(self.input(parallel_in)))
        if self.batch_norms[0] is not None:
            parallel = self.batch_norms[0](parallel)

        # Hidden layers
        for layer, dropout, batch_norm in zip(self.linear, self.dropouts[1:], self.batch_norms[1:]):
            parallel = dropout(F.relu(layer(parallel)))
            if batch_norm is not None:
                parallel = batch_norm(parallel)

        # Output layer
        parallel_out = F.relu(self.output(parallel))

        # Final output combination
        output = parallel_out.reshape(parallel_out.shape[0] / 12, -1)
        return output


class TransformerEncoder(nn.Module):
    """
    This model encodes a given input into a defined chord representation.

    Parameters
    ----------
    """

    def __init__(self):
        super().__init__()
        self.todo = True

    def forward(self, data):
        # pylint: disable=arguments-differ
        pass


class MusicScoreJointModel(nn.Module):
    """
    This model is a combination of an chord encoder (e.g., TransformerEncoder) and a
    chord classifier (e.g., TranspositionInvariantCNNClassifier). The output of the encoder is
    fed into the classifier.

    Parameters
    ----------
    encoder : nn.Module
        The chord encoder model.

    classifier : nn.Module
        The chord classifier model.
    """

    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, data: torch.tensor, stages: Collection[int]) -> torch.tensor:
        # pylint: disable=arguments-differ
        """
        Forward pass one or both modules.

        Parameters
        ----------
        data : torch.tensor
            A batch-first representation of the input data for the forward pass.

        stages : list
            A list of what stages to perform. If 0 is in the list, use the encoder.
            If 1 is in the list, use the classifier.
        """
        if 0 in stages:
            data = self.encoder.forward(data)
        if 1 in stages:
            data = self.classifier.forward(data)

        return data
