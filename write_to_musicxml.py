"""A script that can be used to write an annotate.py or test.py output tsv to a MusicXML file."""
import argparse
from fractions import Fraction
from glob import glob
import logging
from pathlib import Path
from typing import List, Union

from music21.converter import parse
from music21.harmony import ChordSymbol
from music21.stream import Measure, Stream
import pandas as pd


def write_labels_to_score(
    music_xml_path: Union[Path, str],
    labels_tsv_path: Union[Path, str],
    output_path: Union[Path, str],
):
    """
    Write the labels TSV onto the given MusicXML file, and save it in the given path.

    Parameters
    ----------
    music_xml_path : Union[Path, str]
        The path the MusicXML file on which to write the chord labels.

    labels_tsv_path : Union[Path, str]
        The tsv file containing the labels to be written to the score.

    output_path : Union[Path, str]
        The file to write the output MusicXML to.
    """
    logging.info("Writing labels to score for file: %s", str(music_xml_path))
    logging.info("    Using labels from file: %s", str(labels_tsv_path))
    logging.info("    Output being written to file: %s", str(output_path))

    labels_df = pd.read_csv(
        labels_tsv_path,
        sep="\t",
        index_col=0,
        converters={"mc": int, "mc_onset": Fraction, "mn_onset": Fraction, "label": str},
    )

    m21_score: Stream = parse(music_xml_path)
    measures_list: List[Measure] = list(m21_score.recurse().getElementsByClass(Measure))

    # Extract and remove all existing chord symbols
    existing_chord_symbols = []
    for element in m21_score.recurse().getElementsByClass(ChordSymbol):
        element.activeSite.remove(element)
        existing_chord_symbols.append(element)

    for _, label_row in labels_df.iterrows():
        if "Key" in label_row["label"]:
            # Skip key changes
            continue

        try:
            label = label_row["label"]
            label = label.replace("7sus2", "sus2add7")
            chord_symbol = ChordSymbol(label)
        except ValueError:
            logging.error("Skipping unrecognized chord symbol: %s", label)
            continue

        measure: Measure = measures_list[label_row["mc"]]
        measure.insert(label_row["mc_onset"] * 4, chord_symbol)

    m21_score.write("musicxml", fp=output_path)


# Slight hack in order to save some results along the way and not have to ask for
# the accuracy of the same symbol pair twice.
chord_accuracies = dict()


# JASON: COMMENTED OUT LABEL ACCURACY ESTIMATION (lines 78-208 + 287+288)

# def estimate_label_accuracy(
#     music_xml_path: Union[Path, str],
#     labels_tsv_path: Union[Path, str],
# ) -> float:
#     """
#     Estimate the accuracy of the given labels on the given MusicXML file.

#     Parameters
#     ----------
#     music_xml_path : Union[Path, str]
#         The path the MusicXML file containing ground truth labels.

#     labels_tsv_path : Union[Path, str]
#         The tsv file containing the estimated labels.

#     Returns
#     -------
#     float
#         The estimated accuracy of the given labels on the given MusicXML file,
#         as a number between 0 and 1.
#     """
#     def get_musicxml_chord_label_at_time(
#         mc: int, offset: Fraction, measures: List[Measure]
#     ) -> ChordSymbol:
#         """
#         Get the music21 ChordSymbol at a specific position in the score, or the most recent label,
#         if none is at that exact point in the score.

#         Parameters
#         ----------
#         mc : int
#             The mc number to find the Measure object we are looking for.
#         offset : Fraction
#             The offset, in quarter notes, of the score position we're looking for.
#         measures : List[Measure]
#             A List of the Measures in the music xml file.

#         Returns
#         -------
#         ChordSymbol
#             The chord symbol active at the given point in the score.
#         """
#         measure = measures[mc]
#         chord_symbols: List[ChordSymbol] = list(measure.getElementsByClass(ChordSymbol))
#         for chord_symbol in reversed(chord_symbols):
#             if chord_symbol.offset <= offset:
#                 return chord_symbol

#         # Here we need to find the most recent ChordSymbol instead
#         for measure_id in range(mc - 1, -1, -1):
#             measure = measures[measure_id]
#             chord_symbols: List[ChordSymbol] = list(measure.getElementsByClass(ChordSymbol))
#             if len(chord_symbols) > 0:
#                 return chord_symbols[-1]

#         logging.warning("No matching ChordSymbol found for mc=%s and offset=%s.", mc, offset)
#         return None

#     def get_chord_accuracy(score_label: ChordSymbol, pred_label: str) -> float:
#         """
#         Return an accuracy value for the given label given the ground truth symbol.

#         Parameters
#         ----------
#         score_label : ChordSymbol
#             The musicXML file's label.
#         pred_label : str
#             The model's predicted label.

#         Returns
#         -------
#         float
#             A measure of the accuracy of the label, between 0 (totally wrong) and 1
#             (totally correct).
#         """
#         global chord_accuracies

#         if score_label.figure == pred_label:
#             return 1.0

#         if score_label.figure.split("/")[0] == pred_label.split("/")[0]:
#             return 0.5

#         if score_label.figure.replace("7", "") == pred_label.replace("7", ""):
#             return 0.5

#         if (
#             score_label.figure in chord_accuracies
#             and pred_label in chord_accuracies[score_label.figure]
#         ):
#             return chord_accuracies[score_label.figure][pred_label]

#         chord_acc = float(input(f"What is the accuracy of {score_label.figure} and {pred_label}? "))

#         if score_label.figure not in chord_accuracies:
#             chord_accuracies[score_label.figure] = {pred_label: chord_acc}

#         return chord_acc

#     labels_df = pd.read_csv(
#         labels_tsv_path,
#         sep="\t",
#         index_col=0,
#         converters={"mc": int, "mc_onset": Fraction, "mn_onset": Fraction, "label": str},
#     )

#     m21_score: Stream = parse(music_xml_path)
#     measures_list: List[Measure] = list(m21_score.recurse().getElementsByClass(Measure))

#     total_labels = 0
#     total_accuracy = 0

#     for _, label_row in labels_df.iterrows():
#         if "Key" in label_row["label"]:
#             # Skip key changes
#             continue

#         total_labels += 1
#         corresponding_score_label = get_musicxml_chord_label_at_time(
#             label_row["mc"], label_row["mc_onset"] * 4, measures_list
#         )
#         if corresponding_score_label is None:
#             continue

#         chord_acc = get_chord_accuracy(corresponding_score_label, label_row["label"])
#         total_accuracy += chord_acc

#     if total_labels == 0:
#         return 0.0

#     return total_accuracy / total_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write chord labels to a musical score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-x",
        type=Path,
        required=True,
        help="The path to the MusicXML file to write the labels to, or a directory of such files.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=(
            "The path to the labels tsv file to write to the given score, or a directory in which "
            "the labels tsv files are stored, in which to search for the matching labels tsv file."
        ),
    )

    parser.add_argument(
        "-o",
        type=Path,
        required=False,
        default="*_chords",
        help=(
            "The path to write the resulting score to. If not given, this will be "
            "the input path plus `_chords` before the file extension."
        )
    )

    parser.add_argument(
        "--acc-only",
        action="store_true",
        help=(
            "If given, only estimate and print the accuracy of the given labels on the given "
            "score."
        ),
    )

    ARGS = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    music_xml_arg: Path = ARGS.x
    if music_xml_arg.is_dir():
        all_music_xml = [
            Path(x) for x in sorted(
                glob(str(music_xml_arg / "**" / "*.mxl"), recursive=True)
                + glob(str(music_xml_arg / "**" / "*.xml"), recursive=True)
            )
        ]
    else:
        all_music_xml = [music_xml_arg]

    for music_xml in all_music_xml:
        if str(ARGS.o) == "*_chords":
            output = music_xml.parent / (
                music_xml.name.split(".")[0] + "_chords." + music_xml.name.split(".")[1]
            )
        else:
            output = ARGS.o

        labels: Path = ARGS.labels
        if labels.is_dir():
            labels = labels / (str(music_xml.name.split(".")[0]) + ".tsv")

        if not labels.exists():
            logging.warning("No labels found for file %s", str(music_xml))
            continue

        if not ARGS.acc_only:
            write_labels_to_score(music_xml, labels, output)
        # accuracy = estimate_label_accuracy(music_xml, labels)
        # logging.info("Estimated accuracy for file %s: %.2f%%", str(music_xml), accuracy * 100)
