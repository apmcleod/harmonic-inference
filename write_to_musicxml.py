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
            label = label.replace("7sus4", "sus4add7")
            chord_symbol = ChordSymbol(label)
        except ValueError:
            logging.error("Skipping unrecognized chord symbol: %s", label)
            continue

        measure: Measure = measures_list[label_row["mc"]]
        measure.insert(label_row["mc_onset"] * 4, chord_symbol)

    m21_score.write("musicxml", fp=output_path)


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

    ARGS = parser.parse_args()

    music_xml_arg: Path = ARGS.x
    if music_xml_arg.is_dir():
        all_music_xml = [
            Path(x)
            for x in
            sorted(
                glob(str(music_xml_arg / "**" / "*.mxl"), recursive=True) +
                glob(str(music_xml_arg / "**" / "*.xml"), recursive=True)
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
            labels = labels / music_xml.name.split(".")[0] + ".tsv"

    write_labels_to_score(music_xml, labels, output)
