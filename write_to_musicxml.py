"""A script that can be used to write an annotate.py or test.py output tsv to a MusicXML file."""
import argparse
from fractions import Fraction
from pathlib import Path
from typing import Union

from music21.converter import parse
from music21.harmony import ChordSymbol
from music21.stream import Stream
import pandas as pd

from harmonic_inference.data.piece import get_measures_df_from_music21_score


def get_offset(
    mc: int, mc_onset: Fraction, mn_onset: Fraction, measures_df: pd.DataFrame
) -> Fraction:
    # TODO


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
    labels_df = pd.read_csv(labels_tsv_path, sep="\t", index_col=0)

    m21_score: Stream = parse(music_xml_path)
    measures_df = get_measures_df_from_music21_score(m21_score)

    for label_row in labels_df:
        chord_symbol = ChordSymbol(label_row["label"])
        offset = get_offset(label_row["mc"], label_row["mc_onset"], label_row["mn_onset"], measures_df)
        m21_score.insert(offset, chord_symbol)

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
        help="The path to the MusicXML file to write the labels to.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="The path to the labels tsv file to write to the given score.",
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

    music_xml: Path = ARGS.x.absolute()
    labels: Path = ARGS.labels.absolute()
    if str(ARGS.o) == "*_chords":
        output = music_xml.parent / (
            music_xml.name.split(".")[0] + "_chords" + music_xml.name.split(".")[1]
        )
    else:
        output = ARGS.o.absolute()

    write_labels_to_score(music_xml, labels, output)
