"""A script that can be used to write annotate.py or test.py outputs to a musical score."""
import argparse
from pathlib import Path
from typing import Literal, Optional, Union

import ms3


def write_tsvs_to_scores(
    output_tsv_dir: Union[Path, str],
    annotations_base_dir: Union[Path, str],
    suffix: str = "_inferred",
    ask_for_input: bool = True,
    replace: bool = True,
    staff: Optional[Literal[1, 2, 3, 4]] = None,
    voice: Literal[1, 2, 3, 4] = None,
    harmony_layer: Optional[Literal[0, 1, 2]] = None,
    check_for_clashes: bool = True,
):
    """
    Write the labels TSVs from the given output directory onto annotated scores
    (from the annotations_base_dir) in the output directory.

    Parameters
    ----------
    output_tsv_dir : Union[Path, str]
        The path to TSV files containing labels to write onto annotated scores.
        The directory should contain sub-directories for each composer (aligned
        with sub-dirs in the annotations base directory), and a single TSV for each
        output.

    annotations_base_dir : Union[Path, str]
        The path to annotations and MuseScore3 scores, whose sub-directories and file names
        are aligned with those in the output TSV directory.

    suffix : str
        Suffix to be added to the newly created scores. Defaults to _inferred.

    ask_for_input : bool
        What to do if more than one TSV or MuseScore file is detected for a particular fname.
        By default, the user is asked for input.
        Pass False to prevent that and pick the files with the shortest relative paths instead.

    replace : bool
        By default, any existing labels are removed from the scores. Pass False to leave them in,
        which may lead to clashes.

    staff : Optional[int]
        If you pass a staff ID, the labels will be attached to that staff where 1 is the upper
        staff.
        By default, the staves indicated in the 'staff' column of the labels are used, or, if
        such a column is not present, labels will be inserted under the lowest staff -1.

    voice : Optional[Literal[1, 2, 3, 4]]
        If you pass the ID of a notational layer (where 1 is the upper voice, blue in MuseScore),
        the labels will be attached to that one.
        By default, the notational layers indicated in the 'voice' column of the labels are used,
        or, if such a column is not present, labels will be inserted for voice 1.

    harmony_layer : Optional[Literal[0, 1, 2]]
        | By default, the labels are written to the layer specified as an integer in the
          'harmony_layer' of the label. Pass an integer to select a particular layer:
        | * 0 to attach them as absolute ('guitar') chords, meaning that when opened next time,
        |   MuseScore will split and encode those beginning with a note name
        |   (resulting in ms3-internal harmony_layer 3).
        | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
        | * 2 to have MuseScore interpret them as Nashville Numbers

    check_for_clashes : bool
        By default, warnings are thrown when there already exists a label at a position
        (and in a notational layer) where a new one is attached. Pass False to deactivate these
        warnings.


    """
    output_tsv_dir = Path(output_tsv_dir)
    annotations_base_dir = Path(annotations_base_dir)
    outputs_are_under_corpus_path = (output_tsv_dir == annotations_base_dir) or (
        annotations_base_dir in annotations_base_dir.parents
    )
    corpus = ms3.Corpus(annotations_base_dir, only_metadata_fnames=False)
    if not outputs_are_under_corpus_path:
        # if the Corpus directory itself includes any labels we exclude them to avoid ambiguity
        if any(x.is_dir() and x.name == "labels" for x in annotations_base_dir.iterdir()):
            excluded_labels_path = annotations_base_dir / "labels"
            corpus.view.exclude("path", str(excluded_labels_path))
    _ = corpus.add_dir(
        output_tsv_dir,
        filter_other_fnames=True,
        file_re=r"\.tsv$",
        exclude_re="",
    )
    ms3.insert_labels_into_score(
        corpus,
        facet="labels",
        ask_for_input=ask_for_input,
        replace=replace,
        staff=staff,
        voice=voice,
        harmony_layer=harmony_layer,
        check_for_clashes=check_for_clashes,
        print_info=False,
    )
    ms3.store_scores(
        corpus,
        only_changed=True,
        root_dir=output_tsv_dir,
        folder=".",
        suffix=suffix,
        overwrite=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write functional harmony labels to musical scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help=(
            "The directory containing results.tsv model outputs, and to which annotated "
            "MuseScore3 scores to will be saved."
        ),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help=(
            "A directory containing corpora annotation tsvs and MuseScore3 scores, which "
            "will be used to write out labels onto new MuseScore3 score files in the "
            "--output directory."
        ),
    )

    ARGS = parser.parse_args()

    write_tsvs_to_scores(ARGS.output, ARGS.annotations)
