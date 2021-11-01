"""A script that can be used to write annotate.py or test.py outputs to a musical score."""
import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Union

from tqdm import tqdm

import harmonic_inference.utils.eval_utils as eu


def write_tsvs_to_scores(
    output_tsv_dir: Union[Path, str],
    annotations_base_dir: Union[Path, str],
    fuzzy: bool = False,
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

    fuzzy : bool
        If True, the annotations do not need to match exactly in sub-directory structure,
        only in file name.
    """
    output_tsv_dir = Path(output_tsv_dir)
    annotations_base_dir = Path(annotations_base_dir)

    output_paths = sorted(glob(str(output_tsv_dir / "**" / "*.tsv"), recursive=True))
    output_paths = [
        path
        for path in output_paths
        if not path.endswith("_results.tsv") and not path.endswith("results_midi.tsv")
    ]
    for piece_name in tqdm(output_paths, desc="Writing labels to scores"):
        piece_name = Path(piece_name).relative_to(output_tsv_dir)
        try:
            eu.write_labels_to_score(
                output_tsv_dir / piece_name.parent,
                annotations_base_dir if fuzzy else annotations_base_dir / piece_name.parent,
                piece_name.stem,
            )
        except Exception:
            logging.exception("Error writing score out to %s", output_tsv_dir / piece_name.parent)


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

    parser.add_argument(
        "-f",
        "--fuzzy",
        action="store_true",
        help=(
            "Do not require that the annotations sub-directory structure matches the output "
            "directory structure exactly. Rather, matches are made only by filename."
        ),
    )

    ARGS = parser.parse_args()

    write_tsvs_to_scores(ARGS.output, ARGS.annotations, fuzzy=ARGS.fuzzy)
