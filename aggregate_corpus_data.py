"""Aggregate DCML corpus TSVs recursively from a given directory."""
import argparse
from pathlib import Path

from harmonic_inference.data.corpus_reading import aggregate_annotation_dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate DCML corpus TSVs recursively from a given directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("../corpora/annotations"),
        help=(
            "The directory containing the split corpus TSVs "
            "(in notes, measures, and harmonies sub-directories)."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("corpus_data"),
        help="The directory to write out the aggregated TSVs.",
    )

    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Include data for pieces without any annotated harmonies",
    )

    ARGS = parser.parse_args()

    aggregate_annotation_dfs(ARGS.input, ARGS.output, notes_only=ARGS.notes_only)
