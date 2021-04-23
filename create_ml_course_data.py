import logging
import shutil
from pathlib import Path

from tqdm import tqdm

import harmonic_inference.utils.harmonic_utils as hu
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.data_types import TRIAD_REDUCTION, KeyMode, PitchType
from harmonic_inference.data.piece import get_score_piece_from_data_frames

logging.basicConfig(level=logging.DEBUG)

files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs("corpus_data")

composers = sorted(set(name.split("-")[0].strip() for name in files_df.corpus_name.unique()))

base_dir = Path("ML_course_data")

chord_dir = Path(base_dir / "chord")
shutil.rmtree(chord_dir, ignore_errors=True)
chord_dir.mkdir(parents=True, exist_ok=True)

key_dir = Path(base_dir / "key")
shutil.rmtree(key_dir, ignore_errors=True)
key_dir.mkdir(parents=True, exist_ok=True)

chord_reduction = TRIAD_REDUCTION
use_inversions = False
use_relative = False

chord_labels = {KeyMode.MAJOR: set(), KeyMode.MINOR: set()}
key_labels = {KeyMode.MAJOR: set(), KeyMode.MINOR: set()}

for composer in composers:
    # startswith instead of contains because of WFBach/Bach
    composer_df = files_df.loc[files_df["corpus_name"].str.startswith(composer)]

    for file_id, file_row in tqdm(
        composer_df.iterrows(),
        desc=f"Creating {composer} data",
        total=len(composer_df),
    ):
        try:
            piece = get_score_piece_from_data_frames(
                notes_df.loc[file_id],
                chords_df.loc[file_id],
                measures_df.loc[file_id],
                chord_reduction=chord_reduction,
                use_inversions=use_inversions,
                use_relative=use_relative,
                name=f"{file_id}: {file_row['corpus_name']}",
            )
        except Exception as e:
            logging.error(f"No data created for file_id {file_id}")
            logging.exception(e)
            continue

        with open(key_dir / f"{composer}.csv", "a+") as key_file:
            key_symbols = [
                hu.get_scale_degree_from_interval(
                    key.local_tonic - key.global_tonic, key.global_mode, PitchType.TPC
                )
                + ":"
                + str(key.local_mode).split(".")[1]
                for key in piece.get_keys()
            ]
            key_labels[piece.get_keys()[0].global_mode].add([symbol for symbol in key_symbols])
            key_file.write(",".join(key_symbols) + "\n")

        with open(chord_dir / f"{composer}.csv", "a+") as chord_file:
            for start, end in zip(
                piece.get_key_change_indices(),
                list(piece.get_key_change_indices()[1:]) + [len(piece.get_chords())],
            ):
                mode = piece.get_chords()[start].key_mode
                chord_symbols = [
                    hu.get_scale_degree_from_interval(
                        chord.root - chord.key_tonic, mode, PitchType.TPC
                    )
                    + ":"
                    + str(chord.chord_type).split(".")[1][:3]
                    for chord in piece.get_chords()[start:end]
                ]
                chord_labels[mode].add([symbol for symbol in chord_symbols])
                chord_file.write(str(mode).split(".")[1] + ";" + ",".join(chord_symbols) + "\n")

Path(base_dir / "chord_vocab_major.txt").write_text("\n".join(sorted(chord_labels[KeyMode.MAJOR])))
Path(base_dir / "chord_vocab_minor.txt").write_text("\n".join(sorted(chord_labels[KeyMode.MINOR])))
Path(base_dir / "chord_vocab_full.txt").write_text(
    "\n".join(sorted(set(list(chord_labels[KeyMode.MINOR]) + list(chord_labels[KeyMode.MAJOR]))))
)

Path(base_dir / "key_vocab_major.txt").write_text("\n".join(sorted(key_labels[KeyMode.MAJOR])))
Path(base_dir / "key_vocab_minor.txt").write_text("\n".join(sorted(key_labels[KeyMode.MINOR])))
Path(base_dir / "key_vocab_full.txt").write_text(
    "\n".join(sorted(set(list(key_labels[KeyMode.MINOR]) + list(key_labels[KeyMode.MAJOR]))))
)
