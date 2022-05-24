"""Create data for the chord_features project."""
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from harmonic_inference.data.chord import Chord
from harmonic_inference.data.corpus_reading import load_clean_corpus_dfs
from harmonic_inference.data.data_types import (
    MAJOR_MINOR_REDUCTION,
    NO_REDUCTION,
    TRIAD_REDUCTION,
    PitchType,
)
from harmonic_inference.data.key import Key
from harmonic_inference.data.piece import ScorePiece, get_score_piece_from_data_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create data for the chord-features project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("corpus_data-all_subcorpora"),
        help="The directory containing the raw corpus files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chord-features_data"),
        help="The directory in which to save the created data files.",
    )

    ARGS = parser.parse_args()

    files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(ARGS.input)

    # Make data directory
    base_dir = ARGS.output
    base_dir.mkdir(parents=True, exist_ok=True)

    for file_id, file_row in tqdm(files_df.iterrows(), desc="Loading pieces", total=len(files_df)):
        name = file_row["corpus_name"] + ".." + file_row["file_name"]

        try:
            piece: ScorePiece = get_score_piece_from_data_frames(
                notes_df.loc[file_id],
                chords_df.loc[file_id],
                measures_df.loc[file_id],
                chord_reduction=NO_REDUCTION,
                use_inversions=True,
                use_relative=True,
                name=name,
                changes=False,
            )
        except Exception:
            logging.error(f"No data created for file_id {file_id}")
            continue

        if len(piece.get_chords()) == 0:
            logging.error(f"No data created for file_id {file_id}")
            continue

        chord_data = {
            "root_abs_midi": [],
            "root_abs_tpc": [],
            "root_globalrel_midi": [],
            "root_globalrel_tpc": [],
            "root_localrel_midi": [],
            "root_localrel_tpc": [],
            "root_appliedrel_midi": [],
            "root_appliedrel_tpc": [],
            "bass_abs_midi": [],
            "bass_abs_tpc": [],
            "bass_globalrel_midi": [],
            "bass_globalrel_tpc": [],
            "bass_localrel_midi": [],
            "bass_localrel_tpc": [],
            "bass_appliedrel_midi": [],
            "bass_appliedrel_tpc": [],
            "bass_rootrel_midi": [],
            "bass_rootrel_tpc": [],
            "type": [],
            "triad_type": [],
            "third_type": [],
            "inversion": [],
            "global_tonic_midi": [],
            "global_tonic_tpc": [],
            "global_mode": [],
            "local_tonic_abs_midi": [],
            "local_tonic_abs_tpc": [],
            "local_tonic_rel_midi": [],
            "local_tonic_rel_tpc": [],
            "local_mode": [],
            "applied_tonic_abs_midi": [],
            "applied_tonic_abs_tpc": [],
            "applied_tonic_globalrel_midi": [],
            "applied_tonic_globalrel_tpc": [],
            "applied_tonic_localrel_midi": [],
            "applied_tonic_localrel_tpc": [],
            "applied_mode": [],
        }

        for chord in piece.get_chords():
            chord: Chord
            chord_midi: Chord = chord.to_pitch_type(PitchType.MIDI)

            key: Key = piece.get_key_at(chord.onset)
            key_midi: Key = key.to_pitch_type(PitchType.MIDI)

            assert (
                key.relative_tonic == chord.key_tonic
            ), f"{chord.onset}, {piece.get_key_change_indices()}, {piece.get_keys()}"

            chord_data["root_abs_midi"].append(chord_midi.root)
            chord_data["root_abs_tpc"].append(chord.root)
            chord_data["root_globalrel_midi"].append((chord_midi.root - key_midi.global_tonic) % 12)
            chord_data["root_globalrel_tpc"].append(chord.root - key.global_tonic)
            chord_data["root_localrel_midi"].append((chord_midi.root - key_midi.local_tonic) % 12)
            chord_data["root_localrel_tpc"].append(chord.root - key.local_tonic)
            chord_data["root_appliedrel_midi"].append(
                (chord_midi.root - key_midi.relative_tonic) % 12
            )
            chord_data["root_appliedrel_tpc"].append(chord.root - key.relative_tonic)
            chord_data["bass_abs_midi"].append(chord_midi.bass)
            chord_data["bass_abs_tpc"].append(chord.bass)
            chord_data["bass_globalrel_midi"].append((chord_midi.bass - key_midi.global_tonic) % 12)
            chord_data["bass_globalrel_tpc"].append(chord.bass - key.global_tonic)
            chord_data["bass_localrel_midi"].append((chord_midi.bass - key_midi.local_tonic) % 12)
            chord_data["bass_localrel_tpc"].append(chord.bass - key.local_tonic)
            chord_data["bass_appliedrel_midi"].append(
                (chord_midi.bass - key_midi.relative_tonic) % 12
            )
            chord_data["bass_appliedrel_tpc"].append(chord.bass - key.relative_tonic)
            chord_data["bass_rootrel_midi"].append((chord_midi.bass - chord_midi.root) % 12)
            chord_data["bass_rootrel_tpc"].append(chord.bass - chord.root)
            chord_data["type"].append(chord.chord_type)
            chord_data["triad_type"].append(TRIAD_REDUCTION[chord.chord_type])
            chord_data["third_type"].append(MAJOR_MINOR_REDUCTION[chord.chord_type])
            chord_data["inversion"].append(chord.inversion)
            chord_data["global_tonic_midi"].append(key_midi.global_tonic)
            chord_data["global_tonic_tpc"].append(key.global_tonic)
            chord_data["global_mode"].append(key.global_mode)
            chord_data["local_tonic_abs_midi"].append(key_midi.local_tonic)
            chord_data["local_tonic_abs_tpc"].append(key.local_tonic)
            chord_data["local_tonic_rel_midi"].append(
                (key_midi.local_tonic - key_midi.global_tonic) % 12
            )
            chord_data["local_tonic_rel_tpc"].append(key.local_tonic - key.global_tonic)
            chord_data["local_mode"].append(key.local_mode)
            chord_data["applied_tonic_abs_midi"].append(key_midi.relative_tonic)
            chord_data["applied_tonic_abs_tpc"].append(key.relative_tonic)
            chord_data["applied_tonic_globalrel_midi"].append(
                (key_midi.relative_tonic - key_midi.global_tonic) % 12
            )
            chord_data["applied_tonic_globalrel_tpc"].append(key.relative_tonic - key.global_tonic)
            chord_data["applied_tonic_localrel_midi"].append(
                (key_midi.relative_tonic - key_midi.local_tonic) % 12
            )
            chord_data["applied_tonic_localrel_tpc"].append(key.relative_tonic - key.local_tonic)
            chord_data["applied_mode"].append(key.relative_mode)

        df = pd.DataFrame(chord_data)
        df.to_csv(str(base_dir / name), sep="\t", index=False)
