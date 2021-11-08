# Harmonic Inference

This is the repository for our ISMIR 2021 paper "A Modular System for the Harmonic Analysis of Musical Scores using a Large Vocabulary".

## Citing
If you use this code, or refer to the paper, please cite it using the following Bibtex:

```
@inproceedings{McLeod:21,
  title={A Modular System for the Harmonic Analysis of Musical Scores using a Large Vocabulary},
  author={McLeod, Andrew and Rohrmeier, Martin},
  booktitle={International Society for Music Information Retrieval Conference {(ISMIR)}},
  year={2021}
}
```

## Installation
1. Clone this repository
2. Set up an environment using your favorite environment manager with python 3, e.g.:
```
conda create -n harmony python=3
conda activate harmony
```
3. Install the package and dependencies with pip:
```
pip install -e .[dev]
```

## Usage
Given a MusicXML or DCML-style MS3 score (e.g., [these](https://github.com/DCMLab/dcml_corpora)), the `annotate.py` script can be used to generate harmonic annotations for it. Note that in these harmonic annotations, applied roots (e.g., the `vii` of a `V/vii` chord) are treated as key modulations and output as such.

### MusicXML
```
python annotate.py -x -i input --checkpoint {checkpoints-best,checkpoints-fh-best} --csm-version {0,1,2}
```
* If `input` is a directory, it directory will be searched recursively for any MusicXML files. Otherwise, only the given file will be processed.

### DCML
Given a DCML annotation corpus (e.g., [these](https://github.com/DCMLab/dcml_corpora)), you must first [aggregate the data](#DCML-Corpus-Aggregation), then you can use the following command:
```
python annotate.py -i corpus_data --checkpoint {checkpoints-best,checkpoints-fh-best} --csm-version {0,1,2}
```
* The argument `--id num` can be used to only run on the file with id `num` (given in the file `corpus/files.tsv` after aggregation).

### Important Arguments
* `--checkpoint` should point to the models you want to use (pre-trained FH, pre-trained internal, or your own; see [Training](#Training)).
* `--csm-version 0` uses the standard CSM, `1` uses the CSM-I, and `2` uses the CSM-T (which achieved the best performance in our tests).
* To use the exact hyperparameter settings from our paper's grid search, add the arguments `--defaults` (for the internal-trained checkpoints `checkpoints-best`) or `--fh-defaults` (for the F-H-trained checkpoints `checkpoints-fh-best`). __You must still set the `--checkpoint` and `--csm-version` manually.__

Other hyperparameters and options can be seen with `python annotate.py -h`.

For example, the best performing model from the internal corpus can be run with: `python annotate.py --checkpoint checkpoints-best --csm-version 2 --defaults -i [input]`

### Output
The output will go into a directory specified by `--output dir` (default `outputs`), and will be a single tsv file with an index and the additional columns `label`, `mc`, and `mn_onset`:

* `label` is the chord or key label, like `Ab:KeyMode.MINOR` (for the key of Ab minor) or `C:Mm7, inv:1` (for a first inversion C7 chord).
* `mc` is the measure index for this label. __These do not necessarily align with the measure numbers written on the score.__ Rather, they are simply a 0-indexed list of all measures according to the input score file (MusicXML or DCML internal). For example, most score formats do not support repeat signs or key changes in the middle of a measure, so these will be split into multiple `mc`s.
* `mc_onset` is the position, measured in whole notes after the __beginning of the mc__, where this label lies.
* `mn_onset` is the position, measured in whole notes after the __downbeat__, where this label lies. Note that these are relative to the actual downbeat, not the beginning of the `mc`.

#### Example Output
&nbsp; | label | mc | mc_onset | mn_onset
------ | ----- | --- | ------- | --------
0  | f:KeyMode.MINOR | 0 | 0 | 3/4
1  | C:M, inv:0 | 0 | 0 | 3/4
2  | F:m, inv:2 | 1 | 0 | 0
3  | F:m, inv:0 | 2 | 0 | 0
...

#### Applied Dominants
Applied chords (e.g., `V/V` or `viio/III`) are represented in our model as key changes. For example, the following output should be interpreted as `I V/V V I` in `C major`:
&nbsp; | label | mc | mc_onset | mn_onset
------ | ----- | --- | ------- | --------
0  | C:KeyMode.MAJOR | 0 | 0 | 0
1  | C:M, inv:0 | 0 | 0 | 0
2  | G:KeyMode.MAJOR | 1 | 0 | 0
3  | D:M, inv:0 | 1 | 0 | 0
4  | C:KeyMode.MAJOR | 2 | 0 | 0
5  | G:M, inv:0 | 2 | 0 | 0
6  | C:M, inv:0 | 3 | 0 | 0
...

Likewise, the following output should be interpreted as `i V/III III i` in `a minor`:
&nbsp; | label | mc | mc_onset | mn_onset
------ | ----- | --- | ------- | --------
0  | a:KeyMode.MINOR | 0 | 0 | 0
1  | A:m, inv:0 | 0 | 0 | 0
2  | C:KeyMode.MAJOR | 1 | 0 | 0
3  | G:M, inv:0 | 1 | 0 | 0
4  | a:KeyMode.MINOR | 2 | 0 | 0
5  | C:M, inv:0 | 2 | 0 | 0
6  | A:m, inv:0 | 3 | 0 | 0
...

## Writing onto a Score
If you are annotating a score from a DCML-style corpus (e.g., [these](https://github.com/DCMLab/dcml_corpora)), the `write_to_score.py` script can be used to write the outputs of the program (both [`annotate.py`](#Usage)) and [`test.py`](#Experimentation)) directly onto the MuseScore3 files:

```
python write_to_score.py --annotations corpus --scores --output output_dir
```
* `corpus` should point to the DCML corpus directory containing the raw label tsvs and MuseScore3 score files.
* `output_dir` should point to the directory containing the model's outputs. This directory will be searched recursively for output tsv files.

The annotated score will be saved in the directory `output_dir/MS3`.

## Experimentation
Instead of `annotate.py`, the `test.py` script can be used to run a trained model on a large set of annotated scores, and evaluate the results.

Supported corpora are currently the [Functional Harmony corpus](https://gitlab.com/algomus.fr/functional-harmony), and DCML-style corpora (e.g., any of those listed [here](https://github.com/DCMLab/dcml_corpora)).

To run experiments, follow these steps:
1. If using a DCML-style corpus, you must first [aggregate the data](#DCML-Corpus-Aggregation).
2. You must then [create h5 data files](#H5-files) for training, validation, and test splits. These are already created for the FH corpus with default settings in the `h5_data-fh` directory.
3. [Train the 6 modules](#Training) using the test and validation data. Pre-trained versions from the paper are provided in this repository.
4. Run experiments using the `test.py` script (instructions below).

```
python test.py --checkpoint checkpoint_dir --h5 h5_dir -i corpus_data
```
* By default, tests will run on the __validation set__. Include the argument `--test` to run on the test set.
* To see other options, run `python test.py -h`.
* If `corpus_data` points to the FH corpus, rather than aggregated DCML-style tsv data, you must also include the `-x` option.

### Recreating our Experiments
To recreate the settings from our experiments in the paper, use the following commands:

Using models trained on the DCML internal corpora:
* CSM: `python test.py --checkpoint checkpoints-best --defaults -i corpus_data --h5 h5_data --csm-version 0 --test`
* CSM-I: `python test.py --checkpoint checkpoints-best --defaults -i corpus_data --h5 h5_data --csm-version 1 --test`
* CSM-T __top performing model__: `python test.py --checkpoint checkpoints-best --defaults -i corpus_data --h5 h5_data --csm-version 2 --test`

Using models trained on the FH corpus:
* CSM: `python test.py --checkpoint checkpoints-fh-best --fh-defaults -i corpus_data --h5 h5_data --csm-version 0 --test`
* CSM-I: `python test.py --checkpoint checkpoints-fh-best --fh-defaults -i corpus_data --h5 h5_data --csm-version 1 --test`
* CSM-T: `python test.py --checkpoint checkpoints-fh-best --fh-defaults -i corpus_data --h5 h5_data --csm-version 2 --test`

### Outputs
The outputs are provided in four ways.

1. Performance metrics (chord accuracy, key accuracy, etc.) are printed to the log. This is std out by default, but can be printed to a file instead with `--log logfile`. Note that this `logfile` path is relative to the `--output` directory. These metrics results can be averaged using `python test.py --average logfile`.
2. `filename.tsv` is identical to the [`annotations.py` output format](#Output), with an additional `color_name` column, used to [write these labels onto a score with color](#Writing-onto-a-Score). Green indicates that the label is entirely correct, red is entirely incorrect, and yellow is partially correct.
3. `filename_results.tsv` contains one row per onset position in the score, with the columns `gt_key` (the correct key label), `gt_chord` (the correct chord label), `est_key` (the estimated key label), `est_chord` (the estimated chord label), and `duration` the duration (in whole notes) for which the given labels are valid. These are listed starting at the beginning of the piece, and duplicate labels are common. Here is an example:

&nbsp; | gt_key | gt_chord | est_key | est_chord | duration 
------ | ----- | --- | ------- | -------- | ----
0 | d:KeyMode.MINOR | D:m, inv:0 | d:KeyMode.MINOR | D:m, inv:0 | 1/4
1 | d:KeyMode.MINOR | D:m, inv:0 | d:KeyMode.MINOR | D:m, inv:0 | 1/8
2 | d:KeyMode.MINOR | D:m, inv:0 | d:KeyMode.MINOR | D:m, inv:0 | 1/8
3 | d:KeyMode.MINOR | C#:o, inv:2 | d:KeyMode.MINOR | D:m, inv:0 | 1/4
...

4. `filename_results_midi.tsv` is the same as `filename_results.tsv`, but the pitches are in semitone format, rather than TPC (i.e., `C#` and `Db` are the same pitch). Here is the same output, where this difference can be noticed in row id 3:

&nbsp; | gt_key | gt_chord | est_key | est_chord | duration 
------ | ----- | --- | ------- | -------- | ----
0 | d:KeyMode.MINOR | D:m, inv:0 | d:KeyMode.MINOR | D:m, inv:0 | 1/4
1 | d:KeyMode.MINOR | D:m, inv:0 | d:KeyMode.MINOR | D:m, inv:0 | 1/8
2 | d:KeyMode.MINOR | D:m, inv:0 | d:KeyMode.MINOR | D:m, inv:0 | 1/8
3 | d:KeyMode.MINOR | C#/Db:o, inv:2 | d:KeyMode.MINOR | D:m, inv:0 | 1/4
...

## Data Creation
For training the modules, h5 data files must be created from the raw data.

### DCML Corpus Aggregation
To use a DCML annotation corpora (e.g., any of those listed [here](https://github.com/DCMLab/dcml_corpora)), you must first create aggregated tsv data with the `aggregate_corpus_data.py` script:

```
python aggregate_corpus_data.py --input [input_dir] --output corpus_data
```
* `input_dir` will be searched recursively for any DCML-style tsv data files.

Now, `corpus_data` will contain aggregated tsv files for use in model training, annotation, and evaluation.

### H5 files
To create h5 data files use the `create_h5_data.py` script.

From a DCML corpus (aggregated as above): `python create_h5_data.py -i corpus_data -h5 h5_data`  
From the Functional Harmony corpus: `python create_h5_data.py -x -i functional-harmony -h5 h5_data`  
* A pre-created version of the F-H data (with default splits and seed) is in the [h5_data-fh](h5_data-fh) directory.

Now, `h5_data` will contain the h5 data files, split into train, test, and validation. Run `python create_h5_data.py -h` for other arguments, like split sizes and random seeding.

## Training
Pre-trained models for the internal data can be found in [checkpoints-best](checkpoints-best).  
Pre-trained models for the Functional Harmony corpus can be found in [checkpoints-fh-best](checkpoints-fh-best).  

You can inspect the hyperparameters and training logs using `tensorboard --logdir [dir]`.

To train new models from scratch, use the `train.py` script. __You must [create h5 data files first](#H5-files).__

The models will save by default in the `checkpoints` directory, which can be changed with the `--checkpoint` argument.

For the initial chord model (ICM), with DCML data: `python train.py -m icm -i corpus_data -h5 h5_data`  
For the ICM, with Functional Harmony data: `python train.py -m icm -i [functional-harmony-dir] -h5 h5_data`

For the other models: `python train.py -m {ctm,ccm,csm,ktm,ksm} -h5 h5_data`

Other arguments (GPU training, etc.) are listed with `python train.py -h`

### Model kwargs
The `--model-kwargs` argument can be used for training models with different dimensionality for a grid search, as well as CSMs and ICMs with different reductions (e.g., CSM-I and CSM-T in the paper). This argument takes a json file name and passes through the values as keyword arguments to the network's `__init__` method.

The json files used for grid search for the results in the paper are in the [model_jsons-grid_search](model_jsons-grid_search) directory.  
The best json files corresponding with the best models from our grid search are in the [model_jsons-best](model_jsons-best) (internal corpus) and [model_jsons-fh-best](model_jsons-fh-best) (FH corpus) directories.
