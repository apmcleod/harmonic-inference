# harmonic-inference

This is the repository for my ISMIR 2021 paper "A Modular System for the Harmonic Analysis of Musical Scores using a Large Vocabulary".

## Citing
If you use this code, please cite using the following Bibtex:

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
2. Set up an envoriment using your favorite environment manager with python 3, e.g.:
```
conda create -n harmony python=3
conda activate harmony
```
3. Install the package and dependencies with pip:
```
pip install -e .[dev]
```

## Pre-trained Models


## Usage


### Data Creation
For training the modules, h5 data files need to be created from the raw data.

#### DCML Pre-processing
From a DCML annotation corpus (e.g., any of those listed [here](https://github.com/DCMLab/dcml_corpora)), you must first create aggregated tsv data with the `aggregate_corpus_data.py` script:

```
python aggregate_corpus_data.py --input [input_dir] --output corpus_data
```

Now, `corpus_data` will contain aggregated tsv files for use in data creation.

#### H5 files
To create h5 data files, use the `create_h5_data.py` script.

From a DCML corpus (aggregated as above): `python create_h5_data.py -i corpus_data -h5 h5_data`  
From the Functional Harmony corpus: `python create_h5_data.py -x -i functional-harmony -h5 h5_data`  
* A pre-created version of the F-H data (with default splits and seed) is in the h5_data-fh directory.

Now, `h5_data` will contain the h5 data files, split into train, test, and validation. Run `python create_h5_data.py -h` for other arguments, like split sizes and random seeding.

### Training
Pre-trained models for the internal data can be found in checkpoints-best.  
Pre-trained models for the Functional Harmony corpus can be found in the checkpoints-fh-best.  

You can inspect the hyperparameters and training logs using `tensorboard`.

To train new models from scratch, use the `train.py` script.

The models will save by default in the `checkpoints` directory, which can be changed with the `--checkpoint` argument.

For the initial chord model (ICM), with DCML data: `python train.py -m icm -i corpus_data -h5 h5_data`  
For the ICM, with Functional Harmony data: `python train.py -m icm -i functional-harmony -h5 h5_data`

For the other models: `python train.py -m {icm,ctm,ccm,csm,ktm,ksm} -h5 h5_data`

Other arguments are listed with `python train.py -h`

#### Model kwargs
The `--model-kwargs` argument can be used for training models with different dimensionality for a grid search, as well as CSMs and ICMs with different reductions (e.g., CSM-I and CSM-T in the paper). This argument takes a json file name and passes through the values as keyword arguments to the networks `__init__` method.

The json files used for grid search for the results in the paper are in the model_jsons-grid_search directory.  
The best json files corresponding with the best models from our grid search are in the model_jsons-best (internal corpus) and model_jsons-fh-best (FH corpus) directories.
