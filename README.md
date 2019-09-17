# Code013 CL-LexPG

## Information

Code for paper 'Pointer-based Fusion of Bilingual Lexicons into Neural Machine Translation'. Cite:

	@article{LexPG2019,
        Author = {Jetic Gū and Hassan S. Shavarani and Anoop Sarkar},
        Title = {Pointer-based Fusion of Bilingual Lexicons into Neural Machine Translation},
        Year = {2019},
        Eprint = {arXiv:1909.07907},
    }

## Requirements

This package is written in Python. It uses DyNet library for neural network stuff.

Also, the current implementation uses Cython.
To set it up, go to `src` and run the following command:

	> python setup.py build_ext --inplace

Entire experiment was tested with Python 3.6

## Main programme

The main programme is `src/main.py`.

All options can be seen via

	> python main.py -h

It is recommended that one uses config files. Sample config file and a dev config file can be
found in `src`.


## Decoder plugins

Decoder plugins are placed under `src/decoder/plugins`.
To load any one of them, simply use `--plugins` option of the main programme:

	> python main.py --plugins LexPG

## Main components

The main programme calls `support.configHandler` to process all config and arguments.
`configHandler` also loads all dataset.
Config files are located in `src/configs`.

Models can be loaded and saved through `model.ModelBase` methods.

Training is conducted by `trainer`, and evaluation is done by `evaluator`.
The encoder and decoder does not include methods for training, they simply does sentence encoding and decoding.
