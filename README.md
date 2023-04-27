# LLM-AttrProposal
LLM attribute proposal module for 2023 Spring CMU PGM course project

## Setup
Install python environment with `conda create -f environment.yaml`

## Create target models
Run `python code/classifier.py --do_train --do_val` to generate target classification models.

See `scripts/train.sh` for an example use case.

Run `python code/classifier.py -h` to get the description for all options.

## Generate attribute-specific datasets
Run `python code/gen.py` to generate datasets with attribute features included. See `config/attributes/afhq.json` for an example of attribute list file.

See `scripts/gen.sh` for an example use case.

Run `python code/gen.py -h` to get the description for all options.

## Diagnose a target model on each attribute
Run `python code/diag.py` to diagnose target models on each attribute dataset. A table of model evaluation metrics on each attribute will be produced.

See `scripts/diag.sh` for an example use case.

Run `python code/diag.py -h` to get the description for all options.

## Evaluate a target model on a given dataset.
Run `python code/classifier.py --do_test` to evaluate the model on a test dataset.

See `scripts/test.sh` for an example use case.

Run `python code/classifier.py -h` to get the description for all options.
