
BEM - Biomedical Entity-Aware Masking Strategy 
===================

This is an anonymous repository for the EACL 2021 submission:\
*Boosting Low-Resource Biomedical QA via Entity-Aware Masking Strategies*


## Description ##
This repository provides a PyTorch implementation of the *Biomedical Entity-aware Masking* (BEM) core strategy.

The implementation provided builds on top of the Hugging Face libraries ([transformers](https://github.com/huggingface/transformers) and [tokenizers](https://github.com/huggingface/tokenizers)) to fine-tune masked language models by means of BEM.

The core modules that implement the main steps described in the paper are pointed out in the following *Relevant Files* section. 
The full project along with training scripts and auxiliary files will be released upon acceptance.


## Requirements ##
- Python 3.x
- PyTorch >= 1.6.0
- [Transformers 3.0.2](https://github.com/huggingface/transformers)
- [Tokenizers 0.7.0](https://github.com/huggingface/tokenizers)
- [Spacy](https://spacy.io/)
- [SciSpacy](https://allenai.github.io/scispacy/)
- tqdm


## Installation and Usage ##
Additional scripts and information will be released upon acceptance.


## Relevant Files ##
Relevant files of the BEM implementation:

./examples/language-modeling/
- `run_language_modeling.py`: Set up the main flow of execution to detect and extract biomedical entities from a biomedical dataset using SciSpacy; with or without differentiation among UMLS types/concepts.

./src/transformers/data/
- `data_collator.py `: Masking strategy based on the identified entities, optimised to deal with mentions of single or compound words.

./src/transformers/data/datasets/
- `language_modeling.py`: Tokenize and convert tokens to IDs.