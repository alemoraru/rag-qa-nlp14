# rag-qa-nlp14
Project work for DSAIT4090 Natural Language Processing Master's course at TU Delft

[Dexter repository](https://anonymous.4open.science/r/BCQA-05F9/README.md)

The pip command for installing dexter-cqa does not run for python 3.12 versions. The set-up used to install it runs on 
the latest release of python 3.11. The same issue persisted when trying to run it in a conda env which uses python 3.12.
If there are issues with building wheels for the `pyproject.toml` it might be due to a missing rust compiler. 
The compiler can be downloaded from [here](https://www.rust-lang.org/). 

- pip install dexter-cqa
- Install [PyTorch](https://pytorch.org/get-started/locally/)

Create a folder called "data" and include `wiki_musique_corpus.json` there. Then, in `data` folder, create a new folder 
`dataset` and add the `dev.json`, `train.json` and `test.json` files. The retriever will pick up the data files from 
`config.ini`. The data can be downloaded from [here](https://drive.google.com/drive/folders/1aQAfNLq6HB0w4_fVnKMBvKA6cXJGRTpH).