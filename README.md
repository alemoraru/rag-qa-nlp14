# rag-qa-nlp14
Project work for DSAIT4090 Natural Language Processing Master's course at TU Delft

[Dexter repository](https://anonymous.4open.science/r/BCQA-05F9/README.md)

- pip install dexter-cqa
- Install PyTorch

Create a folder called "data" and include `wiki_musique_corpus.json` there. Then, in `data` folder, create a new folder 
`dataset` and add the `dev.json`, `train.json` and `test.json` files. The retriever will pick up the data files from 
`config.ini`.