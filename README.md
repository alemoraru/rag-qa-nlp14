# rag-qa-nlp14
Project work for DSAIT4090 Natural Language Processing Master's course at TU Delft

[Dexter repository](https://anonymous.4open.science/r/BCQA-05F9/README.md)

- pip install dexter-cqa
- Install PyTorch

Create a folder called "data" and include `wiki_musique_corpus.json` there. Then, in `data` folder, create a new folder `dataset` and add the `dev.json`, `train.json` and `test.json` files. The retriever will pick up the data files from `config.ini`.


My huggingface token = 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 with access to the `meta-llama/Llama-3.1-8B` model (one of the newest)

I did changes to the class `llama_engine.py` (you should also chage that, or experiment) belonging to dexter library and also I changed the `Requires-Dist: transformers ==4.47.1` requirement of the dexter library. 

Run command: python3 -m evaluation.LlmOrchestrator

Problem: the chat template in `llama_engine` is not compatible with `meta-llama/Llama-3.1-8B` model.