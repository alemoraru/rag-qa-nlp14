# RAGs for Open Domain Complex QA: NLP Group 14

This repository contains the project work done for
the [DSAIT4090 Natural Language Processing](https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=70120)
Master's course at TU Delft.

## Installation 🚀

1. **Install Python >= 3.9**:
    - Install Python from the [official website](https://www.python.org/downloads/).
    - Ensure that Python is added to the system PATH.
    - Preferably, use a virtual environment to install the required packages.
2. **Install Python Package Requirements**:
    ```sh
    pip install -r requirements.txt
    ```

## Setup 🛠️

1. **Create Data Directory**:
    - Create a folder called `data`.
    - Include `wiki_musique_corpus.json` in the `data` folder.

2. **Create Dataset Directory**:
    - Inside the `data` folder, create a new folder called `dataset`.
    - Download all the respective files from
      the [Google Drive link](https://drive.google.com/drive/folders/1qIZcNcU2wtiJNr3BUyX2GIUtnHEfbQDi)
      that was mentioned within the project assignment document.
    - Once downloaded, add the respective files, namely `dev.json`, `train.json`, and `test.json` to the `dataset`
      folder.

3. **Configure Retriever**:
    - Ensure the retriever picks up the data files from `config.ini`. By default, the file paths are already set in
      accordance with the above steps, so no changes are technically required if the repository was cloned as is.

## Hugging Face Token 🔑

- Use the following Hugging Face token to access the `meta-llama/Llama-3.1-8B` model:
    ```
    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
    ```

> **Note**: Should you encounter any issues with the token, you can generate a new one from
> the [Hugging Face website](https://huggingface.co/docs/hub/en/security-tokens). Note that this is not sufficient,
> as you also need explicit access that is granted to the model. You can find this model
> under [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B).

## Modifications ✏️

- Changes were made to the class `llama_engine.py` in the Dexter library.
- The `Requires-Dist: transformers ==4.47.1` requirement of the Dexter library was also modified.

## Running the Project 🏃

- Execute the following command to run the project:
    ```sh
    python3 -m evaluation.llm_orchestrator
    ```

## Additional Resources 📚

- [Dexter repository](https://anonymous.4open.science/r/BCQA-05F9/README.md)
  - The pip command for installing dexter-cqa does not run for python 3.12 versions. The set-up used to install it runs on 
      the latest release of python 3.11. The same issue persisted when trying to run it in a conda env which uses python 3.12.
      If there are issues with building wheels for the `pyproject.toml` it might be due to a missing rust compiler. 
      The compiler can be downloaded from [here](https://www.rust-lang.org/). 

  - pip install dexter-cqa
  - Install [PyTorch](https://pytorch.org/get-started/locally/)

Create a folder called "data" and include `wiki_musique_corpus.json` there. Then, in `data` folder, create a new folder 
`dataset` and add the `dev.json`, `train.json` and `test.json` files. The retriever will pick up the data files from 
`config.ini`. The data can be downloaded from [here](https://drive.google.com/drive/folders/1aQAfNLq6HB0w4_fVnKMBvKA6cXJGRTpH).
- [Data set Google Drive link](https://drive.google.com/drive/folders/1qIZcNcU2wtiJNr3BUyX2GIUtnHEfbQDi)
- [Hugging Face](https://huggingface.co/)

