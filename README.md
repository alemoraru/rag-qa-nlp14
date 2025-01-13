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

## Code Quality Checks 🧹

For ensuring consistent code quality, we use the following tools:

1. **Black (Code Formatter)** - See [Black Documentation](https://black.readthedocs.io/en/stable/)
2. **Isort (Import Sorter)** - See [Isort Documentation](https://pycqa.github.io/isort/)

We advise running the following commands while in the root directory of this repository before committing any changes:

1. `pip install -r requirements.txt` (if not already done when adding new features)
2. `black --check .` (to check if the code is formatted correctly)
    - If the above fails, run `black .` to automatically format the code.
3. `isort --check-only .` (to check if the imports are sorted correctly)
    - If the above fails, run `isort .` to automatically sort the imports.

> **Note**: The above `black` & `isort` checks are also enforced in the CI pipeline when a pull request is created.
> If the checks fail, the pull request cannot be merged, therefore it is recommended to run these checks locally.
> We also recommend using `pylint` for static code analysis, but we do not enforce it in the CI pipeline.

## Additional Resources 📚

- [Dexter repository](https://anonymous.4open.science/r/BCQA-05F9/README.md)
- [Data set Google Drive link](https://drive.google.com/drive/folders/1qIZcNcU2wtiJNr3BUyX2GIUtnHEfbQDi)
- [Hugging Face](https://huggingface.co/)
