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

## Setup ADORE

1. **Download the pretrained models**

- Create a folder named ```models``` within the ```adore``` directory. 
- Download the correpsonding files for the pre-trained models for passages from ```https://github.com/jingtaozhan/DRhard``` and place them in the following paths:

    ```
    models/star_model
    models/adore_model
    ```
2. **Preprocess the data**

    a) Preprocess the corpus and queries for each data split (```DEV```, ```TRAIN```).
    ```
    cd adore
    python preprocess.py
    ```
    - **What it does**:
        - Reads the raw corpus and queries from the specified dataset path.
        - Applies preprocessing steps (ID conversion, cleaning).
        - Outputs the formatted `.tsv` files and the `qrels` file.


    b) Preprocess the data further by tokenizing.

    ```
    python cd /path/to/project/adore/DRhard_utils
    python preprocess.py
    ```
    - **What it does**:
        - Splits the preprocessed passages or queries into smaller chunks.
        - Tokenizes the data. 
        - Generates mappings from passage/query IDs to offsets for efficient retrieval.
 
 
    By default, the file paths are already set in accordance with the above steps, so no changes are technically required if the repository was cloned as is.

3. **Document embeddings**

    Embeddings are precomputed using the STAR model and required for the ADORE retriever (in case of both training and inference).

    ```
    cd /path/to/project/adore/DRhard_utils
    python ./star/inference.py --data_type passage --mode dev --no_cuda
    ```


4. **Finetune ADORE**

    Start from the pretrained weights and finetune the ADORE model.

    ```
    python adore/train.py --init_path ../models/adore_model --pembed_path ../star_embeddings/passages.memmap --model_save_dir ../models/adore_model_finetuned --log_dir ../models/log --preprocess_dir ../adore_data/passage/preprocess
    ```
 

5. **Inference**

    In order to run inference on the seleceted data split, document embeddings have to be computed as described in step 3.
    ```
    cd /path/to/project/adore

    python DRhard_utils/adore/inference.py --model_dir models/adore_model_finetuned/epoch-6 --output_dir adore_data/passage/evaluate --preprocess_dir adore_data/passage/preprocess --mode dev --dmemmap_path star_embeddings/passages.memmap
    ``` 


6. **Compute relevant IR metrics**
    ```
    cd /path/to/project/adore

    python DRhard_utils/msmarco_eval.py adore_data/passage/preprocess/dev-qrel.tsv adore_data/passage/evaluate/dev.rank.tsv 15                                              
    ```



## ADORE as Retrieval Module

In order to retrieve relevant passages and map qids/pids back to the original ones, run the inference pipeline of ADORE:
```
cd /path/to/project/adore

python adore_inference_pipeline.py                                            
```



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

Assuming you have already run the retrieval step (either using contriever or ADORE), and now you would like to
actually evaluate the RAG pipeline on the provided set of queries, you can run the following command, while
within the root directory of the repository:

```shell
python -m evaluation.llm_orchestrator -s relevant -k 1 -q 2
```

As already noticed the command above is parametrized. What it does is it that it runs the LLM orchestrator using
the `relevant` sampling method (i.e. positive examples), with a top-k value of `1`, and evaluates just `2` queries.
The `llm_orchestrator` is parametrized, meaning you can tweak the following parameters:

```markdown
--sampling_method, -s: The sampling method to use. Options: relevant, negative, random, golden. Required to be set.
--k, -k: The number of top-k documents to use while retrieving answers. Required to be set.
--num_queries, -q: The number of queries to evaluate. Not required to be set, defaults to all queries. Should be used
only for
debugging purposes to evaluate a smaller subset of queries.
--retrieval_results_file, -f: The file containing the retrieval results. Not required to be set, defaults to the
                              responseDict.json file found in the root directory of this repository.
--verbose, -v: Whether to print verbose output. Not required to be set, defaults to False.
```

> **Note**: You can also run `python -m evaluation.llm_orchestrator --help` to see the above information in the
> terminal (will be displayed as a help message in a different way).

Additionally, you will notice that the `llm_orchestrator` also requires to set the `huggingface_token` environment
variable. Depending on your OS and mode of running, this can be done in different ways. For example, if you are
simply running everything from the command line on a Unix system, you can set the environment variable as follows:

```shell
# replace YOUR_HUGGING_FACE_TOKEN with the actual token
export huggingface_token="YOUR_HUGGING_FACE_TOKEN"
```

If the token is not set properly, you will get the following error message:

```shell
...
    raise KeyError(key) from None
KeyError: 'huggingface_token'
```

Refer back to the [Hugging Face Token](#hugging-face-token-) section for more information on how to obtain the token.

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
