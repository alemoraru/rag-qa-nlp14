## ADORE PREP



### 1. Download the pretrained models
Create a folder named ```models``` within the ```adore``` directory. Download the correpsonding files for the pre-trained models for passages from ```https://github.com/jingtaozhan/DRhard``` and place them in the following paths:

```
models/star_model
models/adore_model
```




### 2. Preprocess the data

a) Preprocess the corpus and queries for each data split (```DEV```, ```TRAIN```, ```TEST```).
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

### 3. Document embeddings
Embeddings are precomputed using the STAR model and required for the ADORE retriever (in case of both training and inference).

```
cd /path/to/project/adore/DRhard_utils
python ./star/inference.py --data_type passage --mode dev --no_cuda
```


### 4. Finetune ADORE
Start from the pretrained weights and finetune the ADORE model.

```
python adore/train.py --init_path ../models/adore_model --pembed_path ../star_embeddings/passages.memmap --model_save_dir ../models/adore_model_finetuned --log_dir ../models/log --preprocess_dir ../adore_data/passage/preprocess
 ```
   - **What it does**:
     - Full model finetuning using dynamic hard negatives.
 

### 5. Inference
In order to run inference on the seleceted data split, document embeddings have to be computed as described in step 3.
```
cd /path/to/project/adore

python DRhard_utils/adore/inference.py --model_dir models/adore_model_finetuned/epoch-6 --output_dir adore_data/passage/evaluate --preprocess_dir adore_data/passage/preprocess --mode dev --dmemmap_path star_embeddings/passages.memmap
``` 


### 6. Compute relevant IR metrics:
```
cd /path/to/project/adore

python DRhard_utils/msmarco_eval.py adore_data/passage/preprocess/dev-qrel.tsv adore_data/passage/evaluate/dev.rank.tsv 10                                              
```



## ADORE as Retrieval Module

In order to retrieve relevant passages and map qids/pids back to the original ones, run the inference pipeline of ADORE:
```
cd /path/to/project/adore

python adore_inference_pipeline.py                                            
```
