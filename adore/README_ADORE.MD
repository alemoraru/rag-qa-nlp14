## ADORE PREP



### 1. Download the pretrained models
Create a folder named ```models``` within the ```adore``` directory. Download the correpsonding files for the pre-trained models for passages from ```https://github.com/jingtaozhan/DRhard``` and place them in the following paths:

```
models/star_model
models/adore_model
```




### 2. Preprocess the data

a) Preprocess the corpus and queries for each data split (DEV, TRAIN, TEST) into the format compatible with the ADORE retriver. Create a ```qrels``` file (a set of query-document pairs).
```
cd adore
python load_data.py --path adore_data/passage/dataset/ --split {split}
```
b) Preprocess the data further by tokenizing.
```
python cd DRhard_utils
python preprocess.py
```


### 3. Document embeddings
Embeddings are precomputed the STAR model and required for the ADORE retriever (both training and inference).

```
python ./star/inference.py --data_type passage --mode dev --no_cuda
```


### 4. Finetune ADORE
Start from the pretrained weights and finetune the ADORE model.

```
python adore/train.py --init_path ../models/adore_model --pembed_path ../star_embeddings/passages.memmap --model_save_dir ../models/adore_model_finetuned --log_dir ../models/log --preprocess_dir ../adore_data/passage/preprocess
```


### 5. Inference
In order to run inference document embeddings have to be computed as described in step 3.
```
python ./adore/inference.py --model_dir ../models/adore_model_finetuned/epoch-6 --output_dir ../adore_data/passage/evaluate --preprocess_dir ../adore_data/passage/preprocess --mode dev --dmemmap_path ../star_embeddings/passages.memmap
``` 

Compute relevant IR metrics:
```
python ./msmarco_eval.py ../adore_data/passage/preprocess/dev-qrel.tsv ../adore_data/passage/evaluate/dev.rank.tsv 10                                                
```
