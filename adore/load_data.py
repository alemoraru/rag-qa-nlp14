import os
import csv 
import json
import argparse
import mmh3

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from bs4 import BeautifulSoup


def map_ids_to_int(split):
    input_path = f"../data/dataset/{split}.json"
    output_dir= "adore_data/dataset_mapped/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    output_path = f"{output_dir}{split}.json"
    
    
    
    
    
    with open(input_path, "r", encoding="utf-8") as json_file:
        queries = json.load(json_file)
    

    id_mapping = {}  # for the mappings lol

    # Map ids to integers
    def process_query(query):
        original_id = query['_id']
       
        mapped_id = str(abs(mmh3.hash(original_id, signed=True)))  #non-negative
        id_mapping[mapped_id] = original_id  
        query['_id'] = mapped_id
        return query

    mapped_queries = list(map(process_query, queries))

    # Save the mapped dataset
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(mapped_queries, json_file)

    # Save the mapping
    with open(f"adore_data/dataset_mapped/query_ids_mapping.{split}", "w", encoding="utf-8") as json_file:
        json.dump(id_mapping, json_file)









def load_and_save_corpus():
    input_json_file = "../data/wiki_musique_corpus.json"  
    output_tsv_file = "adore_data/passage/dataset/collection.tsv"  
    

    # If exists, do not preprocess and save
    if os.path.exists(output_tsv_file):
        print("The file already exists ...")
        return
    

    with open(input_json_file, "r", encoding="utf-8") as json_file:
        corpus = json.load(json_file)
    


    # Save corpus
    with open(output_tsv_file, "w", encoding="utf-8", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
      
        #tsv_writer.writerow(["doc_id", "text"]) 
        for doc_id, doc_content in corpus.items():
            text = f"{doc_content.get('title', '')}. {doc_content.get('text', '')}"
            
            # Removing HTML tags (some examples contained tags, e.g., <br>)
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text()


            # Remove line breaks 
            text = text.replace("\n", " ").strip()
            writer.writerow([doc_id, text])


                    

def create_qrels(split, path, data_split):
    config_path = os.getcwd() + '/config_adore.ini'
    loader = RetrieverDataset("wikimultihopqa", "corpus",
                              config_path, split, tokenizer=None)


   
    # From data loader loads list of queries, corpus and relevance labels
    queries, qrels, _ = loader.qrels()

    query_data = [(query.id(), query.text()) for query in queries]
    
    
    # Save qrels
    with open(f"{path}qrels.{data_split}.tsv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        #writer.writerow(["query_id", "0", "doc_id", "relevance"])  # no header!!!
        for query_id, relevance_dict in qrels.items():
            for doc_id, relevance in relevance_dict.items():
                writer.writerow([query_id, 0, doc_id, relevance]) # 0 for legacy purposes 

    print("qrels saved")


    # Save queries
    with open(f"{path}queries.{data_split}.tsv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        #writer.writerow(["query_id", "query_text"])  no header!!!
        for query_id, query_text in query_data:
            writer.writerow([query_id, query_text])

    print("queries saved")









def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="DEV", choices=["DEV", "TRAIN", "TEST"],
        help="The data split to load (e.g., DEV, TRAIN, TEST)"
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="The path to the data directory."
    )
 
    
    args = parser.parse_args()

    # Map string to Split enum
    split_map = {
        "DEV": (Split.DEV, "dev.small"),
        "TRAIN": (Split.TRAIN, "train"),
        "TEST": (Split.TEST, "test")
    }
    map_ids_to_int(split_map[args.split][1])
    create_qrels(split=split_map[args.split][0], path=args.path, data_split=split_map[args.split][1])
    load_and_save_corpus()

if __name__ == "__main__":
    main()


