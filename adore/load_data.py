import argparse
import csv
import json
import os

import mmh3
from bs4 import BeautifulSoup
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset


def load_json(input_path):
    with open(input_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)


def map_ids_to_int(split):
    """Map query ids to integers using hashing."""
    input_path = f"../data/dataset/{split}.json"
    output_dir = "adore_data/dataset_mapped/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = f"{output_dir}{split}.json"
    queries = load_json(input_path=input_path)

    id_mapping = {}  # for the mappings lol

    # Map ids to integers
    def process_query(query):
        original_id = query["_id"]

        mapped_id = str(abs(mmh3.hash(original_id, signed=True)))  # non-negative
        id_mapping[mapped_id] = original_id
        query["_id"] = mapped_id
        return query

    mapped_queries = list(map(process_query, queries))

    # Save the mapped dataset
    save_json(data=mapped_queries, output_path=output_path)

    # Save the mapping
    mapping_path = f"{output_dir}query_ids_mapping.{split}"
    save_json(data=id_mapping, output_path=mapping_path)


def load_and_save_corpus():
    """Preprocess and save the corpus for ADORE."""
    input_json_file = "../data/wiki_musique_corpus.json"

    output_dir = "adore_data/passage/dataset/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_tsv_file = output_dir + "collection.tsv"
    if os.path.exists(output_tsv_file):
        print("The file already exists ...")
        return

    corpus = load_json(input_path=input_json_file)

    # Save corpus
    with open(output_tsv_file, "w", encoding="utf-8", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")

        for doc_id, doc_content in corpus.items():
            text = f"{doc_content.get('title', '')}. {doc_content.get('text', '')}"

            # # Removing HTML tags (some examples contained tags, e.g., <br>)
            # soup = BeautifulSoup(text, "html.parser")
            # text = soup.get_text()

            # Remove line breaks
            text = text.replace("\n", " ").strip()
            writer.writerow([doc_id, text])


def create_qrels(split, path, data_split):
    """Create a qrels file."""
    config_path = os.getcwd() + "/config_adore.ini"
    loader = RetrieverDataset(
        "wikimultihopqa", "corpus", config_path, split, tokenizer=None
    )

    # From data loader loads list of queries, corpus and relevance labels
    queries, qrels, _ = loader.qrels()

    query_data = [(query.id(), query.text()) for query in queries]

    # Save qrels
    with open(
        f"{path}qrels.50k.{data_split}.tsv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        for query_id, relevance_dict in qrels.items():
            for doc_id, relevance in relevance_dict.items():
                writer.writerow(
                    [query_id, 0, doc_id, relevance]
                )  # 0 for legacy purposes

    print("qrels saved")

    # Save queries
    with open(
        f"{path}queries.50k.{data_split}.tsv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        for query_id, query_text in query_data:
            writer.writerow([query_id, query_text])

    print("queries saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="DEV",
        choices=["DEV", "TRAIN", "TEST"],
        help="The data split to load (e.g., DEV, TRAIN, TEST)",
    )
    parser.add_argument(
        "--path", type=str, required=True, help="The path to the output data directory."
    )

    args = parser.parse_args()

    # Map string to Split enum
    split_map = {
        "DEV": (Split.DEV, "dev.small"),
        "TRAIN": (Split.TRAIN, "train"),
        "TEST": (Split.TEST, "test"),
    }
    # map_ids_to_int(split_map[args.split][1])
    create_qrels(
        split=split_map[args.split][0],
        path=args.path,
        data_split=split_map[args.split][1],
    )
    # load_and_save_corpus()


if __name__ == "__main__":
    main()
