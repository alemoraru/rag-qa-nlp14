import json
import os

import torch
from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity


def contriever(loader, to_save=False):
    # From data loader loads list of queries, corpus and relevance labels.
    queries, qrels, corpus = loader.qrels()

    print("FIRST QUERY")
    print(queries[0].id())
    print(queries[0].text())

    # Initialize your retriever configuration
    config_instance = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=32,
        show_progress_bar=True,
    )

    # Perform Retrieval
    contriever_search = Contriever(config_instance)
    similarity_measure = CosineSimilarity()

    # Original
    response = contriever_search.retrieve(
        corpus, queries, 5, similarity_measure, chunk=True, chunksize=400000
    )
    # response = contriever_search.retrieve(corpus, queries, 5, similarity_measure, chunk=True, chunksize=100000)

    if to_save:
        # Print the responses to a file
        with open("../output/contriever_response.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

    return response


def get_loader(split):
    # Ensure in config.ini the path to the raw data files are linked under [Data-Path]
    # ambignq = '<path to the data file>
    # ambignq-corpus = '<path to the corpus file>'
    # You can set the split to one of Split.DEV, Split.TEST or Split.TRAIN
    # Setting tokenizer=None only loads the raw data processed into our standard data classes, if tokenizer is set,
    # the data is also tokenized and stored in the loader.
    config_path = os.getcwd() + "\\config.ini"
    loader = RetrieverDataset(
        "wikimultihopqa", "corpus", config_path, split, tokenizer=None
    )
    return loader


def get_metrics(response, qrels, k_values=None):
    # Evaluate retrieval metrics
    metrics = RetrievalMetrics(k_values=k_values)

    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
    #     ({'NDCG@1': 0.41833, 'NDCG@3': 0.3366, 'NDCG@5': 0.27901},
    #     {'MAP@1': 0.04208, 'MAP@3': 0.0749, 'MAP@5': 0.08751},
    #     {'Recall@1': 0.04208, 'Recall@3': 0.09354, 'Recall@5': 0.11945},
    #     {'P@1': 0.41833, 'P@3': 0.31, 'P@5': 0.2375})
    return metrics


if __name__ == "__main__":
    # TODO Check how to run on CUDA
    print(f"Is CUDA enabled: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(type(Contriever))
    k_values = [1, 3, 5]

    # Run metrics for Split.DEV
    dev_loader = get_loader(Split.DEV)
    dev_queries, dev_qrels, dev_corpus = dev_loader.qrels()
    get_metrics(dev_corpus, dev_qrels, k_values=k_values)

    # Run metrics for Split.TRAIN
    # train_loader = get_loader(Split.TRAIN)
    # response = contriever(train_loader, to_save=True)
    # train_queries, train_qrels, train_corpus = dev_loader.qrels()
    # get_metrics(response, train_qrels, k_values=k_values)
