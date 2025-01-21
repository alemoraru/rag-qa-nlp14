import os

from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity

if __name__ == "__main__":
    print(type(Contriever))
    # Ensure in config.ini the path to the raw data files are linked under [Data-Path]
    # ambignq = '<path to the data file>
    # ambignq-corpus = '<path to the corpus file>'
    # You can set the split to one of Split.DEV, Split.TEST or Split.TRAIN
    # Setting tokenizer=None only loads the raw data processed into our standard data classes, if tokenizer is set,
    # the data is also tokenized and stored in the loader.
    config_path = os.getcwd() + "/data/config.json"
    print(config_path)
    loader = RetrieverDataset(
        "wikimultihopqa", "corpus", config_path, Split.DEV, tokenizer=None
    )

    # Initialize your retriever configuration
    config_instance = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=32,
        show_progress_bar=True,
    )

    # From data loader loads list of queries, corpus and relevance labels.
    queries, qrels, corpus = loader.qrels()

    print("FIRST QUERY")
    print(queries[0].id())
    print(queries[0].text())

    # Perform Retrieval
    contrvr_search = Contriever(config_instance)
    similarity_measure = CosineSimilarity()
    response = contrvr_search.retrieve(
        corpus, queries, 100, similarity_measure, chunk=True, chunksize=400000
    )

    # Evaluate retrieval metrics
    metrics = RetrievalMetrics(k_values=[1, 10, 100])
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
