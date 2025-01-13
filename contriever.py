import os
import torch

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.retriever.dense.Contriever import Contriever

if __name__ == "__main__":
    # TODO Check how to run on CUDA
    print(f'Is CUDA enabled: {torch.cuda.is_available()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(type(Contriever))

    # Ensure in config.ini the path to the raw data files are linked under [Data-Path]
    # ambignq = '<path to the data file>
    # ambignq-corpus = '<path to the corpus file>'
    # You can set the split to one of Split.DEV, Split.TEST or Split.TRAIN
    # Setting tokenizer=None only loads the raw data processed into our standard data classes, if tokenizer is set,
    # the data is also tokenized and stored in the loader.
    config_path = os.getcwd() + '/config.ini'
    print(config_path)
    loader = RetrieverDataset("wikimultihopqa", "corpus",
                              config_path, Split.DEV, tokenizer=None)

    # Initialize your retriever configuration
    config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                       document_encoder_path="facebook/contriever",
                                       batch_size=32, show_progress_bar=True)

    # From data loader loads list of queries, corpus and relevance labels.
    queries, qrels, corpus = loader.qrels()

    print("FIRST QUERY")
    print(queries[0].id())
    print(queries[0].text())
    # FIRST QUERY
    # 8813f87c0bdd11eba7f7acde48001122
    # Who is the mother of the director of film Polish-Russian War (Film)?

    # Perform Retrieval
    contriever_search = Contriever(config_instance)
    similarity_measure = CosineSimilarity()

    response = contriever_search.retrieve(corpus, queries, 5, similarity_measure, chunk=True, chunksize=100000)

    # Original
    # response = contriever_search.retrieve(corpus, queries, 100, similarity_measure, chunk=True, chunksize=400000)

    # Print the responses to a file
    with open('output\contriever_response.txt', 'w') as f:
        f.write(str(response))

    #Evaluate retrieval metrics
    # metrics = RetrievalMetrics(k_values=[1, 3, 5])
    metrics = RetrievalMetrics(k_values=[1])

    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
    #     ({'NDCG@1': 0.41833, 'NDCG@3': 0.3366, 'NDCG@5': 0.27901},
    #     {'MAP@1': 0.04208, 'MAP@3': 0.0749, 'MAP@5': 0.08751},
    #     {'Recall@1': 0.04208, 'Recall@3': 0.09354, 'Recall@5': 0.11945},
    #     {'P@1': 0.41833, 'P@3': 0.31, 'P@5': 0.2375})

    #     TODO:
    #      1. Figure out how to save the contexts (Should work after the printing ??? need to test it tho)
    #      2. Do the same with gold contexts from dev.json

