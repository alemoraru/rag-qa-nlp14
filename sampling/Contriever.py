import json
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.retriever.dense.Contriever import Contriever


class ContrieverPipeline:
    def __init__(self, dataset_name, corpus_name, config_path, split, query_encoder_path, document_encoder_path,
                 batch_size=32, show_progress_bar=True):
        """
        Initialize the Contriever pipeline with the given parameters.
        """
        self.dataset_name = dataset_name
        self.corpus_name = corpus_name
        self.config_path = config_path
        self.split = split
        self.query_encoder_path = query_encoder_path
        self.document_encoder_path = document_encoder_path
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        # Initialize retriever configuration
        self.config_instance = DenseHyperParams(
            query_encoder_path=self.query_encoder_path,
            document_encoder_path=self.document_encoder_path,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar
        )

    def load_data(self, tokenizer=None):
        """
        Load the dataset using the RetrieverDataset.
        """
        self.loader = RetrieverDataset(
            self.dataset_name,
            self.corpus_name,
            self.config_path,
            self.split,
            tokenizer=tokenizer
        )
        self.queries, self.qrels, self.corpus = self.loader.qrels()

    def retrieve(self, top_k=15, chunk=True, chunksize=512):
        """
        Perform retrieval using the Contriever model.
        """
        contrvr_search = Contriever(self.config_instance)
        similarity_measure = CosineSimilarity()
        self.response = contrvr_search.retrieve(
            self.corpus, self.queries, top_k, similarity_measure, chunk=chunk, chunksize=chunksize
        )

    def evaluate_metrics(self, k_values=[1, 3, 5]):
        """
        Evaluate retrieval metrics.
        """
        metrics = RetrievalMetrics(k_values=k_values)
        results = metrics.evaluate_retrieval(qrels=self.qrels, results=self.response)
        print("Metrics:", results)
        return results

    def save_response(self, output_path="responseDictContriever.json"):
        """
        Serialize and save the retrieval response to a JSON file.
        """
        with open(output_path, "w") as file:
            json.dump(self.response, file)
        print(f"Responses have been saved to {output_path}.")


if __name__ == "__main__":
    # Configuration parameters
    DATASET_NAME = "wikimultihopqa"
    CORPUS_NAME = "corpus"
    CONFIG_PATH = "../config.ini"  # local path configuration to the dataset
    SPLIT = Split.DEV  # we use dev.json which contains: questions, contexts and answers in this analysis
    QUERY_ENCODER_PATH = DOCUMENT_ENCODER_PATH = "facebook/contriever"  # path to the pre-trained contriever model from Huggingface

    # Initialize the pipeline
    pipeline = ContrieverPipeline(
        dataset_name=DATASET_NAME,
        corpus_name=CORPUS_NAME,
        config_path=CONFIG_PATH,
        split=SPLIT,
        query_encoder_path=QUERY_ENCODER_PATH,
        document_encoder_path=DOCUMENT_ENCODER_PATH
    )

    # Load data
    pipeline.load_data()

    # Perform retrieval
    pipeline.retrieve()

    # Save responses
    pipeline.save_response()

    # Evaluate metrics
    pipeline.evaluate_metrics()
