import json
import os
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dto.query import Query, QueryContext


class SamplingGenerator:
    """
    SamplingGenerator class that is responsible for the sampling methods for retrieving documents.
    """

    def __init__(self, response_dict):
        self.response_dict = response_dict

    def relevant_sampling(self, k):
        """
        Retrieve top K relevant documents.
        :param k: Number of relevant docs to be retrieved
        :return: Dictionary containing the query_id and the top K relevant docs
        """

        result_dict = {}
        for query_id, top_docs in self.response_dict.items():
            # Sort top_docs by 'cosine similarity' in descending order
            sorted_top_docs = dict(
                sorted(top_docs.items(), key=lambda item: item[1], reverse=True)
            )
            result_dict[query_id] = list(sorted_top_docs.items())[:k]

        return result_dict

    def negative_sampling(self, k):
        """
        Perform negative sampling from the response dictionary
        and append the negative docs with the top K relevant docs.
        :param k: Number of relevant docs to be retrieved
        :return: Dictionary containing the query_id and the top K relevant docs
        """
        result_dict = {}
        for query_id, top_docs in self.response_dict.items():
            # Sort top_docs by 'cosine similarity' in descending order
            sorted_top_docs = dict(
                sorted(top_docs.items(), key=lambda item: item[1], reverse=True)
            )

            # Get the first k and the last n negative docs
            top_k = list(sorted_top_docs.items())[:k]
            negative_docs = list(sorted_top_docs.items())[
                k:
            ]  # potential negative documents are a total of 15 - topK

            result_dict[query_id] = top_k + negative_docs
        return result_dict

    def random_sampling(self, k, random_amount=2):
        """
        Perform random sampling from the response dictionary
        and append the random docs with the top k relevant docs.
        :param k: number of relevant docs
        :param random_amount: number of random docs to be sampled
        :return: Dictionary containing the query_id and the top k relevant docs + random docs
        """
        if k == 1:
            random_amount = 1
        result_dict = {}
        queries = list(self.response_dict.keys())  # Get all query IDs
        for query_id, top_docs in self.response_dict.items():
            # Sort top_docs by 'cosine similarity' in descending order
            sorted_top_docs = dict(
                sorted(top_docs.items(), key=lambda item: item[1], reverse=True)
            )

            # Get the first k docs
            top_k = list(sorted_top_docs.items())[:k]

            # Extract the document IDs of the top k docs
            top_k_doc_ids = {doc_id for doc_id, _ in top_docs.items()}

            # Initialize the random_docs list
            random_docs = []
            attempts = 0

            # Attempt to find random docs from other queries
            while len(random_docs) < random_amount and attempts < 10 * len(queries):
                # Pick a random query that is not the current one
                random_query_id = random.choice(queries)
                
                if random_query_id == query_id:
                    continue
    
                # Get the documents of the random query
                random_query_docs = self.response_dict[random_query_id]

                # Pick one random document from the random query
                for doc_id, score in random_query_docs.items():
                    if doc_id not in top_k_doc_ids and doc_id not in {doc[0] for doc in random_docs}:
                        random_docs.append((doc_id, score))
                        break

                attempts += 1

            result_dict[query_id] = top_k + random_docs

        return result_dict


    def golden_context_sampling(self, k):
        """
        Perfom top K retrieval of golden documents for each query
        based on the embeddings cosine similarity between a query and the document
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/dataset/dev.json")
        file_path = os.path.abspath(file_path)

        with open(file_path, "r") as file:
            queries = json.load(file)

        queries_with_k_context = []
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        for sub_json in queries:
            context = sub_json.get("context")
            question = sub_json.get("question")
            query_contexts = [
                QueryContext(name=item[0], context=item[1]) for item in context
            ]

            document_texts = [f"{doc.name} {doc.context}" for doc in query_contexts]
            query_embedding = model.encode(question, convert_to_tensor=False)
            document_embeddings = model.encode(document_texts, convert_to_tensor=False)

            # Compute cosine similarity between the query and each document
            similarity_scores = cosine_similarity(
                [query_embedding], document_embeddings
            )[0]

            top_k_indices = np.argsort(similarity_scores)[::-1][:k]
            top_k_documents = [query_contexts[i] for i in top_k_indices]

            queries_with_k_context.append(
                Query(
                    sub_json.get("_id"),
                    sub_json.get("answer"),
                    sub_json.get("type"),
                    question,
                    top_k_documents,
                )
            )

        return queries_with_k_context
