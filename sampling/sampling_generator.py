import random


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

    def negative_sampling(self, k, negative_amount=2):
        """
        Perform negative sampling from the response dictionary
        and append the negative docs with the top K relevant docs.
        :param k: Number of relevant docs to be retrieved
        :param negative_amount: Number of negative docs to be retrieved
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
            negative_docs = list(sorted_top_docs.items())[-negative_amount:]
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

        result_dict = {}
        for query_id, top_docs in self.response_dict.items():
            # Sort top_docs by 'cosine similarity' in descending order
            sorted_top_docs = dict(
                sorted(top_docs.items(), key=lambda item: item[1], reverse=True)
            )

            # Get the first k docs
            top_k = list(sorted_top_docs.items())[:k]

            # Get remaining docs by excluding the top k docs
            remaining_docs = list(sorted_top_docs.items())[k:]

            # Get random docs from the remaining docs
            random_docs = random.sample(
                remaining_docs, min(random_amount, len(remaining_docs))
            )
            result_dict[query_id] = top_k + random_docs

        return result_dict
