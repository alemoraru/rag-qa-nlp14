import json

class SamplingGenerator:
    def __init__(self, response_dict):
         self.response_dict = response_dict

    def negative_sampling(self, k, negative_amount = 2):
        """
        Perform negative sampling from the response dictionary
        and append the negative docs with the top k relevant docs.
        """
        resultDict = {}
        for query_id, top_docs in self.response_dict.items():
            # Sort top_docs by 'cosine similarity' in descending order
            sorted_top_docs = dict(sorted(top_docs.items(), key=lambda item: item[1], reverse=True))

            # Get the first k and the last n negative docs
            top_k = list(sorted_top_docs.items())[:k]
            negative_docs = list(sorted_top_docs.items())[-negative_amount:]
            resultDict[query_id] = top_k + negative_docs
        return resultDict
        

    def random_sampling(self, k, random_amount = 2):
        """
        Perform random sampling from the response dictionary 
        and append the random docs with the top k relevant docs.
        """
        pass
    

    def relevant_sampling(self, k):
        """
        Retrieve top k relevant documents.
        """
        resultDict = {}
        for query_id, top_docs in self.response_dict.items():
            # Sort top_docs by 'cosine similarity' in descending order
            sorted_top_docs = dict(sorted(top_docs.items(), key=lambda item: item[1], reverse=True))
            resultDict[query_id] = dict(list(sorted_top_docs.items())[:k])
        
        return resultDict




