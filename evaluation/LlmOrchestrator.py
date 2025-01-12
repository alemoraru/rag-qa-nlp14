import json
import os

from dto.Query import Document, Query, QueryContext
from evaluation.LlamaEngine import LlamaEngine
from sampling.SamplingGenerator import SamplingGenerator


# available models: openai, llama, flant5, mistral
class EvalPipeline:
    def __init__(self, model="llama", golden_evaluation=False):
        self.golden_eval = golden_evaluation
        self.llm_instance = LlamaEngine(data="", model_name="meta-llama/Llama-3.2-1B-Instruct")

    def set_golden_eval(self, golden: bool):
        self.golden_eval = golden

    def evaluate_queries(self, queries: list[Query]):
        """
        For each query, the LLM is called to answer the query's prompt.
        """

        prompt_type = "document"
        if self.golden_eval:
            prompt_type = "context"

        for query in queries:
            prompt, documents = self.create_prompt(query, prompt_type)
            llm_answer = self.llm_instance.get_llama_completion(user_prompt=prompt, documents=documents)
            # print(llm_answer)
            print(f"Answer {llm_answer}")
            query.add_result(self.extract_correct_answer(llm_answer))

        return queries

    @staticmethod
    def create_prompt(query: Query, prompt_type="document"):
        """
        Create a prompt to the query contains either the context (gold docs) of the query (prompt_type=context)
        or the ones retrieved (promt_type = document).
        """

        context = []
        if prompt_type == "document":
            context = query.documents_to_dict()
        else:
            context = query.context_to_dict()

        return query.question, context

    @staticmethod
    def extract_correct_answer(output):
        # Extract the response after "Assistant:"
        assistant_response = output.split("Answer: ")[-1].strip()

        return assistant_response

    @staticmethod
    def assess_result(query: Query):
        """
        Compare the actual vs the llm-based answer of the query based on Exact Match on words.
        """

        return 1 if query.answer.lower() in query.result.lower() else 0


def perform_evaluation(sampling, k=1):
    sampling_docs = retrieve_sampling(file="responseDict", sampling="negative", k=k)

    with open("samplingOut", "w") as file:
        json.dump(sampling_docs, file)
    print("Aggregating data")
    queries = aggregate_data(sampling_docs)

    print("Starting the query evaluation with LLAMA")
    eval_pipeline = EvalPipeline(model="llama")

    if sampling == "golden":
        print("Serving golden docs")
        eval_pipeline.set_golden_eval(golden=True)

    answers = eval_pipeline.evaluate_queries(queries[:2])
    print("Evaluating the results")
    correct_answers = 0
    for query in answers:
        correct_answers += eval_pipeline.assess_result(query)

    print(
        f"Sampling on {sampling} with k {k} with result: correct answers {correct_answers} out of total {len(answers)} -> {correct_answers / len(answers)}")


def retrieve_sampling(file="responseDict", sampling="relevant", k=1):
    """
    Select top k docs from the retrieved ones.
    The sampling can contain either: 
        - relevant k documents only
        - relevant and negative with ratio k:2
        - relevant and random with ratio k:2
    """

    with open(f"{file}.json", "r") as file:
        response_dict = json.load(file)

    sampling_generator = SamplingGenerator(response_dict)

    if sampling == "relevant":
        return sampling_generator.relevant_sampling(k)
    elif sampling == "negative":
        return sampling_generator.negative_sampling(k)
    elif sampling == "random":
        return sampling_generator.random_sampling(k)
    else:
        return Exception("Provide one of the following supported sampling types: relevant, negative or random.")


def aggregate_data(response_dict):
    """
    Aggregate the retrievers response with the datasets 
    and create dto classes of type Query that represents a query
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../data/dataset/dev.json")
    file_path = os.path.abspath(file_path)

    with open(file_path, "r") as file:
        queries = json.load(file)

    file_path = os.path.join(script_dir, "../data/wiki_musique_corpus.json")
    file_path = os.path.abspath(file_path)

    with open(file_path, "r") as file:
        corpus = json.load(file)

    queries_dto = []
    for query_id, sorted_docs in response_dict.items():
        query = find_query_by_id(queries, query_id)
        for doc_id, _ in sorted_docs:
            document = find_document_by_id(corpus, doc_id)
            query.add_document(document)
        queries_dto.append(query)

    return queries_dto


def find_query_by_id(data, query_id: str):
    """
    Finds a query by id and retrieves a Query object
    """
    for subjson in data:
        if subjson.get("_id") == query_id:
            query_contexts = [QueryContext(name=item[0], context=item[1]) for item in subjson.get("context")]
            return Query(query_id, subjson.get("answer"), subjson.get("type"), subjson.get("question"), query_contexts)

    print(f"Found zero matching query with id: {query_id}")
    return None


def find_document_by_id(data, doc_id: str):
    """
    Finds a document by id and retrieves a Document object
    """
    for key, subjson in data.items():
        if key == doc_id:
            return Document(doc_id, subjson.get("title"), subjson.get("text"))

    print(f"Found zero matching docs with id: {doc_id}")
    return None


if __name__ == "__main__":
    print("-----Starting golden eval-----")
    # Eval golden docs
    perform_evaluation(sampling="golden")

    # #Eval relevant docs only top 1
    # perform_evaluation(sampling = "relevant", k = 1)
    # #Eval relevant docs only top 3
    # perform_evaluation(sampling = "relevant", k = 3)
    # #Eval relevant docs only top 5
    # perform_evaluation(sampling = "relevant", k = 5)

    # #Eval negative docs with top 5 relevant, ratio 5:2
    # perform_evaluation(sampling = "negative", k = 5)

    # #Eval random docs with top 5 relevant, ratio 5:2
    # perform_evaluation(sampling = "random", k = 5)
