import argparse
import json
import logging
import os
import time
from enum import Enum
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dto.query import Document, Query, QueryContext
from evaluation.llama_engine import LlamaEngine
from sampling.sampling_generator import SamplingGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]:  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SamplingMethod(Enum):
    """Enumeration for available sampling methods."""

    GOLDEN = "golden"
    RELEVANT = "relevant"
    NEGATIVE = "negative"
    RANDOM = "random"


class EvalQA:
    """
    Class responsible for evaluating the QA answers based on the provided method.
    """

    def __init__(self, golden_answer: str, predicted_answer: str, log_results=False):
        self.golden_answer = golden_answer
        self.predicted_answer = predicted_answer
        self.log_results = log_results

        if self.log_results:
            logging.info(f'Golden: "{self.golden_answer}"')
            logging.info(f'Predicted: "{self.predicted_answer}"')

    def exact_match(self) -> int:
        """
        Evaluate the QA answers based on the exact match method.
        :return: 1 if the answer is correct, 0 otherwise
        """

        exact_match_classification = (
            1 if self.golden_answer.lower() in self.predicted_answer.lower() else 0
        )

        if self.log_results:
            logging.info(f"Exact match: {exact_match_classification}")
        return exact_match_classification

    def f1_score(self, threshold=0.75) -> int:
        """
        Evaluate the QA answers based on the F1 score method.
        If the F1 score is greater than the threshold, then
        the answer is considered correct (1), otherwise incorrect (0).
        :param threshold: the threshold for the F1 score to be considered correct
        :return: 1 if the answer is correct, 0 otherwise
        """

        # Tokenize the answers
        golden_tokens = self.golden_answer.split()
        predicted_tokens = self.predicted_answer.split()

        # Create sets for the tokens
        common_tokens = set(golden_tokens) & set(predicted_tokens)

        # Count common tokens, and total tokens in each answer
        num_common_tokens = len(common_tokens)
        if num_common_tokens == 0:
            return 0  # No common tokens, F1 score is 0, therefore answer is incorrect

        # Calculate precision and recall
        precision = num_common_tokens / len(predicted_tokens)
        recall = num_common_tokens / len(golden_tokens)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_classification = 1 if f1 >= threshold else 0

        if self.log_results:
            logging.info(f"F1 score: {f1} -> {f1_classification}")

        return f1_classification


class EvalPipeline:
    """
    Evaluation pipeline for the LLM model in the context of RAG for QA.
    Available models: openai, llama, flant5, mistral
    """

    def __init__(self, model="llama", golden_evaluation=False, verbose=False):
        self.golden_eval = golden_evaluation
        self.llm_instance = LlamaEngine(
            data="", model_name="meta-llama/Llama-3.2-1B-Instruct"
        )
        self.verbose = verbose

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
            llm_answer = self.llm_instance.get_llama_completion(
                user_prompt=prompt, documents=documents
            )
            # print(f"Answer {self.extract_correct_answer(llm_answer)}")
            query.set_result(self.extract_correct_answer(llm_answer))

        return queries

    @staticmethod
    def create_prompt(query: Query, prompt_type="document"):
        """
        Create a prompt to the query contains either the context (gold docs) of the query (prompt_type=context)
        or the ones retrieved (prompt_type = document).
        """

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

    def assess_result(self, query: Query):
        """
        Compare the actual answer vs the LLM-based answer (i.e. result)
        using both exact match and F1 score.
        :param query: the query to be evaluated
        :return: a tuple containing the exact match and classification using F1 score
        """

        eval_result = EvalQA(query.answer, query.result, self.verbose)
        return eval_result.exact_match(), eval_result.f1_score()


def perform_evaluation(
    sampling_method: SamplingMethod = SamplingMethod.GOLDEN,
    k: int = 1,
    num_queries: Optional[int] = None,
    retrieval_results_file: str = "responseDict",
    verbose: bool = False,
    not_adore: bool = False,
) -> None:
    """
    Perform evaluation on the RAG QA pipeline using the provided sampling method and K argument.
    :param sampling_method: the sampling method to be used
    :param k: the number of relevant docs to be retrieved for each query
    :param num_queries: the number of queries to be evaluated (if None, all queries are evaluated)
    :param retrieval_results_file: the path to the retrieval results file to be used during evaluation
    :param verbose: flag to log intermediate results
    :return: None
    """

    start_time = time.time()
    logging.info(
        f"Starting evaluation with: {sampling_method}, K={k} for {num_queries if num_queries else 'all'} queries "
        f"using the retrieval results from {retrieval_results_file}"
    )
    eval_pipeline = EvalPipeline(model="llama", verbose=verbose)

    # Retrieve the sampling docs
    if sampling_method == SamplingMethod.GOLDEN:
        eval_pipeline.set_golden_eval(golden=True)

    sampling_docs = retrieve_sampling(
        file=retrieval_results_file, sampling=sampling_method, k=k, not_adore=not_adore
    )

    if sampling_method != SamplingMethod.GOLDEN:
        logging.info("Aggregating data...")
        queries = aggregate_data(sampling_docs)
        if sampling_method == SamplingMethod.NEGATIVE:
            queries = find_hard_negatives(queries, k)  # find hard negatives
    else:
        queries = sampling_docs  # use already processed queries

    logging.info("Starting the query evaluation with LLAMA...")
    if num_queries:
        answers = eval_pipeline.evaluate_queries(queries[:num_queries])
    else:
        answers = eval_pipeline.evaluate_queries(queries)

    logging.info("Getting the results...")
    correct_answers_exact_match = 0
    correct_answers_f1_score = 0

    for query in answers:
        exact_match, f1_score = eval_pipeline.assess_result(query)
        correct_answers_exact_match += exact_match
        correct_answers_f1_score += f1_score

    num_answers = len(answers)

    logging.info(
        f"Sampling with {sampling_method} with K={k} gives results:\n"
        f"\tcorrect answers (EM): {correct_answers_exact_match} -> {correct_answers_exact_match} out of {num_answers} = {correct_answers_exact_match / num_answers}\n"
        f"\tcorrect answers (F1): {correct_answers_f1_score} -> {correct_answers_f1_score} out of {num_answers} = {correct_answers_f1_score / num_answers}"
    )

    end_time = time.time()
    logging.info(f"Execution time: {end_time - start_time} seconds")


def retrieve_sampling(file: str, sampling=SamplingMethod.RELEVANT, k=1, not_adore=True):
    """
    Select top k docs from the retrieved ones.
    The sampling can contain either:
        - relevant k documents only
        - relevant and negative with ratio k:2
        - relevant and random with ratio k:2
    """

    with open(f"{file}", "r") as file:
        response_dict = json.load(file)

    sampling_generator = SamplingGenerator(response_dict, not_adore)
    if sampling == SamplingMethod.RELEVANT:
        return sampling_generator.relevant_sampling(k)
    if sampling == SamplingMethod.NEGATIVE:
        return sampling_generator.negative_sampling(k)
    if sampling == SamplingMethod.RANDOM:
        return sampling_generator.random_sampling(k)
    if sampling == SamplingMethod.GOLDEN:
        return sampling_generator.golden_context_sampling(k)

    raise Exception(
        "Provide one of the following supported sampling types: relevant, negative or random."
    )


def find_hard_negatives(queries, k):
    """
    Compute hard negatives but ranking 15-topK relevant documents with the ground truth
    Hard negatives will be the lowest similar documents.
    :param queries: queries to find hard negatives for
    :param k: used to derive the negative ratio of documents. For k = 1, the ration is 1, else for k = 3 or 5 the ration is 2.
    """
    negative_amount = 1
    if k != negative_amount:
        negative_amount = 2

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for query in queries:
        scores = []
        for neg in query.documents[k:]:
            sum_score = 0
            for c in query.query_context:
                negative_doc_text = f"{neg.title} {neg.text}"
                ground_truth_doc = f"{c.name} {c.context}"
                neg_embedding = model.encode(negative_doc_text, convert_to_tensor=False)
                context_embedding = model.encode(
                    ground_truth_doc, convert_to_tensor=False
                )
                similarity_score = cosine_similarity(
                    [neg_embedding], [context_embedding]
                )[0][0]
                sum_score += similarity_score
            avg_score = sum_score / len(query.query_context)
            scores.append(avg_score)
        indexed_scores = list(enumerate(scores))
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1])
        lowest_two = sorted_scores[:negative_amount]
        hard_negatives = []
        for id, _ in lowest_two:
            hard_negatives.append(query.documents[k + id])
        query.documents = query.documents[:k] + hard_negatives
    print(queries)
    return queries


def aggregate_data(response_dict: str):
    """
    Aggregate the retriever's response with the datasets
    and create dto classes of type Query that represent queries.
    :param response_dict: the response dictionary containing the query_id and the top k relevant docs
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
    for sub_json in data:
        if sub_json.get("_id") == query_id:
            query_contexts = [
                QueryContext(name=item[0], context=item[1])
                for item in sub_json.get("context")
            ]
            return Query(
                query_id,
                sub_json.get("answer"),
                sub_json.get("type"),
                sub_json.get("question"),
                query_contexts,
            )

    logging.info(f"Found zero matching query with id: {query_id}")
    return None


def find_document_by_id(data, doc_id: str):
    """
    Finds a document by id and retrieves a Document object
    """
    for key, sub_json in data.items():
        if key == doc_id:
            return Document(doc_id, sub_json.get("title"), sub_json.get("text"))

    logging.info(f"Found zero matching docs with id: {doc_id}")
    return None


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--sampling_method",
        "-s",
        type=str,
        choices=[e.value for e in SamplingMethod],
        default=SamplingMethod.RELEVANT.value,
        required=True,
        help="Sampling method to be used for the evaluation.",
    )
    args_parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=1,
        required=True,
        help="Number of relevant documents to be retrieved.",
    )
    args_parser.add_argument(
        "--retrieval_results_file",
        "-f",
        type=str,
        default="responseDict",
        required=False,
        help="Path to the retrieval results file to use during evaluation (must be in JSON format).",
    )
    args_parser.add_argument(
        "--num_queries",
        "-q",
        type=int,
        default=None,
        required=False,
        help="Number of queries to be evaluated.",
    )
    args_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        required=False,
        help="Flag to be verbose in logging intermediate results.",
    )
    args_parser.add_argument(
        "-a" "--not_adore",
        action="store_false",
        required=False,
        help="If ADORE is used as a retriever.",
    )
    args = args_parser.parse_args()

    # Input validation of the provided arguments
    if args.k < 1:
        raise ValueError(
            "The number of relevant documents to be retrieved should be at least 1."
        )
    if args.num_queries and args.num_queries < 1:
        raise ValueError("The number of queries to be evaluated should be at least 1.")
    if not os.path.exists(f"{args.retrieval_results_file}"):
        raise FileNotFoundError(
            f"The provided retrieval results file '{args.retrieval_results_file}' does not exist."
        )

    # Actually perform the evaluation using the provided arguments
    perform_evaluation(
        sampling_method=SamplingMethod(args.sampling_method),
        k=args.k,
        num_queries=args.num_queries,
        retrieval_results_file=args.retrieval_results_file,
        verbose=args.verbose,
    )
