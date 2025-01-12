import os

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

access_token_read = os.environ["huggingface_token"]
login(token=access_token_read)


class LlamaEngine:

    def __init__(
        self,
        data,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        temperature=0.3,
        top_n=1,
        max_new_tokens=15,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    @staticmethod
    def format_prompt(query, documents):
        """
        Formats the input prompt for a retrieval-augmented generation (RAG) pipeline.

        Args:
            documents (list): List of dictionaries with 'title' and 'text' as keys.
            query (str): User's query.

        Returns:
            str: Formatted prompt for the chat template.
        """
        concatenated_string = "\n".join(
            f"Title: {doc['title']}\nContent: {doc['text']}" for doc in documents
        )

        prompt = (
            "<|begin_of_text|>\n"
            "System: Based on the documents below, answer directly the question.\n"
            f"Documents:\n{concatenated_string}\n\n"
            f"Question: {query}\n"
            "Answer: "
        )

        return prompt

    def get_llama_completion(self, user_prompt: str, documents):
        conversation = self.format_prompt(user_prompt, documents)
        inputs = self.tokenizer(
            conversation, return_tensors="pt", truncation=False, max_length=131072
        )

        output = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1000,
            temperature=0.3,
            top_p=0.9,
        )

        # Decode the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
