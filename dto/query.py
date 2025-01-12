"""
Module providing classes to represent a Query object,
a Document object, and a QueryContext object.
"""


class Document:
    """
    Class to represent a Document object.
    A document contains its unique ID, title, as well as its text (to be used as context).
    """

    def __init__(self, uid: str, title: str, text: str):
        self.uid = uid
        self.title = title
        self.text = text

    def print_obj(self):
        """Method to print the object."""
        print(f"Document(id = {self.uid}, title={self.title}, text={self.text})")


class QueryContext:
    """
    Class to represent a QueryContext object.
    A query context contains its unique name and a list of contexts.
    """

    def __init__(self, name: str, context: list):
        self.name = name
        self.context = context

    def print_obj(self):
        """Method to print the object."""
        print(f"QueryContext(name={self.name}, context={self.context})")


class Query:
    """
    Class to represent a Query object.
    A query contains its unique ID, (gold) answer, type, question,
    as well as a list of documents and contexts.
    """

    def __init__(
        self, query_id: str, answer: str, type: str, question: str, contexts=None
    ):
        if contexts is None:
            contexts = []
        self.query_id = query_id
        self.answer = answer
        self.type = type
        self.question = question
        self.documents = []
        self.query_context = contexts
        self.result = ""  # here comes the answer of the llm to this query

    def add_document(self, document: Document):
        """Method to add a Document object to the list of documents"""
        self.documents.append(document)

    def add_context(self, context: QueryContext):
        """Method to add a QueryContext object to the list of contexts."""
        self.query_context.append(context)

    def set_result(self, result: str):
        """Method to set the result of the LLM to the query object."""
        self.result = result

    def context_to_dict(self, k):
        """
        Method to convert the list of provided contexts to a list of dictionaries,
        wherein each dictionary contains the name and text of the query context.
        :return: List of dictionaries containing the name and text of the top K query documents.
        """

        result = []
        for c in self.query_context[:k]:
            result.append({"title": c.name, "text": "".join(c.context)})
        return result

    def documents_to_dict(self):
        """
        Method to convert the list of provided documents to a list of dictionaries,
        wherein each dictionary contains the title and text of the document.
        :returns: List of dictionaries containing the title and text of the document.
        """

        result = []
        for d in self.documents:
            result.append({"title": d.title, "text": d.text})
        return result

    def print_obj(self):
        """Method to print the object."""

        print(
            f"Query(query_id={self.query_id}, answer={self.answer}, "
            f"type={self.type}, query={self.question}, context={self.query_context}, "
            f"documents={self.documents})"
        )
