class Document:
    """
    Class to represent a Document object.
    A document contains its unique ID, title, as well as its text (to be used as context).
    """

    def __init__(self, id: str, title: str, text: str):
        self.id = id
        self.title = title
        self.text = text

    def print_obj(self):
        """Method to print the object."""
        print(f"Document(id = {self.id}, title={self.title}, text={self.text})")


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
    A query contains its unique ID, (gold) answer, type, question, as well as a list of documents and contexts.
    """

    def __init__(self, query_id: str, answer: str, type: str, question: str, contexts=None):
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
        """Method to add a QueryContext object to the list of contexts"""
        self.query_context.append(context)

    def add_result(self, result: str):
        self.result = result

    def context_to_dict(self):
        result = []
        for c in self.query_context:
            result.append({"title": c.name, "text": "".join(c.context)})
        return result

    def documents_to_dict(self):
        result = []
        for d in self.documents:
            result.append({"title": d.title, "text": d.text})
        return result

    def print_obj(self):
        """Method to print the object."""
        print(f"Query(query_id={self.query_id}, answer={self.answer}, "
              f"type={self.type}, query={self.question}, context={self.query_context}, documents={self.documents})")
