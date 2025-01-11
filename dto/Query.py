class Document:
    def __init__(self, id: str, title: str, text: str):
        self.id = id
        self.title = title
        self.text = text

    def printObj(self):
        print(f"Document(id = {self.id}, title={self.title}, text={self.text})")

class QueryContext:
    def __init__(self, name: str, context: list):
        self.name = name
        self.context = context

    def printObj(self):
        print(f"QueryContext(name={self.name}, context={self.context})")

class Query:
    def __init__(self, query_id: str, answer: str, type: str, question: str, contexts = []):
        self.query_id = query_id
        self.answer = answer
        self.type = type
        self.question = question
        self.documents = []
        self.query_context = contexts
        self.result = "" # here comes the answer of the llm to this query

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
    
    def printObj(self):
        print(f"Query(query_id={self.query_id}, answer={self.answer}, "
                f"type={self.type}, query={self.question}, context={self.query_context}, documents={self.documents})")
    