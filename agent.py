from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from LangGraph import build_graph
import chromadb


class Agent:
    def __init__(self, model_type="openai", prompt="Be a helpful assistant", tools=None):
        if model_type == "openai":
            self.model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        elif model_type == "groq":
            self.model = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
        else:
            raise ValueError("Unsupported model type. Please choose 'openai' or 'groq'.")
        
        self.tools = tools if tools else []
        self.system_prompt = prompt
        
        # Build the graph using the function build_graph
        self.graph = build_graph(llm=self.model, prompt=self.system_prompt, tools=self.tools)

    def invoke(self, messages):
        # Initialize the state with the provided messages
        initial_state = {"messages": messages}
        
        # Run the graph synchronously and obtain the output
        graph_output = self.graph.invoke(initial_state)
        
        return graph_output["messages"]

    @staticmethod
    def initialize_vector_db(directory="docs/", collection_name="pdf_documents", persist_directory=".chromadb/"):
        # Use PersistentClient as per the updated ChromaDB documentation
        client = chromadb.PersistentClient(path=persist_directory)
        embedding_func = OpenAIEmbeddings()
        collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)

        # Check if the collection already has documents
        if collection.count() > 0:
            print("Vector database already initialized.")
            return collection

        # Load PDFs and add to ChromaDB if the collection is empty
        print("Initializing vector database...")
        documents = load_pdfs(directory)
        collection.add_documents(documents)
        return collection

# Function to load PDFs (can be placed in a utility module)
def load_pdfs(directory):
    from langchain.document_loaders import PyPDFLoader
    pdf_loader = PyPDFLoader(directory)
    return pdf_loader.load_and_split()