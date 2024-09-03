from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download NLTK resources
nltk.download('punkt')

DATA_PATH = "AI_rights.pdf"
DB_FAISS_PATH = "vectorstores/db_faiss"


# Create vector database
def create_vector_db():
    # Load the PDF file
    loader = UnstructuredPDFLoader(DATA_PATH)
    documents = loader.load()

    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-miniLM-L6-V2',
                                       model_kwargs={'device': 'cpu'})

    # Create FAISS vector store
    db = FAISS.from_documents(texts, embeddings)

    # Save the vector store locally
    db.save_local(DB_FAISS_PATH)


if __name__ == '__main__':
    create_vector_db()

