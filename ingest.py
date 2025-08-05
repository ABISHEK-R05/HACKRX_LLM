from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the document
def load_document(file_path):
    print("Loading document...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")
    return pages

# 2. Chunk the document
def chunk_document(pages):
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split the document into {len(chunks)} chunks.")
    return chunks

# 3. Embed and Store the chunks
def create_vector_store(chunks):
    print("Creating vector store... (This may take a moment)")
    # Use a pre-trained sentence-transformer model to create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Use Chroma to create a vector store from the chunks and embeddings
    # We will persist this to disk so we don't have to re-create it every time
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./db" # The directory to save the database
    )
    print("Vector store created successfully.")
    return vectorstore

if __name__ == '__main__':
    pdf_path = "documents/sample_policy.pdf"
    
    pages = load_document(pdf_path)
    chunks = chunk_document(pages)
    create_vector_store(chunks)