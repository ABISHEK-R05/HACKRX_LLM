from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your sample PDF
pdf_path = "documents/sample_policy.pdf"

# Create a loader instance
loader = PyPDFLoader(pdf_path)

# Load the document. This returns a list of 'Document' objects, one for each page.
pages = loader.load()
print(f"Successfully loaded {len(pages)} pages.")

# --- NEW: CHUNKING THE DOCUMENT ---

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # The maximum size of a chunk (in characters)
    chunk_overlap=200, # The number of characters to overlap between chunks
    length_function=len
)

# Split the loaded pages into chunks
chunks = text_splitter.split_documents(pages)
print(f"Split the document into {len(chunks)} chunks.")

# --- LETS INSPECT A CHUNK ---
print("\n--- Content of First Chunk ---")
print(chunks[0].page_content)
print("\n--- Metadata of First Chunk ---")
print(chunks[0].metadata)