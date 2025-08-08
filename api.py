import os
import json
import requests
import uuid
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from starlette.background import BackgroundTask

# --- 1. SETUP ---
class QueryRequest(BaseModel):
    url: str
    query: str

class Extraction(BaseModel):
    extracted_content: str = Field(description="The specific content extracted from the document that answers the user's query.")
    source_quote: str = Field(description="The exact source sentence from the document that was used to form the answer.")

app = FastAPI(title="PDF Highlighting API")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 2. LLM and CORE LOGIC SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key="AIzaSyD1ffrRJp2uakby2t3hx1McpsPm8aZzCQM"  # <-- PASTE YOUR GEMINI API KEY HERE
)
structured_llm = llm.with_structured_output(Extraction)

# --- FINAL, SIMPLIFIED HIGHLIGHTING FUNCTION ---
def highlight_pdf(pdf_path: str, quote: str) -> str:
    """Highlights all occurrences of a quote in a PDF and saves to a new file."""
    doc = fitz.open(pdf_path)
    # Clean the quote by collapsing all whitespace into single spaces.
    # This makes the search much more robust to variations in newlines and spacing.
    clean_quote = " ".join(quote.split())
    
    print(f"[Highlighter] Searching for full quote: '{clean_quote[:70]}...'")
    
    for page in doc:
        # search_for returns a list of fitz.Rect objects for every match
        areas = page.search_for(clean_quote)
        
        if areas:
            print(f"[Highlighter] Found full quote on page {page.number + 1}. Applying highlight.")
            # Iterate through all found areas and apply a highlight annotation
            for inst in areas:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

    # Save the modified document to a new file
    unique_id = uuid.uuid4()
    highlighted_path = f"static/highlighted_{unique_id}.pdf"
    doc.save(highlighted_path, garbage=4, deflate=True, clean=True)
    doc.close()
    return highlighted_path

def get_pdf_text_and_path(url: str):
    """Downloads a PDF and returns its text and local path."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        temp_pdf_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return " ".join(text.split()), temp_pdf_path
    except Exception as e:
        return f"Could not process PDF: {e}", None

prompt = PromptTemplate(
    template="""
    You are an expert at analyzing documents. A user wants to know about a specific topic from the document text provided below.
    Your task is to:
    1. Analyze the full text and extract the part that directly answers the user's query.
    2. Provide the exact source sentence(s) from the document that justifies your answer.

    USER QUERY: "{user_query}"
    FULL DOCUMENT TEXT: "{page_content}"
    """,
    input_variables=["page_content", "user_query"],
)

chain = prompt | structured_llm

def cleanup_files(paths: list):
    """Task to delete files in the background after response is sent."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

# --- API ENDPOINT ---
@app.post("/process-document")
async def process_document(request: QueryRequest):
    if '.pdf' not in request.url.split('?')[0].lower():
        raise HTTPException(status_code=400, detail="This endpoint only works with PDF URLs.")

    page_content, temp_pdf_path = get_pdf_text_and_path(request.url)
    if not temp_pdf_path:
        raise HTTPException(status_code=400, detail=page_content)
    
    print("[LLM] Extracting information...")
    llm_result = chain.invoke({"page_content": page_content, "user_query": request.query})
    
    highlighted_file_path = highlight_pdf(temp_pdf_path, llm_result.source_quote)
    
    cleanup_task = BackgroundTask(cleanup_files, paths=[temp_pdf_path, highlighted_file_path])
    
    server_url = "http://127.0.0.1:8000"
    final_response = {
        "llm_answer": llm_result.dict(),
        "download_url": f"{server_url}/{highlighted_file_path}"
    }
    
    return final_response