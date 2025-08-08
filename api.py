import os
import json
import requests
import uuid
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse

# --- 1. SETUP ---
class QueryRequest(BaseModel):
    url: str
    query: str

class BatchQueryRequest(BaseModel):
    url: str
    queries: List[str]

class Extraction(BaseModel):
    extracted_content: str = Field(description="The specific content extracted from the document that answers the user's query.")
    source_quote: str = Field(description="The exact source sentence from the document that was used to form the answer.")

app = FastAPI(title="Batch Extraction API")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 2. LLM and CORE LOGIC SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key="AIzaSyD1ffrRJp2uakby2t3hx1McpsPm8aZzCQM"
)
structured_llm = llm.with_structured_output(Extraction)

def highlight_pdf(pdf_path: str, quote: str) -> str:
    doc = fitz.open(pdf_path)
    clean_quote = " ".join(quote.split())
    for page in doc:
        areas = page.search_for(clean_quote)
        if areas:
            for inst in areas:
                page.add_highlight_annot(inst)
    unique_id = uuid.uuid4()
    highlighted_path = f"static/highlighted_{unique_id}.pdf"
    doc.save(highlighted_path, garbage=4, deflate=True, clean=True)
    doc.close()
    return highlighted_path

def get_pdf_text_and_path(url: str):
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
    Your task is to: 1. Analyze the full text and extract the part that directly answers the user's query.
    2. Provide the exact source sentence(s) from the document that justifies your answer.
    USER QUERY: "{user_query}"
    FULL DOCUMENT TEXT: "{page_content}"
    """,
    input_variables=["page_content", "user_query"],
)
chain = prompt | structured_llm

def cleanup_files(paths: list):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

# --- 3. API ENDPOINTS ---

# --- Endpoint 1: Single query with highlighting ---
@app.post("/process-document")
async def process_document(request: QueryRequest):
    if '.pdf' not in request.url.split('?')[0].lower():
        raise HTTPException(status_code=400, detail="This endpoint only works with PDF URLs.")
    page_content, temp_pdf_path = get_pdf_text_and_path(request.url)
    if not temp_pdf_path:
        raise HTTPException(status_code=400, detail=page_content)
    llm_result = chain.invoke({"page_content": page_content, "user_query": request.query})
    highlighted_file_path = highlight_pdf(temp_pdf_path, llm_result.source_quote)
    cleanup_task = BackgroundTask(cleanup_files, paths=[temp_pdf_path, highlighted_file_path])
    server_url = "http://127.0.0.1:8000"
    final_response = {"llm_answer": llm_result.dict(), "download_url": f"{server_url}/{highlighted_file_path}"}
    return final_response

# --- Endpoint 2: Multiple queries, JSON only ---
@app.post("/batch-extract")
async def batch_extract(request: BatchQueryRequest):
    page_content, temp_pdf_path = get_pdf_text_and_path(request.url)
    if not temp_pdf_path:
        raise HTTPException(status_code=400, detail=page_content)
    
    cleanup_task = BackgroundTask(cleanup_files, paths=[temp_pdf_path])
    all_results = []
    print(f"[LLM] Processing {len(request.queries)} queries in a batch...")
    for query in request.queries:
        llm_result = chain.invoke({"page_content": page_content, "user_query": query})
        all_results.append({"query": query, "answer": llm_result.dict()})
    
    return JSONResponse(content={"batch_results": all_results}, background=cleanup_task)