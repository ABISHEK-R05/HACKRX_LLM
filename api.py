import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader

# --- 1. SETUP ---

# Define the input data structure for our API
class QueryRequest(BaseModel):
    url: str
    query: str

# Create the FastAPI app
app = FastAPI(
    title="LLM Document Extractor API",
    description="An API to extract specific information from a webpage using an LLM."
)

# Load the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key="AIzaSyAA9Xo0cHf_VwyA5_jae82hyBoADvp0z48"  # <-- PASTE YOUR GEMINI API KEY HERE
)

# --- 2. THE CORE LOGIC (from your script) ---

def get_web_content(url: str) -> str:
    """Fetches text content from a URL."""
    try:
        loader = WebBaseLoader(url)
        content = " ".join([doc.page_content for doc in loader.load()])
        return " ".join(content.split())
    except Exception as e:
        return f"Error fetching URL: {e}"

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
    You are an expert at summarizing and extracting key information.
    A user wants to know about a specific topic from a webpage.
    Analyze the full text from the webpage provided below and extract only the part that answers the user's query.

    USER QUERY: "{user_query}"
    FULL WEBPAGE TEXT: "{page_content}"

    Return your answer in the following JSON format. The key should be "extracted_content".
    {format_instructions}
    """,
    input_variables=["page_content", "user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

# --- 3. DEFINE THE API ENDPOINT ---

@app.post("/extract")
async def extract_information(request: QueryRequest):
    """
    Takes a URL and a query, and returns the extracted information.
    """
    print(f"Received request for URL: {request.url}")
    page_content = get_web_content(request.url)

    if "Error fetching" in page_content:
        return {"error": page_content}

    result = chain.invoke({
        "page_content": page_content,
        "user_query": request.query
    })
    return result