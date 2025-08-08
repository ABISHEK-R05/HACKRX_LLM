import os
import json
import requests
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader

# --- 1. THE CORE LOGIC ---

def get_pdf_text_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    print(f"\n[Tool] Downloading and parsing PDF from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        temp_pdf_path = "temp_policy.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        
        os.remove(temp_pdf_path)
        return " ".join(text.split())
        
    except Exception as e:
        print(f"Error processing PDF from URL: {e}")
        return f"Could not process the PDF from the URL due to an error: {e}"

# --- 2. LLM SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key="AIzaSyD1ffrRJp2uakby2t3hx1McpsPm8aZzCQM"
)

parser = JsonOutputParser()
prompt = PromptTemplate(
    template="""
    You are an expert at analyzing policy documents. A user wants to know about a specific topic from the provided document.
    Analyze the full text from the document below and extract only the part that answers the user's query.

    USER QUERY: "{user_query}"
    FULL DOCUMENT TEXT: "{document_content}"

    Return your answer in the following JSON format. The key should be "extracted_content".
    {format_instructions}
    """,
    input_variables=["document_content", "user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | llm | parser

# --- 3. RUN THE SYSTEM ---
if __name__ == '__main__':
    # --- Get User Inputs ---
    target_url = input("Please enter the URL of the PDF: ")
    user_query = input("What information do you want to extract? ")

    # --- Step 1: Get the PDF text from the URL ---
    document_text = get_pdf_text_from_url(url=target_url)

    # --- Step 2: Use the LLM to extract the specific part ---
    if "Could not process" not in document_text:
        print("\n[LLM] Extracting the specific information based on the query...")
        final_result = chain.invoke({
            "document_content": document_text,
            "user_query": user_query
        })
    else:
        final_result = {"extracted_content": document_text}

    # --- Step 3: Print the final, structured response ---
    print("\n--- Final Structured Output ---")
    print(json.dumps(final_result, indent=2))