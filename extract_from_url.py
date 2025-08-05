import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
# --- NEW IMPORT ---
from langchain_community.document_loaders import WebBaseLoader

# --- REPLACED FUNCTION ---
def get_web_content(url: str) -> str:
    """Fetches the entire text content from a given URL."""
    print(f"\n[Real Tool] Fetching content from '{url}'...")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        # Combine the content of all loaded document parts into one string
        content = " ".join([doc.page_content for doc in documents])
        # Replace multiple newlines/spaces for cleaner processing
        return " ".join(content.split())
    except Exception as e:
        print(f"Error fetching URL content: {e}")
        return f"Could not fetch content from the URL due to an error: {e}"

# --- 1. DEFINE THE LLM and OUTPUT PARSER ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key="AIzaSyAA9Xo0cHf_VwyA5_jae82hyBoADvp0z48"  # <-- PASTE YOUR GEMINI API KEY HERE
)

# --- 2. DEFINE THE PROMPT AND THE DESIRED JSON STRUCTURE ---
parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
    You are an expert at summarizing and extracting key information.
    A user wants to know about a specific topic from a webpage.
    Analyze the full text from the webpage provided below and extract only the part that answers the user's query.

    USER QUERY:
    "{user_query}"

    FULL WEBPAGE TEXT:
    "{page_content}"

    Return your answer in the following JSON format. The key should be "extracted_content".
    {format_instructions}
    """,
    input_variables=["page_content", "user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- 3. BUILD THE EXTRACTION CHAIN ---
# Note: This is slightly different now. We pass the content and query to the prompt.
chain = prompt | llm | parser

# --- 4. RUN THE SYSTEM ---
if __name__ == '__main__':
    # --- User Inputs ---
    target_url = "https://en.wikipedia.org/wiki/India" # Example URL
    user_query = "What is the etymology and name of India?" # Example query

    # --- Step 1: Use the real web loader to get the page content ---
    extracted_information = get_web_content(url=target_url)

    # --- Step 2: Use the LLM chain to extract the specific part ---
    if "Could not fetch" not in extracted_information:
        print("\n[LLM] Extracting the specific information based on the query...")
        final_result = chain.invoke({
            "page_content": extracted_information,
            "user_query": user_query
        })
    else:
        final_result = {"extracted_content": extracted_information}

    # --- Step 3: Print the final, structured response ---
    print("\n--- Final Structured Output ---")
    print(json.dumps(final_result, indent=2))