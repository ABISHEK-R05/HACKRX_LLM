import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI # <-- CHANGED IMPORT
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. LOAD OUR KNOWLEDGE BASE ---
print("Loading knowledge base...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
print("Knowledge base loaded successfully.")


# --- 2. DEFINE THE PROMPT TEMPLATE ---
template = """
You are an expert insurance claims adjudicator AI. 
Your task is to evaluate a claim based on the user's query and the provided policy clauses.
You also need to summarize the relevant policy clauses that support your decision at the last as summarization.
First, analyze the user's query to understand the key details.
Then, carefully review the following retrieved policy clauses to make your decision:
<context>
{context}
</context>

Based on your analysis, determine if the claim should be Approved or Rejected.
Provide your final answer ONLY in the following JSON format:

{{
  "decision": "Approved" | "Rejected" | "Pending Information",
  "amount_covered": <number> | null,
  "justification": [
    {{
      "clause_reference": "Clause X.Y",
      "reasoning": "A detailed explanation of how this specific clause applies to the user's query, leading to the decision."
    }}
  ]
}}

User Query: {question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# --- 3. DEFINE THE LLM and OUTPUT PARSER ---
# --- THIS IS THE MODIFIED PART FOR GEMINI ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # A fast and capable model
    temperature=0,
    google_api_key="AIzaSyAA9Xo0cHf_VwyA5_jae82hyBoADvp0z48"  # <-- PASTE YOUR GEMINI API KEY HERE
)
parser = JsonOutputParser()


# --- 4. BUILD THE RAG CHAIN ---
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# --- 5. INVOKE THE CHAIN WITH A SAMPLE QUERY ---
if __name__ == '__main__':
    print("\n--- Processing Query ---")
    query = "I am a 46-year-old male who had knee surgery in Pune. My insurance policy is only 3 months old. Is my claim covered?"
    
    print(f"Query: {query}\n")
    
    # Run the chain
    result = chain.invoke(query)
    
    # Print the structured JSON result
    import json
    print("--- Result ---")
    print(json.dumps(result, indent=2))