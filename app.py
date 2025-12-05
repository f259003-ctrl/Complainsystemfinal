import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
from utils.ingest import extract_pdf_text, chunk_text
from utils.vectorstore import build_vectorstore, retrieve
from utils.compliance import check_rule

# -----------------------------------
# Streamlit Config
# -----------------------------------
st.set_page_config(page_title="Contract Compliance Checker", layout="wide")
st.title("📄 Contract Compliance Checker (Gemini + RAG)")
st.write("Upload a contract PDF and evaluate compliance automatically.")

# -----------------------------------
# Load Gemini API
# -----------------------------------
API_KEY = st.secrets["AIzaSyDvxtiL9asne6fZv8ZxuMEweq_C5-jSMuw"]
genai.configure(api_key=API_KEY)

# Load Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# File Uploads
# -----------------------------------
pdf_file = st.file_uploader("Upload PDF contract", type=["pdf"])
rules_file = st.file_uploader("Upload Rules JSON", type=["json"])

if pdf_file and rules_file:

    # Save PDF temporarily
    pdf_path = "/tmp/contract.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    st.info("Extracting PDF...")
    text = extract_pdf_text(pdf_path)

    # Chunk & Build Vectorstore
    chunks = chunk_text(text)
    index, vectors = build_vectorstore(chunks, embedder)

    rules = json.loads(rules_file.read().decode())

    st.success("PDF processed successfully!")

    st.header("Compliance Results")

    results = []
    for rule in rules:
        st.subheader(f"Rule: {rule['name']}")
        with st.spinner("Checking..."):
            result = check_rule(rule, index, vectors, chunks, embedder)
        st.write(result)
        results.append({"rule": rule["name"], "result": result})

    st.download_button(
        "⬇️ Download Results",
        data=json.dumps(results, indent=2),
        file_name="compliance_results.json",
        mime="application/json"
    )
