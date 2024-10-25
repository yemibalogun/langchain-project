import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def get_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()
    return [embeddings.embed_text(chunk) for chunk in text_chunks]

def store_embeddings_faiss(embeddings):
    d = len(embeddings [0]) # dimension of the embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    return index

def answer_question(index, question, text_chunks, embeddings, model):
    question_embedding = OpenAIEmbeddings().embed_text(question)
    _, top_k_indices = index.search(np.array([question_embedding]).astype('float32'), k=5)
    relevant_chunks = [text_chunks[i] for i in top_k_indices[0]]
    
    context = "\n\n".join(relevant_chunks)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQueston: {question}"
    answer = model(prompt)
    return answer  

def ask_question_interface():
    st.title("PDF Manual Q&A App")
    uploaded_file = st.file_uploader("Upload a PDF manual", type="pdf")
    
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        text_chunks = split_text(pdf_text)
        embeddings = get_embeddings(text_chunks)
        index = store_embeddings_faiss(embeddings)
        
        question = st.text_input("Ask a question about the manual:")
        if question:
            answer = answer_question(index, question, text_chunks, embeddings, model=OpenAI())
            st.write(answer)
    