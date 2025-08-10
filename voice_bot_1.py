#!/usr/bin/env python
# coding: utf-8

# In[1]:

__import__('pysqlite3')
import sys
sys.modules['sqlite3']=sys.modules.pop('pysqlite3')


import pysqlite3 as sqlite3

import os
import tempfile
import streamlit as st
import whisper
from gtts import gTTS

# Disable Chroma telemetry
os.environ['CHROMA_DISABLE_TELEMETRY'] = '1'

# LangChain and RAG
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from st_audiorec import st_audiorec  # pip install streamlit-audiorec

# --- Config ---
PDF_FILE = "menu.pdf"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
CHROMA_DB_DIR = "chroma_db"
WHISPER_MODEL_NAME = "base"

# --- Load Whisper Model ---
@st.cache_resource(show_spinner="Loading Whisper model...")
def load_whisper():
    try:
        return whisper.load_model(WHISPER_MODEL_NAME)
    except Exception as e:
        st.error(f"Whisper model loading failed: {e}")
        st.stop()

# --- Initialize RAG Pipeline ---
@st.cache_resource(show_spinner="Setting up RAG pipeline...")
def init_rag():
    if not os.path.exists(PDF_FILE):
        st.error(f"PDF file not found: {PDF_FILE}")
        st.stop()

    try:
        loader = UnstructuredPDFLoader(file_path=PDF_FILE)
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            collection_name="local-rag",
            embedding_function=embeddings
        )
    else:
        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            collection_name="local-rag",
            persist_directory=CHROMA_DB_DIR
        )

    try:
        llm = ChatOllama(model=LLM_MODEL)
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        st.stop()

    # Prompt for multi-query
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Generate five rephrasings of: {question}"
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm,
        prompt=prompt
    )

    # Final RAG chain
    answer_template = "Use only the context below to answer the question.\n{context}\nQuestion: {question}"

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(answer_template)
        | llm
        | StrOutputParser()
    )

    return chain

# --- Load Models ---
whisper_model = load_whisper()
rag_chain = init_rag()

# --- Streamlit UI ---
st.title("🎙️ TALK_2_ORDER [VoiceBot RAG]")
st.write("Click record and ask a question about the PDF using your voice.")

audio_bytes = st_audiorec()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio_path = tmp_audio.name

    try:
        if os.path.getsize(tmp_audio_path) < 1024:
            st.warning("Audio too short or empty. Try again.")
            st.stop()

        with st.spinner("Transcribing your question..."):
            try:
                result = whisper_model.transcribe(tmp_audio_path)
                query = result.get("text", "").strip()
            except Exception as e:
                st.error(f"Whisper transcription failed: {e}")
                st.stop()

        if not query:
            st.warning("Speech not recognized. Please try again.")
            st.stop()

        st.markdown(f"**You asked:** {query}")

        with st.spinner("Generating answer..."):
            try:
                response = rag_chain.invoke({"question": query})
            except Exception as e:
                st.error(f"RAG pipeline failed: {e}")
                st.stop()

        st.markdown(f"**Bot says:** {response}")

        # Text-to-Speech
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                tts = gTTS(text=response)
                tts.save(tts_file.name)
                tts_file.seek(0)
                audio_response = tts_file.read()
            st.audio(audio_response, format="audio/mp3")
        except Exception as e:
            st.error(f"TTS failed: {e}")

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


# In[ ]:




