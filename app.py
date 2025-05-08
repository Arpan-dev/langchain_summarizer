import os
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from utils import extract_video_id, get_video_info, get_transcript
from summarizer import get_prompt_template, summarize_content
from llama_index.readers.web import BeautifulSoupWebReader
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="YT/URL Summarizer", page_icon="🧠", layout="centered")

# Load and apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🔐 API Configuration")
    api_key = st.text_input("Enter GROQ API Key", type="password")
    model = st.selectbox("Select Model", ["gemma2-9b-it", "llama-3.3-70b-versatile"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.3)
    chunk_size = st.slider("Chunk Size", 500, 4000, 2000, step=100)
    overlap = st.slider("Chunk Overlap", 0, 500, 100, step=10)
    st.markdown("---")
    st.markdown("🛠️ **LangChain Summarizer**\nBuilt with ❤️ by OpenAI")

# This will fetch from secrets in Streamlit Cloud or fall back to .env locally
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
else:
    st.error("❌ Please set your Groq API Key in Streamlit secrets or .env")

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>📺📰 LangChain Summarizer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: grey;'>Summarize YouTube videos or website articles using LLMs</p>",
    unsafe_allow_html=True
)

# --- MAIN INPUT AREA ---
st.markdown("### 🔗 Enter a YouTube or Website URL")
url = st.text_input("Provide your Url", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

# Prepare LLM
llm = ChatGroq(temperature=temp, model_name=model)
prompt = get_prompt_template()

# --- MAIN ACTION BUTTON ---
if st.button("🚀 Generate Summary"):
    if not api_key:
        st.error("❗ Please enter your API key in the sidebar.")
    elif "youtube.com" in url or "youtu.be" in url:
        with st.spinner("📼 Processing YouTube Video..."):
            video_id = extract_video_id(url)
            title, author = "Unknown", "Unknown"
            try:
                title, author = get_video_info(url)
            except:
                pass

            st.markdown("### 🎬 Video Information")
            st.markdown(f"- **Title:** {title}")
            st.markdown(f"- **Author:** {author}")

            docs = get_transcript(video_id, url)
            if docs:
                summary = summarize_content(docs, llm, prompt, chunk_size, overlap)
                st.success("✅ Summary generated successfully!")
                st.markdown("### 📝 Summary")
                st.markdown(summary)
                with st.expander("📄 View Transcript"):
                    st.write(docs[0].page_content)
            else:
                st.error("❌ No transcript found for this video.")
    else:
        with st.spinner("🌐 Processing Website..."):
            try:
                loader = BeautifulSoupWebReader()
                raw_docs = loader.load_data(urls=[url])
                docs = [Document(page_content=doc.text) for doc in raw_docs]
                summary = summarize_content(docs, llm, prompt, chunk_size, overlap)
                st.success("✅ Summary generated successfully!")
                st.markdown("### 📝 Summary")
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Failed to summarize URL: {str(e)}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>© 2025 LangChain Summarizer</p>", unsafe_allow_html=True)
