import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils import extract_video_id, get_video_info, get_transcript_url
from summarizer import get_prompt_template, summarize_content
from llama_index.readers.web import BeautifulSoupWebReader
from langchain.schema import Document
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Page configuration
st.set_page_config(page_title="YT/URL Summarizer", page_icon="🧠", layout="centered")

# Load and apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🔐 API Configuration")
    api_key_input = st.text_input("Enter API Key (Groq, OpenAI)", type="password")

    provider = None
    model_options = []

    if api_key_input:
        if api_key_input.startswith("gsk_"):
            provider = "groq"
            model_options = ["Llama3-70B-8192", "deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.3-70b-versatile"]
        elif api_key_input.startswith("sk-"):
            provider = "openai"
            model_options = ["gpt-4o-mini","gpt-3.5-turbo", "gpt-4.1", "gpt-4o"]
        else:
            st.warning("Unknown API key format. Use a valid Groq, OpenAI, or HuggingFace key.")

    selected_model = st.selectbox("Select Model", model_options) if model_options else None
    temp = st.slider("Temperature", 0.0, 1.0, 0.4)
    chunk_size = st.slider("Chunk Size", 500, 4000, 3000, step=200)
    overlap = st.slider("Chunk Overlap", 0, 500, 100, step=10)
    st.markdown("---")
    st.markdown("🛠️ **LangChain Summarizer**\nBuilt with ❤️ by OpenAI/Groq")
    
api_key = api_key_input

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>📺📰 LangChain Summarizer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: grey;'>Summarize YouTube videos or website articles using LLMs</p>",
    unsafe_allow_html=True
)

# --- MAIN INPUT AREA ---
st.markdown("### 🔗 Enter a YouTube or Website URL")
url = st.text_input("Provide your Url", placeholder="https://www.XYZ.com/v=...", label_visibility="collapsed")


# Prepare LLM based on provider
llm = None
if provider == "groq":
    llm = ChatGroq(temperature=temp, model_name=selected_model, groq_api_key=api_key_input)
elif provider == "openai":
    llm = ChatOpenAI(temperature=temp, model_name=selected_model, openai_api_key=api_key_input)
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

            docs = get_transcript_url(video_id)
            if docs:
                try:
                    summary = summarize_content(docs, llm, prompt, chunk_size, overlap)
                    st.success("✅ Summary generated successfully!")
                    st.markdown("### 📝 Summary")
                    st.markdown(f"<p style='color: grey; font-size: 0.9em;'>🔧 Powered by <b>{provider.upper()}</b> - Model: <i>{selected_model}</i></p>", unsafe_allow_html=True)
                    st.markdown(summary)
                    with st.expander("📄 View Transcript"):
                        st.write(docs[0].page_content)
                except Exception as e:
                    st.error("❌ Token size too large for this transcript. Either use other LLM model with large context size or use another URL ✅")
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
                st.markdown(f"<p style='color: grey; font-size: 0.9em;'>🔧 Powered by <b>{provider.upper()}</b> - Model: <i>{selected_model}</i></p>", unsafe_allow_html=True)
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Failed to summarize URL: {str(e)}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>© 2025 LangChain Summarizer</p>", unsafe_allow_html=True)
