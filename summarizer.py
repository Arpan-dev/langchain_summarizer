from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def get_prompt_template():
    return PromptTemplate(
        template="""
        You are an expert content summarizer specializing in extracting key insights from videos and websites.
        Your task is to analyze the provided content and produce a clear, concise summary. 
        Your output should include:
        - A brief overview of the video's or website's topic and purpose.
        - Key takeaways presented as bullet points.
        - Use of headers and subheaders to logically group related points.
        - Clear, informative language suitable for a general audience.
        Content:{text}
        """,
        input_variables=["text"]
    )

def summarize_content(docs, llm, prompt, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(docs)
    chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain.invoke(docs_split)['output_text']
