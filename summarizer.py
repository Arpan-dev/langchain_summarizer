from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
""" 
Description: Function to set the prompt template for the LLM to act.
"""

def get_prompt_template():
    return PromptTemplate(
        template="""
                    You are a multilingual expert assistant skilled in translation and summarization.

                    Your task is to:
                    1. Detect the language of the given content.
                    2. If it's not in English, translate it into fluent English.
                    3. Summarize the translated content thoroughly while preserving all important details.

                    Instructions for the summary:
                    - Start with a **brief overview** of the topic and purpose.
                    - Include **key takeaways** as clear bullet points.
                    - Use **headings and subheadings** to organize information.
                    - Maintain a professional and accessible tone for a general audience.
                    - If the content is long, provide a detailed summary — do not limit output length unnecessarily.

                    Input Content:
                    {text}

                    Your response:
                    """,
                    input_variables=["text"]
                    
                )

""" 
Description: Function to summarize the docs extracted and then chaining the prompt, llm to perform the summarization.
"""
def summarize_content(docs, llm, prompt, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(docs)
    chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain.invoke(docs_split)['output_text']
