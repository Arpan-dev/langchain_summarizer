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
                You are a highly sophisticated AI summarization expert with subject matter expertise across multiple domains. Your task is to create comprehensive yet concise summaries that capture the essence and key information from any content.

                CONTENT TO SUMMARIZE:
                {text}

                ANALYSIS INSTRUCTIONS:
                1. First, identify the language of the content. If not English, translate to fluent, natural English.
                2. Determine the content type (video transcript, article, technical document, etc.) and adapt your summarization approach accordingly.
                3. Identify the primary topic, key arguments, main conclusions, and most important supporting evidence.
                4. Note any critical numerical data, statistics, dates, or specific claims.
                5. Recognize the intended audience and purpose of the original content.

                SUMMARIZATION FORMAT:
                
                ## EXECUTIVE SUMMARY (2-3 sentences providing the essential overview)

                ## CONTENT TYPE AND CONTEXT
                - Source type: [Video/Article/Document/etc.]
                - Primary topic: [Main subject matter]
                - Target audience: [Who the content appears to be created for]
                
                ## KEY POINTS
                - Present 3-7 bullet points capturing the most important information
                - Each point should be concise (1-2 sentences) but specific
                - Include any critical numbers, statistics, or specific claims
                
                ## DETAILED BREAKDOWN
                - Organize this section with clear headings reflecting the content's structure
                - Include relevant details, examples, and supporting evidence
                - Preserve the logical flow and relationships between ideas
                - Maintain nuance in arguments or complex topics
                
                ## CONCLUSIONS & IMPLICATIONS
                - Summarize the main takeaways or conclusions
                - Note any calls to action or future directions mentioned
                - If applicable, mention limitations or considerations

                IMPORTANT GUIDELINES:
                - Maintain objectivity and do not inject personal opinions
                - Preserve technical accuracy - never oversimplify complex concepts
                - Scale the summary length to match content complexity (longer for dense/complex content)
                - Use clear, accessible language while preserving domain-specific terminology when necessary
                - For instructional content, ensure key steps or processes are preserved in order
                - For scientific/academic content, accurately represent methodologies and findings
                - For opinion pieces, clearly distinguish opinions from factual claims
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
