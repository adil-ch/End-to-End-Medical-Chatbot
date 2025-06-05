
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the context below to answer the question. 
If the answer is not in the context, say "I don't know".

Context: {context}
Question: {question}
Answer:"""
)