from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = "IDtW285O1k6vzHO0Z7O2C8Awc1b1ftosFvhh3225"

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k" :3})

llm = Cohere(model="command", temperature=0.5)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,  # Your vector DB retriever
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get" , methods = ["GET" , "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"query" : msg})
    print(response["result"]) 
    return str(response["result"])

if __name__ == '__main__':
    app.run( host = '0.0.0.0' , port = 8080 , debug = True )

