# import os
# import requests
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Weaviate
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# import weaviate
# from weaviate.embedded import EmbeddedOptions
# from dotenv import load_dotenv, find_dotenv

# # Load OpenAI API key from .env file

# # Step 1: Load and chunk data
# url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
# res = requests.get(url)
# with open("state_of_the_union.txt", "w") as f:
#     f.write(res.text)

# # Load the data
# loader = TextLoader('./state_of_the_union.txt')
# documents = loader.load()

# # Chunk the data
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)

# # Step 2: Setup vector database with Weaviate
# client = weaviate.Client(
#     embedded_options=EmbeddedOptions()
# )

# # Step 3: Generate vector embeddings for each chunk using OpenAI
# vectorstore = Weaviate.from_documents(
#     client=client,
#     documents=chunks,
#     embedding=OpenAIEmbeddings(),
#     by_text=False
# )

# # Step 4: Define retriever for semantic search
# retriever = vectorstore.as_retriever()

# # Step 5: Define the RAG pipeline components
# # Define LLM (ChatOpenAI)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Define prompt template
# template = """You are an assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know. 
# Use two sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
# """

# prompt = ChatPromptTemplate.from_template(template)

# # Step 6: Setup RAG pipeline
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()} 
#     | prompt 
#     | llm 
#     | StrOutputParser()
# )

# # Example usage of the RAG system
# def ask_question(question):
#     return rag_chain.invoke({"question": question})

# # Example question
# response = ask_question("What did the president say about Intel's CEO?")
# print(response)



import os
import getpass
import bs4
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Set up the OpenAI API key
# Define a directory to cache the vector store
vectorstore_dir = "./vectorstore_cache"

# Step 2: Load the blog content using WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),  # URL of the blog
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()  # Load the documents from the webpage

# Step 3: Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Step 4: Check if the vector store exists
if os.path.exists(vectorstore_dir):
    print("Loading vector store from cache...")
    vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=OpenAIEmbeddings())
else:
    print("Creating and caching new vector store...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=vectorstore_dir)
    # vectorstore.persist()  # Save the vector store to disk

# Step 5: Set up a retriever to get relevant content based on user queries
retriever = vectorstore.as_retriever()

# Step 6: Load the RAG prompt template from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Helper function to format the documents into a string for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 7: Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Chain retriever and format
    | prompt  # Use the loaded prompt template
    | ChatOpenAI(model="gpt-4o-mini")  # LLM: ChatOpenAI model
    | StrOutputParser()  # Output parsing to string
)

# Step 8: Query the RAG system with a question
response = rag_chain.invoke("What is Task Decomposition?")
print(response)

# Cleanup is optional if you're caching
