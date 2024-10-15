

import os
import shutil
import requests
import nest_asyncio
import asyncio
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv
import pandas as pd
from ragas.dataset_schema import EvaluationDataset
# Apply nest_asyncio to avoid event loop issues
nest_asyncio.apply()

# Check if the repository already exists; if not, download it
repo_dir = "Sample_Docs_Markdown"

if not os.path.exists(repo_dir):
    os.system(f"git clone https://huggingface.co/datasets/explodinggradients/Sample_Docs_Markdown")
else:
    print(f"{repo_dir} already exists, skipping download.")

# Load documents using DirectoryLoader
path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

# Load OpenAI API key from environment variables or .env file
load_dotenv()  # Ensure you have a .env file with

os.environ["OPENAI_API_KEY"] = ""  # Replace 'your-openai-key' if not using .env

# Wrap the LLM with LangchainLLMWrapper using GPT-4
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# Generate the test set with the loaded documents
generator = TestsetGenerator(llm=evaluator_llm)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

# Export and inspect the generated testset as a Pandas DataFrame
df = dataset.to_pandas()
print(df)

# Optionally, save to a CSV file for further inspection
df.to_csv("generated_testset.csv", index=False)


