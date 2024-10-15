
# Load the required dataset
from datasets import load_dataset
# Select required evaluation metrics from Ragas
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate

# Wrapping OpenAI GPT-4 model in LangchainLLMWrapper for evaluation
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Load dataset from huggingface hub
dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")

# Print a few samples from the dataset
print("Sample data points from the dataset:")
print(dataset["eval"][0])  # First sample
print(dataset["eval"][1])  # Second sample
print(dataset["eval"][2])  # Third sample

# Import Ragas' EvaluationDataset and create an evaluation dataset
from ragas import EvaluationDataset

# Convert dataset to Ragas evaluation dataset
eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

# Define evaluator LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# Choose the metrics for evaluation
metrics = [LLMContextRecall(), FactualCorrectness(), Faithfulness()]

# Run the evaluation
results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)

# Export the evaluation results to a pandas DataFrame and display the first few rows
df = results.to_pandas()
print(df.head())
