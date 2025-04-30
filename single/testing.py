from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Define local llama3.2 model
model = ChatOllama(
    base_url="http://192.168.100.3:11434",
    model="llama3.2:latest",
    temperature=0,
)

# Define output schema
class Essay(BaseModel):
    question: str = Field(description="Set up a question")
    answer: str = Field(description="Answer to resolve the question")

# Initialize JSON parser with the schema
parser = JsonOutputParser(pydantic_object=Essay)

# Define prompt with format instructions
prompt = PromptTemplate(
    template=(
        "You are a JSON API that returns a structured short essay question and answer in the following JSON format.\n"
        "{format_instructions}\n"
        "Respond only in JSON format.\n"
        "User query: {query}"
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Build the pipeline: Prompt → LLM → JSON Parser
chain = prompt | model | parser

# Run the chain with a specific query
result = chain.invoke({
    "query": "Generate a short essay question and answer about climate change."
})

# Print the structured result
print(result)
