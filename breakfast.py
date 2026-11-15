
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables
load_dotenv()

# 2. Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 3. Initialize ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

# 4. Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Suggest exactly 5 healthy breakfast ideas.

    Follow these formatting rules:
    - Use numbered points from 1 to 5.
    - Write each idea on exactly one line.
    - Do not include any introduction or preamble.
    - Keep each idea simple and practical.
    """
)

# 5. Create LangChain chain
chain = prompt | llm

# 6. Invoke the model
response = chain.invoke({})

# 7. Print only the response text
print(response.content)
