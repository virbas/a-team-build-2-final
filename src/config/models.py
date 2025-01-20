from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
