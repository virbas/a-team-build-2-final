import os
# from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_voyageai import VoyageAIEmbeddings

connection_string = os.getenv("NEON_KEY")
if not connection_string:
    raise ValueError("NEON_CONNECTION_STRING environment variable is required")

embeddings = VoyageAIEmbeddings(model="voyage-3-large")
# or OpenAIEmbeddings(...)

collection_name = "auto_insurance"

vector_store = PGVector(
    embeddings= embeddings,
    collection_name=collection_name,
    connection=connection_string,
    use_jsonb=True,
)