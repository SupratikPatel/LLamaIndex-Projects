import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import StorageContext,load_index_from_storage




load_dotenv()
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")
os.environ['OPENAI_API_KEY']=str(os.getenv("OPENAI_API_KEY"))

documents=SimpleDirectoryReader("data").load_data()
index=VectorStoreIndex.from_documents(documents,show_progress=True)
query_engine=index.as_query_engine()

retriever=VectorIndexRetriever(index=index,similarity_top_k=4)
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.80)

query_engine=RetrieverQueryEngine(retriever=retriever,
                                  node_postprocessors=[postprocessor])

response=query_engine.query("What is attention is all you need all about?")

pprint_response(response,show_source=True)
print(response)


PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are transformers?")
print(response)