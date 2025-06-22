#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import CSVLoader

loader = CSVLoader(file_path="products.csv")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="product_db"
)

db.persist()
print("✅ Product knowledge embedded locally!")
