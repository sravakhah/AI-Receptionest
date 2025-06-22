from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# Load the same embeddings model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the local vector database with embedding function attached
db = Chroma(persist_directory="product_db", embedding_function=embedding)

llm = ChatOllama(model="mistral")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

while True:
    query = input("🧑 Ask something: ")
    if query.lower() in ['exit', 'quit']:
        break
    response = qa.run(query)
    print("🤖", response)
