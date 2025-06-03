from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from config import DATA_PATH, EMBEDDING_MODEL
try:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name = 'restaurant-reviews',
        persist_directory = DATA_PATH,
        embedding_function = embeddings
    )

    print(f"Number of documents in vector store : {len(vector_store.get()['documents'])}")

    retriever = vector_store.as_retriever(search_kwargs = {'k' : 3})
    question = "What do people say about the crust of the pizza?"

    docs = retriever.invoke(question)

    print('\n Retreived Docs : ')
    for i, doc in enumerate(docs):
        print(f'\n--- Document {i+1} ---')
        print(f'Content : {doc.page_content}')
        print(f'Metadata : {doc.metadata}')
except Exception as e:
    print(f'error occured : {e}')
