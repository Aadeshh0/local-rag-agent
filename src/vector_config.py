# vector_config.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from config import CSV_FILE, DATA_PATH, EMBEDDING_MODEL, retrival_settings, vector_store_settings
import logging
import time

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)

_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f'Initializing instance : {EMBEDDING_MODEL}')
        _embeddings_instance = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings_instance

def load_data_from_csv(csv_path:str):
    logger.info(f'Loading data from {csv_path}')
    start_time = time.time()
    try:
        df = pd.read_csv('csv_path')
        logger.info(f'Loaded {len(df)} rows from csv file.')

        documents = []
        ids = []

        for i, row in df.iterrows():
            title = str(row.get('Title', '')).strip()
            review = str(row.get('Review', '')).strip()
            rating = row.get('Rating', 'N/A')
            date = row.get('Date', 'N/A')

            if not review or review.lower() in ['nan', 'none', '']:
                continue

            content = f'Restaurant: {title}\nReview: {review}'

            doc = Document(
                page_content = content,
                metadata = {
                    'title' : title,
                    'rating' : rating,
                    'date' : date,
                    'source' : 'csv',
                    'doc_id' : str(i)
                }
            )
            documents.append(doc)
            ids.append(str(i))

        load_time = time.time() - start_time
        logger.info(f'Document preparation: {load_time:.2f}s | Created {len(documents)} documents')

        return documents, ids
    except Exception as e:
        logger.error(f'Error occured  loading csv :{e}')
        return [], []
    





def load_data_from_csv(csv_path: str):
    """Optimized CSV loading with better document creation"""
    logger.info(f"📂 Loading data from: {csv_path}")
    start_time = time.time()
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded {len(df)} rows from CSV")
        
        documents = []
        ids = []

        for i, row in df.iterrows():
            # Create more informative content
            title = str(row.get('Title', '')).strip()
            review = str(row.get('Review', '')).strip()
            rating = row.get('Rating', 'N/A')
            date = row.get('Date', 'N/A')
            
            # Skip empty reviews
            if not review or review.lower() in ['nan', 'none', '']:
                continue
            
            # Create document with structured content
            content = f"Restaurant: {title}\nReview: {review}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'title': title,
                    'rating': rating, 
                    'date': date,
                    'source': 'csv',
                    'doc_id': str(i)
                }
            )
            documents.append(doc)
            ids.append(str(i))
        
        load_time = time.time() - start_time
        logger.info(f"⏱️ Document preparation: {load_time:.2f}s | Created {len(documents)} documents")
        
        return documents, ids
        
    except Exception as e:
        logger.error(f"❌ Error loading CSV: {e}")
        return [], []

def create_vectorstore():
    """Optimized vector store creation with better configuration"""
    logger.info("🚀 Setting up vector store...")
    start_time = time.time()
    
    embeddings = get_embeddings()
    
    # Use DATA_PATH directly
    csv_file_path = os.path.join(DATA_PATH, 'realistic_restaurant_reviews.csv')
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        logger.error(f"❌ CSV file not found: {csv_file_path}")
        logger.info(f"🔍 Current working directory: {os.getcwd()}")
        logger.info(f"🔍 Looking for file at: {os.path.abspath(csv_file_path)}")
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Check if vector store already exists
    chroma_file_path = os.path.join(DATA_PATH, 'chroma-collections.parquet')
    
    vector_store = Chroma(
        collection_name=vector_store_settings['collection_name'],
        persist_directory=DATA_PATH,
        embedding_function=embeddings
    )
    
    # Add documents if vector store is empty or doesn't exist
    if not os.path.exists(chroma_file_path) or vector_store._collection.count() == 0:
        logger.info("📦 Vector store empty, adding documents...")
        add_start = time.time()
        
        documents, ids = load_data_from_csv(csv_file_path)
        
        if documents:
            # Add documents in batches for better performance
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                logger.info(f"📝 Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            add_time = time.time() - add_start
            logger.info(f"✅ Documents added in {add_time:.2f}s")
        else:
            logger.warning("⚠️ No documents to add!")
    else:
        logger.info("✅ Using existing vector store")
    
    # Create optimized retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            'k': retrival_settings['k'],
            'fetch_k': retrival_settings['fetch_k'],
            'lambda_mult': retrival_settings['lambda_mult']
        }
    )
    
    setup_time = time.time() - start_time
    logger.info(f"⏱️ Vector store setup completed in {setup_time:.2f}s")
    
    return retriever










# def create_vectorstore():
#     logger.info('Setting up your vector store...')
#     start_time = time.time()

#     embeddings = get_embeddings()
#     db_location = os.path.dirname(CSV_FILE)

#     chroma_file_path = os.path.join(db_location, 'chroma-collections.parquet')

#     vector_store = Chroma(
#         collection_name = vector_store_settings['collection_name'],
#         persist_directory = db_location,
#         embedding_function = embeddings
#     )

#     if not os.path.exists(chroma_file_path) or vector_store._collection.count() == 0:
#         logger.info('Vector store empty, adding documents...')
#         add_start = time.time()

#         documents, ids = load_data_from_csv(CSV_FILE)

#         if documents:
#             batch_size = 50
#             for i in range(0, len(documents), batch_size):
#                 batch_docs = documents[i: i + batch_size]
#                 batch_ids = ids[i: i + batch_size]
#                 vector_store.add_documents(documents=batch_docs, ids = batch_ids)
#                 logger.info(f'Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}')

#             add_time = time.time() - add_start
#             logger.info(f'Documents added in {add_time:.2f}sec')
#         else:
#             logger.warning('NO DOCUMENTS TO ADD!')
#     else:
#         logger.info('USING EXISTING VECTOR STORE')

#     retriever = vector_store.as_retriever(
#         search_type = 'mmr',
#         search_kwargs = {
#             'k' : retrival_settings['k'],
#             'fetch_k' : retrival_settings['fetch_k'],
#             'lambda_mult' : retrival_settings['lambda_mult']
#         }
#     )

#     setup_time = time.time() - start_time
#     logger.info(f'Vector store setup completed in {setup_time:.2f}sec')

#     return retriever


# def create_filtered_retriever(rating_filter = None, date_filter = None):

#     logger.info('Creating filter retriever...')

#     embedings = get_embeddings()
#     db_location = os.path.dirname(CSV_FILE)

#     vector_store = Chroma(
#         collection_name = vector_store_settings['collection_name'],
#         persist_directory=db_location,
#         embedding_function=embedings
#     )

#     filter_conditions = {}

#     if rating_filter:
#         filter_conditions['rating'] = rating_filter
#     if date_filter:
#         filter_conditions['date'] = date_filter

#     search_kwargs = {
#         'k' : retrival_settings['k'],
#         'fetch_k' : retrival_settings['fetch_k'],
#         'lambda_mult' : retrival_settings['lambda_mult']
#     }

#     if filter_conditions:
#         search_kwargs['filter'] = filter_conditions
    
#     return vector_store.as_retriever(
#         search_type = 'mmr',
#         search_kwargs = search_kwargs
#     )










# # def load_data_from_csv(csv_path : str):
# #     df = pd.read_csv(csv_path)
# #     documents = []
# #     ids = []

# #     for i, row in df.iterrows():
# #         doc = Document(
# #             page_content = row['Title'] + " " + row['Review'],
# #             metadata = {'rating' : row['Rating'], 'date' : row['Date']},
# #             id = str(i)
# #         )
# #         documents.append(doc)
# #         ids.append(str(i))
# #     return documents, ids

# # def create_vectorstore():
# #     embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL)
# #     db_location = DATA_PATH
# #     # add_documents = not os.path.exists(os.path.join(DATA_PATH, "chroma-collections.parquet"))
# #     chroma_file_path = os.path.join(db_location, 'chroma-collections.parquet')

# #     vector_store = Chroma(
# #         collection_name = 'restaurant_reviews',
# #         persist_directory = db_location,
# #         embedding_function = embeddings
# #     )

# #     if not os.path.exists(chroma_file_path):
# #         print("📦 Adding documents to vector store...")

# #         csv_path = os.path.join(DATA_PATH, 'realistic_restaurant_reviews.csv')
# #         documents, ids = load_data_from_csv(csv_path)

# #         print(f"📝 Prepared {len(documents)} documents.\n")
# #         vector_store.add_documents(documents=documents, ids=ids)

# #     return vector_store.as_retriever(search_kwargs={'k':5})
