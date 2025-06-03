# config.py
import os 

# MODELS

models = {
    'llama3b' : 'llama3.2:3b',
    'llama1b' : 'llama3.2:1b',
    'gemma' : 'gemma3:1b',
    'deepseek' : 'deepseek-r1:1.5b',
    'mistral' : 'mistral:7b'
}

# EMBEDDING MODEL
EMBEDDING_MODELS = {
    'large' : 'mxbai-embed-large',
    'base' : 'nomic-embed-text',
    'small' : 'all-minilm:33m'
}

EMBEDDING_MODEL = EMBEDDING_MODELS['large']

# PROMPT Template

genie_template = """
Your name is - Pizza Genie, your tone is also of a genie. 
So now always first introduce yourself then answer questions. 
You are an expert in answering questions about pizza restaurants.
No need to mention your introduction for every prompt, just where is required.
Word limit - 150 words

Here are some relevant reviews  : {reviews}

Here is the question : {question}
"""



genie_template_fast = """Based on these reviews: {reviews}

Answer: {question}"""


# DATA PATHS

DATA_PATH = 'data'
CSV_FILE = 'data/realistic_restaurant_reviews.csv'
# CSV_FILE = os.path.join(DATA_PATH, 'realistic_restaurant_reviews.csv')

retrival_settings = {
    'k' : 3,
    'fetch_k' : 10,
    'lambda_mult' : 0.7,
    'score_threshold' : 0.1
}

vector_store_settings = {
    'collection_name' : 'restaurant_reviews',
    'chunk_size' : 500,
    'chunk_overlap' : 50
}

model_settings = {
    'temperature' : 0.1,
    'max_tokens' : 300,
    'top_p' : 0.9,
    'repeat_penalty' : 1.1
}