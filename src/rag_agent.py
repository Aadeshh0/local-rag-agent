#rag_agent.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from config import model_settings
import time
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedRagAgent:
    """
    Optimized RAG Agent with connection 
    """

    def __init__(self):
        self._model_cache = {}
        self._chain_cache = {}

    def get_model(self, model_name: str):

        if model_name not in self._model_cache:
            logger.info(f'\nInitiazling model : {model_name}')
            self._model_cache[model_name] = OllamaLLM(
                model = model_name,
                temperature = model_settings['temperature'],
                top_p = model_settings['top_p'],
                repeat_penalty = model_settings['repeat_penalty']
            )
        return self._model_cache[model_name]
    
    def create_chain(self, model_name:str, prompt_template:str):
        cache_key = f'{model_name}_{hash(prompt_template)}'

        if cache_key not in self._chain_cache:
            logger.info(f'Creating new chain for {model_name}')
            model = self.get_model(model_name)
            prompt = ChatPromptTemplate.from_template(prompt_template)
            self._chain_cache[cache_key] = prompt | model

        return self._chain_cache[cache_key]
    
    def handle_question(self, chain, retriever, question:str):

        try:
            start_time = time.time()

            retrieval_start = time.time()
            relevant_docs = retriever.invoke(question) # calls the retriever
            retrieval_time = time.time() - retrieval_start

            context_start = time.time()
            if relevant_docs:
                # Limit context length for faster processing
                context = self._prepare_context(relevant_docs, max_length=1500)
            else:
                context = "No relevant reviews found."
            context_time = time.time() - context_start

            generation_start = time.time()
            result = chain.invoke({
                'reviews' : context,
                'question' : question
            })
            generation_time = time.time() - generation_start

            total_time = time.time() - start_time
            logger.info(f'Retrieval : {retrieval_time:.2f}s | Context: {context_time:.2f}s | Generation : {generation_time:.2f}s')

            return result
        except Exception as e:
            logger.error(f'Error handling question : {e}')
            return f'Sorry, I encountered an error : {str(e)}'
        
    def _prepare_context(self, docs, max_length: int = 1500):
        context_parts = []
        current_length = 0

        for doc in docs:
            content = doc.page_content

            if hasattr(doc, 'metadata') and doc.metadata:
                rating = doc.metadata.get('rating', 'N/A')
                content = f'[Rating : {rating}] {content}'
            
            if current_length + len(content) <= max_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                remaining_space = max_length - current_length
                if remaining_space > 100:
                    context_parts.append(content[:remaining_space] + "...")
                break
        
        return '\n\n'.join(context_parts)
    
_rag_agent_instance = None

def get_rag_agent():
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = OptimizedRagAgent()
    return _rag_agent_instance

def create_chain(model_name:str, prompt_template:str):
    agent = get_rag_agent()
    return agent.create_chain(model_name, prompt_template)

def handle_question(chain, retriever, question:str):
    agent = get_rag_agent()
    return agent.handle_question(chain, retriever, question)




