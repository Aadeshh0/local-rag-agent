import time
import functools
from typing import Callable, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def time_function(func_name : str = None):
    """Decorator to time function execution"""
    def decorator(func : Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            exec_time = end_time - start_time

            name = func_name or func.__name__
            logger.info(f'{name} : {exec_time:.2f}s')
            return result 
        return wrapper
    return decorator

class PerformanceProfiler:
    """this class times code blocks"""
    def __init__(self, name : str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exec_type, exc_val, exc_tb):
        end_time = time.time()
        exec_time = end_time - self.start_time
        logger.info(f'{self.name} : {exec_time:.2f}s')

def benchmark_retrieval_and_generation(chain, retriever, question:str):
    print(f'\n Benchmarking : {question[:50]}')
    print('==' * 60)

    total_start = time.time()

    with PerformanceProfiler("Document Retrieval"):
        relevant_docs = retriever.invoke(question)

    with PerformanceProfiler("Context Preparation"):
        context = '\n'.join([doc.page_content for doc in relevant_docs])

    with PerformanceProfiler("LLM Generation"):
        result = chain.invoke({
            'reviews' : context,
            'question' : question
        })

    total_time = time.time() - total_start

    print(f'Total time : {total_time:.2f}s')
    print(f'Retrieved {len(relevant_docs)} documents')
    print(f'Context length: {len(context)} characters')
    print('=' * 60)

    return result

def quick_benchmark(questions: list, chain, retriever, iterations: int = 1):
    print('\n Quick Benchmark Results')
    print('-' * 50)

    total_times = []

    for i, question in enumerate(questions):
        print(f'\n Test {i+1}/{len(questions)}')

        times = []
        for iteration in range(iterations):
            start = time.time()
            _ = benchmark_retrieval_and_generation(chain, retriever, question)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        total_times.append(avg_time)
        print(f' Avergage Time : {avg_time:.2f}s')

    print(f'\nOverall Average : {sum(total_times)/len(total_times):.2f}s')

Test_questions = [
    "What are the best pizza places?",
    "Show me restaurants with 5-star ratings",
    "Which restaurants have the worst reviews?",
    "Tell me about Italian restaurants",
    "What do people say about the service?"
]

if __name__ == '__main__':
    from vector_config import create_vectorstore
    from rag_agent import create_chain
    from config import models, genie_template

    print("Setting up the components")
    retriever = create_vectorstore()
    chain = create_chain(models['llama1b'], genie_template)

    quick_benchmark(Test_questions[:2], chain, retriever)