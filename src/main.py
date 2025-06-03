# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_config import create_vectorstore
from rag_agent import create_chain, handle_question
from config import models, genie_template
import time 

def main():
    retriever = create_vectorstore()
    chain = create_chain(models['llama1b'], genie_template)

    try:
        while True:
            print()
            question = input("Ask your question : ")
            print()
            if question.lower().strip() in  ['bye', 'quit', 'kill']:
                break

            
            print('🧞‍♂️✨ Genie is brewing your solution... 🧪\n')
            result = handle_question(chain, retriever, question)
            print(result)
            print('\n------')

    except Exception as e:
        print(f'Error occured : {e}')

if __name__ == "__main__":
    main()
