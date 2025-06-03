# gradio_appy.py

import gradio as gr
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_config import create_vectorstore
from config import models, genie_template, EMBEDDING_MODEL
from rag_agent import create_chain, handle_question
import time 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing RAG components...")
startup_start = time.time()

retriever = create_vectorstore()
chain = create_chain(models['llama1b'], genie_template)

startup_time = time.time() - startup_start
logger.info(f'Startup completed in {startup_time:.2f}sec')

chat_history = []

def genie_chat(question, history = None):
    if not question or not question.strip():
        return 'Kindly ask me your query!'
    
    question = question.strip()

    if question.lower() in ['bye', 'exit', 'quit']:
        return 'Farewell, mortal. \nMay you feast on perfect pizza dishes.'
    
    try:
        start_time = time.time()
        logger.info('Processing : {question[:50]}...')

        result = handle_question(chain, retriever, question)

        response_time = time.time() - start_time
        logger.info(f'Response granted in {response_time:.2f}s')

        chat_history.append({
            'question' : question,
            'answer' : result,
            'timestamp' : time.strftime('%H:%M:%S'),
            'response_time' : f'{response_time:.2f}s'
        })

        if len(chat_history) > 10:
            chat_history.pop(0)

        return result
    except Exception as e:
        logging.error(f'Error occured : {e}')
        return f'Something went wrong : {str(e)}'
    
def get_chat_history():
    if not chat_history:
        return 'No conversation history yet.'
    
    history_text = '*' * 5 + 'Recent Conversations:' + '*' * 5 + '\n\n'
    for i, chat in enumerate(chat_history[-5:], 1):
        history_text += f"**{i}. [{chat['timestamp']}] ({chat['response_time']})**\n"
        history_text += f"Q: {chat['question'][:100]}...\n"
        history_text += f"A: {chat['answer'][:200]}...\n\n"

    return history_text

def clear_history():
    global chat_history
    chat_history = []
    return 'Chat history cleared!'

with gr.Blocks(
    title = 'Restaurant Genie - RAG App',
    theme = gr.themes.Soft(),
    css = """
    .container { max-width: 800px; margin: auto; }
    .header { text-align: center; padding: 20px; }
    .performance-info {
        background : #f4f4f4;
        padding : 12px;
        border-radius : 4px;
        margin : 12px 0;
        font-size : 12px;
    }
    """
) as interface:
    
    gr.Markdown("""
    # Restaurant Genie - RAG App
    ### Ask anything about restaurant reviews, ratings, or recommendations !
    ### It utilises Llama 3.2 Model with 1B params and Vector Search using Chroma
    """, elem_classes=['header']
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label = 'Ask the restaurant genie',
                placeholder='e.g. Which is the best pizza place?',
                lines = 2
            )
        
            with gr.Row():
                submit_btn = gr.Button('Ask Genie', variant='primary', scale=1)
                clear_btn = gr.Button('Clear history', scale=1)

            answer_output = gr.Textbox(
                label = "Genie's Reply",
                lines = 8,
                max_lines=15
            )
        
        with gr.Column(scale=1):
            gr.Markdown('### Performance Info')
            performance_info = gr.Textbox(
                label = 'Last Response Time',
                value = 'Ready to serve!',
                lines = 2,
                interactive=False
            )

            history_btn = gr.Button('Show history')
            history_output = gr.Textbox(
                label = 'Chat History',
                lines = 6,
                interactive= False
            )

            history_btn = gr.Button("📚 Show History")
            history_output = gr.Textbox(
                label="Chat History",
                lines=6,
                interactive=False
            )
    
    # Example questions
    gr.Markdown("""
    ### 💡 Example Questions:
    - "What are the highest rated restaurants?"
    - "Show me restaurants with poor service"
    - "Which places have the best pizza?"
    - "Tell me about Italian restaurants"
    - "What do people complain about most?"
    """)

    def submit_with_performance(question):
        start = time.time()
        response = genie_chat(question)
        duration = time.time() - start
        perf_info = f"Response Time: {duration:.2f}s\nModel: Llama 3.2 1B\nEmbedding: {EMBEDDING_MODEL}"
        return response, perf_info
    
    submit_btn.click(
        fn=submit_with_performance,
        inputs=[question_input],
        outputs=[answer_output, performance_info]
    )
    
    question_input.submit(
        fn=submit_with_performance,
        inputs=[question_input],
        outputs=[answer_output, performance_info]
    )
    
    history_btn.click(
        fn=get_chat_history,
        outputs=[history_output]
    )
    
    clear_btn.click(
        fn=clear_history,
        outputs=[history_output]
    )

if __name__ == '__main__':
    logger.info('Launching Gradio interfcae...')
    interface.launch(
        share = True,
        server_name = '0.0.0.0',
        server_port=7860,
        show_error=True
    )

