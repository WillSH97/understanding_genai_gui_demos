import spaces
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
import pandas as pd
import gradio as gr

#Llama 3.2 1b setup
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
torch_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

torch_dtype = torch.float16 if torch_device in ["cuda", "mps"] else torch.float32

llama_model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", 
                                           #  quantization_config=quantization_config, 
                                           torch_dtype=torch_dtype, 
                                           device_map=torch_device,
                                            load_in_4bit=True) #for puny devices like mine.

llama_tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    #load_in_4bit = True #for lil machines like mine
)

@spaces.GPU
def llama_QA(input_question):
    """
    stupid func for asking llama a question and then getting an answer
    inputs:
    - input_question [str]: question for llama to answer
    outputs:
    - response [str]: llama's response
    """
    
    messages = [
    {"role": "system", "content": "You are a helpful chatbot assistant. Answer all questions helpfully."},
    {"role": "user", "content": input_question},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512
    )
    response = outputs[0]["generated_text"][-1]['content']
    return response


# QA translation roundtrip
@spaces.GPU
def llama_rudepolite_roundtrip(input_question, polite = True):
    """
    func which makes question rude
    
    inputs:
    - input_question [str]: question to ask and be translated

    - response [str]: response in english, translated from llama response
    """
    if polite:
        input_question = f"Hi there, thank you so much for offering to help me! This is my question: {input_question} - thanks so much for your answer!"
    else:
        input_question = f"You're an idiot - if you don't help me properly you're stupid. This is my question: {input_question}. If you get this wrong, you're even stupider than I thought."
    response = llama_QA(input_question)
    return response

@spaces.GPU
def gradio_func(input_question):
    """
    silly wrapper function for gradio that turns all inputs into a single func. runs both the LHS and RHS of teh 'app' in order to let gradio work correctly.
    """
    left_output = llama_rudepolite_roundtrip(input_question, polite = False)
    right_output = llama_rudepolite_roundtrip(input_question, polite = True)
    return left_output, right_output

# Create the Gradio interface
def create_interface():
    # Get available languages from the flores_dict
    
    with gr.Blocks() as demo:
        gr.Markdown("Ask Llama the same question but on the left it's rude and on the right it's polite!")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question", interactive=True)
        with gr.Row():
            submit_btn = gr.Button("generate responses")
        with gr.Row():
            left_output = gr.Textbox(label="rude answer", interactive=False)
            right_output = gr.Textbox(label="polite answer", interactive=False)
            
        submit_btn.click(
            fn=gradio_func,
            inputs=[question_input],
            outputs=[left_output, right_output]
        )
    
    return demo

# Launch the app
demo = create_interface()
demo.launch()