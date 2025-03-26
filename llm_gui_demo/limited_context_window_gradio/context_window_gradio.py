'''
I need:
- a slider at the top
- chat window
'''




from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import pipeline
import pandas as pd
import gradio as gr
import os
import copy

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer, TextIteratorStreamer


# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
torch_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

torch_dtype = torch.float16 if torch_device in ["cuda", "mps"] else torch.float32

llama_model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", 
                                           #  quantization_config=quantization_config, 
                                           torch_dtype=torch_dtype, 
                                           device_map=torch_device)

llama_tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# streamer = TextStreamer(llama_tokenizer)

llama32_1b_pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    # streamer = streamer,
)

def context_window_limiting(history: list[dict], context_window: int):
    '''
    cull full messages until you have the desired context length

    TO DO
    '''
    
    return

def llama32_1b_chat(message, history, context_window) -> str: 
    "simplifies pipeline output to only return generated text"
    input_history = copy.deepcopy(history)
    input_history.append({"role": "user", "content": message})
    
    ##add sth about context window here

    outputs = llama32_1b_pipe(
        input_history,
        max_new_tokens=512
    )
    return outputs[-1]['generated_text'][-1]['content']
    


# Create the Gradio interface
def create_interface():
    
    with gr.Blocks() as demo:
        gr.Markdown("change context window lmao")
        with gr.Row():
            context_window = gr.Slider(32, 512, value=256, label="size of context window", info="choose context window size")
        with gr.Row():
            gr.ChatInterface(fn=llama32_1b_chat, additional_inputs = [context_window], type="messages", title="context_window")
    
    return demo

# Launch the app
demo = create_interface()
demo.launch()
