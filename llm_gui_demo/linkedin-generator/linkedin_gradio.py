from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import pipeline
import pandas as pd
import gradio as gr
import os
import copy
import spaces

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer, TextIteratorStreamer


# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
torch_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

torch_dtype = torch.float16 if torch_device in ["cuda", "mps"] else torch.float32

llama_model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", 
                                           #  quantization_config=quantization_config, 
                                           torch_dtype=torch_dtype, 
                                           device_map=torch_device,
                                            load_in_4bit=True) #for puny devices like mine.

llama_tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# streamer = TextStreamer(llama_tokenizer)

llama32_3b_pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    # streamer = streamer,
)



@spaces.GPU
def llama32_3b_chat(message) -> str: 
    "simplifies pipeline output to only return generated text"
    input_history = [{"role": "system", "content": """You are a machine that takes literally any text, and then turns it into a stereotypical linkedin post. Make sure to use phrases like "Excited for the opportunity", "Here's what X taught me about B2B sales", "grateful for my time at X", "excited for the new adventures at {insert company name}", and "professional growth mindset"
    """}]
    input_history.append({"role": "user", "content": message})
    ##add sth about context window here

    outputs = llama32_3b_pipe(
        input_history,
        max_new_tokens=512
    )
    return outputs[-1]['generated_text'][-1]['content']
    


# Create the Gradio interface
def create_interface():
    
    with gr.Blocks() as demo:
        gr.Markdown("""LinkedIn Post Generator - fixing the chat "head" to force a chat model to only generate linkedin posts
                    """)
        with gr.Row():
            text_input = gr.Textbox(label="input for Linkedin Post Generator", value = "Excited for my new opportunity at {x company} after {y} years at {z company}! To new adventures!")
        with gr.Row():
            submit_btn = gr.Button("Translate")
        with gr.Row():
            text_output = gr.Textbox(interactive=False)

        submit_btn.click(
            fn=llama32_3b_chat,
            inputs=[text_input],
            outputs=[text_output]
        )
    
    return demo

# Launch the app
demo = create_interface()
demo.launch()
