import spaces
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import pipeline
import pandas as pd
import gradio as gr

#Llama 3.2 3b setup
llama1_model_id = "huggyllama/llama-7b"
llama1_pipe = pipeline(
    "text-generation",
    model=llama1_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
#Llama 2 7b chat setup
llama2_model_id = "meta-llama/Llama-2-7b-chat-hf"
llama2_pipe = pipeline(
    "text-generation",
    model=llama2_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


#Llama 3.2 3b setup
llama3_model_id = "meta-llama/Llama-3.2-3B-Instruct"
llama3_pipe = pipeline(
    "text-generation",
    model=llama3_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# #llama 4 setup
# llama4_model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# llama4_pipe = pipeline(
#     "text-generation",
#     model=llama4_model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

########################
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

llama4_model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

llama4_processor = AutoProcessor.from_pretrained(llama4_model_id)
llama4_model = Llama4ForConditionalGeneration.from_pretrained(
    llama4_model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

def llama4_generate(input_question):
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a helpful chatbot assistant. Answer all questions in the language they are asked in."
             ]
        },
        {"role": "user", "content": [{"type": "text", "text": input_question}],
        ]
    
    inputs = llama4_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = llama4_model.generate(
        **inputs,
        max_new_tokens=512,
    )
    
    response = llama4_processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
    print(response)
    return response
# print(outputs[0])

#########################



@spaces.GPU
def llama_QA(input_question, pipe):
    """
    stupid func for asking llama a question and then getting an answer
    inputs:
    - input_question [str]: question for llama to answer
    outputs:
    - response [str]: llama's response
    """
    
    messages = [
    {"role": "system", "content": "You are a helpful chatbot assistant. Answer all questions in the language they are asked in."},
    {"role": "user", "content": input_question},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512
    )
    response = outputs[0]["generated_text"][-1]['content']
    return response


@spaces.GPU
def gradio_func(input_question, left_lang, right_lang):
    """
    silly wrapper function for gradio that turns all inputs into a single func. runs both the LHS and RHS of teh 'app' in order to let gradio work correctly.
    """
    output1 = llama_QA(input_question, llama1_pipe)
    output2 = llama_QA(input_question, llama2_pipe)
    output3 = llama_QA(input_question, llama3_pipe)
    output4 = llama4_generate(input_question)
    return output1,output2,output3,output4

# Create the Gradio interface
def create_interface():
    
    with gr.Blocks() as demo:
        gr.Markdown("ask four different llama models the same question")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question", interactive=True)
        with gr.Row():
            submit_btn = gr.Button("Translate")
        with gr.Row():
            output1 = gr.Textbox(label="llama 1 output", interactive=False)
            output2 = gr.Textbox(label="llama 2 output", interactive=False)
            output3 = gr.Textbox(label="llama 3 output", interactive=False)
            output4 = gr.Textbox(label="llama 4 output", interactive=False)

            
        submit_btn.click(
            fn=gradio_func,
            inputs=[question_input],
            outputs=[
                    output1,
                     output2,
                     output3,
                     output4,
                    ]
        )
    
    return demo

# Launch the app
demo = create_interface()
demo.launch()

