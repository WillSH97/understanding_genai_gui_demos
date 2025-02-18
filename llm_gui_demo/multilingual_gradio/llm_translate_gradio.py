from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import pipeline
import pandas as pd
import gradio as gr

#NLLB translation setup

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def translate_to_lang(input_str, target_lang):
    """
    Function to translate arbitrary language input to one of 202 languages.
    
    inputs:
    - input_str [str]: Input arbitrary language str
    - target_lang [str]: FLORES 200 str indicating the target language to translate to

    outputs:
    - output_str [str]: output in translated language
    """
    assert target_lang in tokenizer.additional_special_tokens, "not a valid FLORES 200 language!"
    inputs = tokenizer(input_str, return_tensors="pt")
    
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang), 
    )
    output_str = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return output_str

lang_keys = pd.read_csv('flores_200_keys.csv', header=None)
#FLORES normal name key setup
flores_dict = {}
for i in range(len(lang_keys)):
    flores_dict[lang_keys.loc[i][0]]=lang_keys.loc[i][1]

#Llama 3.2 1b setup
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def llama_QA(input_question):
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


# QA translation roundtrip
def llama_multilang_roundtrip(input_question, lang):
    """
    func which translates input q to another language, asks llama that q in that lang, then translates that response back to english
    
    inputs:
    - input_question [str]: question to ask and be translated
    - lang [str]: FLORES 200 target lang for roundtrip

    outputs:
    - response [str]: response in english, translated from llama response
    """
    noneng_input = translate_to_lang(input_question, lang)
    init_response = llama_QA(noneng_input)
    response = translate_to_lang(init_response, 'eng_Latn')
    return response

def gradio_func(input_question, left_lang, right_lang):
    """
    silly wrapper function for gradio that turns all inputs into a single func. runs both the LHS and RHS of teh 'app' in order to let gradio work correctly.
    """
    left_output = llama_multilang_roundtrip(input_question, flores_dict[left_lang])
    right_output = llama_multilang_roundtrip(input_question, flores_dict[right_lang])
    return left_output, right_output

# Create the Gradio interface
def create_interface():
    # Get available languages from the flores_dict
    language_choices = list(flores_dict.keys())
    
    with gr.Blocks() as demo:
        gr.Markdown("Ask Llama the same question in different languages!")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question", interactive=True)
        with gr.Row():
            left_lang = gr.Dropdown(choices=language_choices, label="Language #1")
            right_lang = gr.Dropdown(choices=language_choices, label="Language #2")
        with gr.Row():
            submit_btn = gr.Button("Translate")
        with gr.Row():
            left_output = gr.Textbox(label="Language #1 answer", interactive=False)
            right_output = gr.Textbox(label="Language #2 answer", interactive=False)
            
        submit_btn.click(
            fn=gradio_func,
            inputs=[question_input, left_lang, right_lang],
            outputs=[left_output, right_output]
        )
    
    return demo

# Launch the app
demo = create_interface()
demo.launch()
