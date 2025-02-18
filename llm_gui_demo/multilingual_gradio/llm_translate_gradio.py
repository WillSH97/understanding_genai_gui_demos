from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import pipeline

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


