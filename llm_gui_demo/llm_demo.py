import os
import copy

#local demo imports and config
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer, TextIteratorStreamer


# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
torch_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

torch_dtype = torch.float16 if torch_device in ["cuda", "mps"] else torch.float32

llama_model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", 
                                           #  quantization_config=quantization_config, 
                                           torch_dtype=torch_dtype, 
                                           device_map=torch_device)

llama_tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

streamer = TextStreamer(llama_tokenizer)


# def llama32_1b_streamchat(messages):
#     inputs = llama_tokenizer.apply_chat_template(messages, add_generation_prompt = True)
#     inputs = torch.tensor(inputs).to(torch_device).unsqueeze(0)
#     stream = llama_model.generate(inputs, streamer=streamer, max_new_tokens = 256)
#     return stream

llama32_1b_pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    # streamer = streamer,
)

def llama32_1b_chat(messages) -> str: 
    "simplifies pipeline output to only return generated text"
    outputs = llama32_1b_pipe(
        messages,
        max_new_tokens=512
    )
    return outputs[-1]['generated_text'][-1]['content']
    

