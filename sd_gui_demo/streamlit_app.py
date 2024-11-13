import streamlit as st
import random
from PIL import Image
import io
from stable_diffusion_demo import StableDiffusion, text_embedder

st.title("Image Generator App")

# Input fields
prompt = st.text_input("Prompt", value="").split(',')
negative_prompt = st.text_input("Negative Prompt", value="").split(',')
height = st.number_input("Height", min_value=1, max_value=1024, value=512)
width = st.number_input("Width", min_value=1, max_value=1024, value=512)
num_inference_steps = st.number_input("Number of Inference Steps", min_value=1, max_value=100, value=50)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=50.0, value=7.5)
seed = st.number_input("Seed", min_value=0, value=42)

# Generate button
if st.button("Generate Image"):
    # Generate the image
    
    generated_image = StableDiffusion(uncond_embeddings=negative_prompt, 
                                      text_embeddings=prompt, 
                                      height=height, 
                                      width=width, 
                                      num_inference_steps=num_inference_steps, 
                                      guidance_scale=guidance_scale, 
                                      seed=seed,
                                     )

    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_column_width=True)

# Note about the placeholder function
st.markdown("""
I made this in like 5 minutes don't look at me.
""")