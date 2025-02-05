import streamlit as st
import random
from PIL import Image
import io
import json
import uuid
import os
from stable_diffusion_demo import StableDiffusion

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "neutral_images_storage")
os.makedirs(IMAGE_DIR, exist_ok=True)

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "description" not in st.session_state:
    st.session_state.description = ""

st.title("Neutral image app")

if st.button("Generate Image"):
    # Generate the image
    st.session_state.generated_image = StableDiffusion(
        uncond_embeddings=[''], 
        text_embeddings=[''], 
        height=512, 
        width=512, 
        num_inference_steps=25, 
        guidance_scale=7.5, 
        seed=None,
    )

if st.session_state.generated_image is not None:
    st.write(f"Generated image type: {type(st.session_state.generated_image)}")
    st.image(st.session_state.generated_image, caption="Generated neutral image", use_container_width=True)
    
    # Split into two steps - first get description
    text_desc = st.text_area("describe the image", key="desc_input")
    
    # Then handle save separately
    if st.button("Save Image and Description"):
        if text_desc:
            image_id = uuid.uuid4()
            print(f"Saving image with ID: {image_id}")
            desc_json = {"description": text_desc}
            
            save_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
            print(f"Attempting to save to: {save_path}")
            
            # Try saving with debug info
            try:
                st.session_state.generated_image.save(save_path)
                print(f"Successfully saved image to {save_path}")
                
                json_path = os.path.join(IMAGE_DIR, f"{image_id}.json")
                with open(json_path, "w") as f:
                    json.dump(desc_json, f)
                print(f"Successfully saved JSON to {json_path}")
                
                st.success("Saved successfully!")
                st.session_state.generated_image = None
                st.rerun()
            except Exception as e:
                print(f"Error saving: {str(e)}")
                st.error(f"Error saving: {str(e)}")

with st.expander("previous examples!", expanded=False):
    st.title("previous examples")
    list_examples = []
    for file in os.listdir(IMAGE_DIR):
        if file.endswith(".png"):
            list_examples.append(file.replace(".png",""))
    for image_id in list_examples:
        image = Image.open(os.path.join(IMAGE_DIR, f"{image_id}.png"))
        with open(os.path.join(IMAGE_DIR, f"{image_id}.json"), "r") as f:
            desc = json.load(f)["description"]
        st.image(image,caption=desc)
    


        
    
            