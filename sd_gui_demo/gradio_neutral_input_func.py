import gradio as gr
import random
from PIL import Image
import io
import json
import uuid
import os
from stable_diffusion_demo import StableDiffusion

# Setup directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "neutral_images_storage")
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_image():
    """Generate a neutral image using Stable Diffusion"""
    generated_image = StableDiffusion(
        uncond_embeddings=[''],
        text_embeddings=[''],
        height=512,
        width=512,
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=None,
    )
    return generated_image

def save_image_and_description(image, description):
    """Save the generated image and its description"""
    if image is None:
        return "No image to save!", None, None
    
    if not description:
        return "Please provide a description!", None, None
        
    try:
        image_id = uuid.uuid4()
        save_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
        json_path = os.path.join(IMAGE_DIR, f"{image_id}.json")
        
        # Save image
        image.save(save_path)
        
        # Save description
        desc_json = {"description": description}
        with open(json_path, "w") as f:
            json.dump(desc_json, f)
        
        # Return success message, clear the image output, and return updated gallery
        return "Saved successfully!", None, load_previous_examples()
    except Exception as e:
        return f"Error saving: {str(e)}", None, None

def load_previous_examples():
    """Load all previously saved images and descriptions"""
    examples = []
    for file in os.listdir(IMAGE_DIR):
        if file.endswith(".png"):
            image_id = file.replace(".png", "")
            image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
            json_path = os.path.join(IMAGE_DIR, f"{image_id}.json")
            
            if os.path.exists(json_path):
                image = Image.open(image_path)
                with open(json_path, "r") as f:
                    desc = json.load(f)["description"]
                examples.append((image, desc))
    return examples

# Create the Gradio interface
with gr.Blocks(title="Neutral Image App") as demo:
    gr.Markdown("# Neutral Image App")
    
    with gr.Row():
        with gr.Column():
            generate_btn = gr.Button("Generate Image")
            # Disable image upload by setting interactive=False
            image_output = gr.Image(type="pil", label="Generated Image", interactive=False)
            description_input = gr.Textbox(label="Describe the image", lines=3)
            save_btn = gr.Button("Save Image and Description")
            status_output = gr.Textbox(label="Status")
    
    with gr.Accordion("Previous Examples", open=False):
        gallery = gr.Gallery(
            label="Previous Images",
            show_label=True,
            elem_id="gallery"
        )#.style(grid=2, height="auto")
    
    # Set up event handlers
    generate_btn.click(
        fn=generate_image,
        outputs=[image_output]
    )
    
    # Updated to include gallery refresh in outputs
    save_btn.click(
        fn=save_image_and_description,
        inputs=[image_output, description_input],
        outputs=[status_output, image_output, gallery]  # Added gallery to outputs
    )
    
    # Load previous examples on startup
    demo.load(
        fn=load_previous_examples,
        outputs=[gallery]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()