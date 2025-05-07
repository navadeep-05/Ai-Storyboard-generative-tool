import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io
import base64
import re

# Device and model setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo"

# Initialize pipeline with error handling
try:
    pipe = DiffusionPipeline.from_pretrained(
        model_repo_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    pipe = pipe.to(device)
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}. Ensure model access on Hugging Face and sufficient GPU memory.")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 512  # Reduced for memory efficiency

def generate_storyboard(
    story_prompt,
    num_frames,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    color_scheme,
    art_style,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        # Validate inputs
        if not story_prompt:
            return "Error: Story prompt cannot be empty.", None, seed
        num_frames = int(num_frames)
        if num_frames < 1 or num_frames > 6:
            return "Error: Number of frames must be between 1 and 6.", None, seed
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            return f"Error: Image size must not exceed {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}.", None, seed

        # Break story into scenes
        scenes = re.split(r'[.!?]+', story_prompt)[:num_frames]
        scenes = [s.strip() for s in scenes if s.strip()]
        if len(scenes) < num_frames:
            scenes += [story_prompt] * (num_frames - len(scenes))

        # Set seed
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate images
        images = []
        character_prompt = "a pirate captain with red hair, wearing a black hat and brown coat"
        for scene in scenes:
            prompt = f"{character_prompt} in {scene}, {color_scheme} colors, {art_style} style, detailed background"
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            images.append(image)

        # Create storyboard grid
        rows = (num_frames + 1) // 2  # 2 images per row
        cols = 2
        canvas = Image.new('RGB', (cols * width, rows * height), 'white')
        for i, img in enumerate(images):
            row, col = i // cols, i % cols
            canvas.paste(img, (col * width, row * height))

        # Convert to bytes for download
        buffered = io.BytesIO()
        canvas.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return canvas, f'<a href="data:image/png;base64,{img_str}" download="storyboard.png" class="generate-btn">Download Storyboard</a>', seed

    except Exception as e:
        return f"Error: {str(e)}", None, seed

# Gradio interface
css = """
#col-container {
    margin: 0 auto;
    max-width: 800px;
}
.generate-btn {
    background-color: #3b82f6;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    text-decoration: none;
    display: inline-block;
}
.generate-btn:hover {
    background-color: #2563eb;
}
.error {
    color: #dc2626;
    font-weight: bold;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# AI-Powered Storyboard Generator")
        
        with gr.Row():
            story_prompt = gr.Textbox(
                label="Story Description",
                placeholder="E.g., A pirate captain sails to a mysterious island...",
                lines=3,
            )
            run_button = gr.Button("Generate Storyboard", variant="primary")

        num_frames = gr.Slider(
            label="Number of Frames",
            minimum=1,
            maximum=6,
            step=1,
            value=3,
        )
        
        result = gr.Image(label="Storyboard", show_label=True)
        download_link = gr.HTML(label="Download")

        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="E.g., blurry, low quality",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            color_scheme = gr.Dropdown(
                label="Color Scheme",
                choices=["Vibrant", "Warm", "Cool", "Monochrome"],
                value="Vibrant",
            )
            art_style = gr.Dropdown(
                label="Art Style",
                choices=["Comic Book", "Watercolor", "Anime", "Oil Painting"],
                value="Anime",
            )
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                value=0.0,  # SDXL-Turbo works best with low guidance
            )
            num_inference_steps = gr.Slider(
                label="Inference Steps",
                minimum=1,
                maximum=10,
                step=1,
                value=2,  # Optimized for SDXL-Turbo
            )

        gr.Examples(
            examples=[
                ["A pirate captain discovers a treasure island. Storms rage. A battle ensues.", 3, "", 0, True, 512, 512, 0.0, 2, "Vibrant", "Anime"],
                ["An astronaut lands on a red planet. Aliens greet him. A feast begins.", 4, "", 0, True, 512, 512, 0.0, 2, "Cool", "Watercolor"],
            ],
            inputs=[story_prompt, num_frames, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, color_scheme, art_style],
        )

    demo.load()
    gr.on(
        triggers=[run_button.click, story_prompt.submit],
        fn=generate_storyboard,
        inputs=[
            story_prompt,
            num_frames,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            color_scheme,
            art_style,
        ],
        outputs=[result, download_link, seed],
    )

if __name__ == "__main__":
    demo.launch()
