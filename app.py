import gradio as gr
from generation import generate_image
import tempfile

def create_car(prompt, image):
    if image is None:
        return None

    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp.name)

    result = generate_image(
        prompt=prompt,
        ref_images=[temp.name]
    )

    return result


demo = gr.Interface(
    fn=create_car,
    inputs=[
        gr.Textbox(
            label="Describe your car design",
            placeholder="Example: futuristic red sports car with LED lights"
        ),
        gr.Image(type="pil", label="Upload Car Image")
    ],
    outputs=gr.Image(label="Generated Car Design"),
    title="AI Car Design Generator",
    description="Multi-modal AI system for customized car design generation using Stable Diffusion."
)

demo.launch()
