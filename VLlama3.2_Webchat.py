import gradio as gr
import torch
from PIL import Image
import ollama
import io
import base64

class LlamaWebChat:
    def __init__(self):
        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Image processing configurations
        self.max_image_size = (800, 800)  # Resize large images
    
    def process_image(self, image):
        """
        Preprocess uploaded image:
        - Resize large images
        - Convert to RGB
        - Return a base64-encoded string
        """
        if image is None:
            return None
        
        # Convert to PIL Image if not already
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Convert to RGB mode
        image = image.convert('RGB')
        
        # Resize if the image is too large
        image.thumbnail(self.max_image_size, Image.LANCZOS)
        
        # Convert image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_response(self, prompt, image=None):
        """
        Generate response using Ollama with optional image processing
        """
        try:
            # Prepare the message
            messages = [{"role": "user", "content": prompt}]
            
            # If an image is uploaded, include it in the payload
            if image is not None:
                messages[0]["images"] = [image]
            
            # Generate response using Ollama
            response = ollama.chat(
                model='llama3.2-vision',  # Ensure this matches your local model name
                messages=messages
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def create_interface(self):
        """
        Create Gradio interface
        """
        with gr.Blocks() as demo:
            with gr.Row():
                # Prompt input
                txt = gr.Textbox(label="Enter your prompt", lines=4)
                
                # Image upload
                img_input = gr.Image(type="filepath", label="Upload Image (Optional)")
            
            # Submit button
            btn = gr.Button("Generate Response")
            
            # Output textbox
            output = gr.Textbox(label="Response", lines=10)
            
            # Button click event
            btn.click(
                fn=lambda prompt, image: self.generate_response(
                    prompt, self.process_image(image) if image else None
                ),
                inputs=[txt, img_input],
                outputs=output
            )
        
        return demo

# Main execution
if __name__ == "__main__":
    # Initialize web chat
    webchat = LlamaWebChat()
    
    # Launch Gradio interface
    interface = webchat.create_interface()
    interface.launch(
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",  # Allow external access
        server_port=7860  # Customizable port
    )
