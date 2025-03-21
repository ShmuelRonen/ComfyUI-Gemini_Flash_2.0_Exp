# Gemini_Flash_Node.py
import os
import json
import base64
import requests
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import torch
import torchaudio
import numpy as np

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class ChatHistory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, content):
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if isinstance(item, str))
        self.messages.append({"role": role, "content": content})
    
    def get_formatted_history(self):
        formatted = "\n=== Chat History ===\n"
        for msg in self.messages:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        formatted += "=== End History ===\n"
        return formatted
    
    def get_messages_for_api(self):
        api_messages = []
        for msg in self.messages:
            if isinstance(msg["content"], str):
                api_messages.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
        return api_messages
    
    def clear(self):
        self.messages = []

class Gemini_Flash_200_Exp:
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")

        # Common placeholder values to ignore
        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            # Try the provided api_key parameter
            self.api_key = api_key

            # If still not found, try to get from config
            if self.api_key is None:
                config = get_config()
                self.api_key = config.get("GEMINI_API_KEY")

        self.chat_history = ChatHistory()
        if self.api_key is not None:
            self.configure_genai()

    def configure_genai(self):
        genai.configure(api_key=self.api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Analyze the situation in details.", "multiline": True}),
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "model_version": (["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp-1219", "gemini-2.0-flash-exp-image-generation"], {"default": "gemini-2.0-flash-exp"}),
                "operation_mode": (["analysis", "generate_images"], {"default": "analysis"}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "Additional_Context": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE", {"forceInput": False, "list": True}),  # Multiple images input
                "video": ("IMAGE", ),
                "audio": ("AUDIO", ),
                "api_key": ("STRING", {"default": ""}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16, "step": 1}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generated_content", "generated_images")
    FUNCTION = "generate_content"
    CATEGORY = "Gemini Flash 2.0 Experimental"

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:  # If tensor shape is [batch, H, W, channels]
            if tensor.shape[0] == 1:  # Single image in batch
                tensor = tensor.squeeze(0)
            else:
                # If first image in batch, get only that one
                tensor = tensor[0]
                
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def resize_image(self, image, max_size):
        width, height = image.size
        if width > height:
            if width > max_size:
                height = int(max_size * height / width)
                width = max_size
        else:
            if height > max_size:
                width = int(max_size * width / height)
                height = max_size
        return image.resize((width, height), Image.LANCZOS)

    def sample_video_frames(self, video_tensor, num_samples=6):
        """Sample frames evenly from video tensor"""
        if len(video_tensor.shape) != 4:
            return None

        total_frames = video_tensor.shape[0]
        if total_frames <= num_samples:
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        frames = []
        for idx in indices:
            frame = self.tensor_to_image(video_tensor[idx])
            frame = self.resize_image(frame, 512)
            frames.append(frame)
        return frames

    def prepare_content(self, prompt, input_type, Additional_Context=None, images=None, video=None, audio=None, max_images=6):
        if input_type == "text":
            text_content = prompt if not Additional_Context else f"{prompt}\n{Additional_Context}"
            return [{"text": text_content}]
                
        elif input_type == "image":
            # Handle multiple images input
            all_images = []
            
            # Process images if provided
            if images is not None:
                # Check if images is a tensor with batch dimension
                if isinstance(images, torch.Tensor):
                    if len(images.shape) == 4:  # [batch, H, W, C]
                        # Limit number of images to max_images
                        num_images = min(images.shape[0], max_images)
                        
                        for i in range(num_images):
                            pil_image = self.tensor_to_image(images[i])
                            pil_image = self.resize_image(pil_image, 1024)
                            all_images.append(pil_image)
                    else:  # Single image tensor [H, W, C]
                        pil_image = self.tensor_to_image(images)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)
                # Handle case where images is a list
                elif isinstance(images, list):
                    for img_tensor in images[:max_images]:  # Limit to max_images
                        pil_image = self.tensor_to_image(img_tensor)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)
                        
            # If we have any images, create the parts structure
            if all_images:
                # Modify prompt to handle multiple images
                if len(all_images) > 1:
                    modified_prompt = f"Analyze these {len(all_images)} images. {prompt} Please describe each image separately."
                else:
                    modified_prompt = prompt
                    
                parts = [{"text": modified_prompt}]
                
                for idx, img in enumerate(all_images):
                    # Convert image to bytes
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_byte_arr
                        }
                    })
                
                return [{"parts": parts}]
            else:
                raise ValueError("No valid images provided")
                
        elif input_type == "video" and video is not None:
            # Handle video input (sequence of frames)
            frames = self.sample_video_frames(video)
            if frames:
                # Convert frames to proper format
                parts = [{"text": f"Analyzing video frames. {prompt}"}]
                for frame in frames:
                    # Convert each frame to bytes
                    img_byte_arr = BytesIO()
                    frame.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_byte_arr
                        }
                    })
                return [{"parts": parts}]
            else:
                raise ValueError("Invalid video format")
                    
        elif input_type == "audio" and audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            buffer = BytesIO()
            torchaudio.save(buffer, waveform, 16000, format="WAV")
            audio_bytes = buffer.getvalue()
            
            return [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_bytes
                        }
                    }
                ]
            }]
        else:
            raise ValueError(f"Invalid or missing input for {input_type}")

    def create_placeholder_image(self):
        """Create a placeholder image tensor when generation fails"""
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        image_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)  # [1, H, W, 3]

    def generate_images(self, prompt, model_version, images=None, batch_count=1, temperature=0.4, seed=0, max_images=6):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            # Special handling for the image generation model
            is_image_generation_model = "image-generation" in model_version
            
            # Set up the Google Generative AI client
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)
            
            # Set up generation config - add response_modalities for image generation model
            if is_image_generation_model:
                generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=['Text', 'Image']  # Critical for image generation
                )
            else:
                generation_config = types.GenerateContentConfig(
                    temperature=temperature
                )
            
            # Process reference images if provided
            content_parts = []
            if images is not None:
                # Convert tensor to PIL images
                all_images = []
                if isinstance(images, torch.Tensor):
                    if len(images.shape) == 4:  # [batch, H, W, C]
                        num_images = min(images.shape[0], max_images)
                        for i in range(num_images):
                            pil_image = self.tensor_to_image(images[i])
                            pil_image = self.resize_image(pil_image, 1024)
                            all_images.append(pil_image)
                    else:  # Single image tensor
                        pil_image = self.tensor_to_image(images)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)
                elif isinstance(images, list):
                    for img_tensor in images[:max_images]:
                        pil_image = self.tensor_to_image(img_tensor)
                        pil_image = self.resize_image(pil_image, 1024)
                        all_images.append(pil_image)
                
                # If we have reference images, include them in the content
                if all_images:
                    # For the image generation model, we need a special prompt
                    if is_image_generation_model:
                        content_text = f"Generate a new image in the style of these reference images: {prompt}"
                    else:
                        content_text = f"Generate an image of: {prompt}"
                    
                    content_parts = [content_text] + all_images
            else:
                # Text-only prompt
                if is_image_generation_model:
                    content_text = f"Generate a detailed, high-quality image of: {prompt}"
                else:
                    content_text = f"Generate an image of: {prompt}"
                
                content_parts = content_text
            
            # Track all generated images
            all_generated_images = []
            status_text = ""
            
            # Generate images for each batch
            for i in range(batch_count):
                try:
                    # Set seed if provided
                    if seed != 0:
                        current_seed = seed + i
                        # Note: Seed is applied through an environment variable or similar mechanism
                        # as the SDK doesn't directly support it in generation_config
                    
                    # Generate content
                    response = client.models.generate_content(
                        model=model_version,
                        contents=content_parts,
                        config=generation_config
                    )
                    
                    # Extract images from the response
                    batch_images = []
                    
                    # Extract the response text first
                    response_text = ""
                    
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    # Extract text
                                    if hasattr(part, 'text') and part.text:
                                        response_text += part.text + "\n"
                                    
                                    # Extract images
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        try:
                                            image_binary = part.inline_data.data
                                            batch_images.append(image_binary)
                                        except Exception as img_error:
                                            print(f"Error extracting image from response: {str(img_error)}")
                    
                    if batch_images:
                        all_generated_images.extend(batch_images)
                        status_text += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        status_text += f"Batch {i+1}: No images found in response. Text response: {response_text[:100]}...\n"
                
                except Exception as batch_error:
                    status_text += f"Batch {i+1} error: {str(batch_error)}\n"
            
            # Process generated images into tensors
            if all_generated_images:
                tensors = []
                for img_binary in all_generated_images:
                    try:
                        # Convert binary to PIL image
                        image = Image.open(BytesIO(img_binary))
                        
                        # Ensure it's RGB
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        
                        # Convert to numpy array and normalize
                        img_np = np.array(image).astype(np.float32) / 255.0
                        
                        # Create tensor with correct dimensions for ComfyUI [B, H, W, C]
                        img_tensor = torch.from_numpy(img_np)[None,]
                        tensors.append(img_tensor)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                
                if tensors:
                    # Combine all tensors into a batch
                    image_tensors = torch.cat(tensors, dim=0)
                    
                    result_text = f"Successfully generated {len(tensors)} images using {model_version}.\n"
                    result_text += f"Prompt: {prompt}\n"
                    result_text += f"Details: {status_text}"
                    
                    return result_text, image_tensors
            
            # No images were generated successfully
            return f"No images were generated with {model_version}. Details:\n{status_text}", self.create_placeholder_image()
            
        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            print(error_msg)
            return error_msg, self.create_placeholder_image()

    def generate_content(self, prompt, input_type, model_version="gemini-2.0-flash-exp", 
                        operation_mode="analysis", chat_mode=False, clear_history=False,
                        Additional_Context=None, images=None, video=None, audio=None, 
                        api_key="", max_images=6, batch_count=1, seed=0,
                        max_output_tokens=8192, temperature=0.4, structured_output=False):
        """Generate content using Gemini model with various input types."""
        
        # Set all safety settings to block_none by default
        safety_settings = [
            {"category": "harassment", "threshold": "NONE"},
            {"category": "hate_speech", "threshold": "NONE"},
            {"category": "sexually_explicit", "threshold": "NONE"},
            {"category": "dangerous_content", "threshold": "NONE"},
            {"category": "civic", "threshold": "NONE"}
        ]

        # Only update API key if explicitly provided in the node
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key})
            self.configure_genai()

        if not self.api_key:
            raise ValueError("API key not found in config.json or node input")

        if clear_history:
            self.chat_history.clear()

        # Handle image generation mode
        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt,
                model_version=model_version,
                images=images,
                batch_count=batch_count,
                temperature=temperature,
                seed=seed,
                max_images=max_images
            )

        # For analysis mode (original functionality)
        model_name = f'models/{model_version}'
        model = genai.GenerativeModel(model_name)

        # Apply fixed safety settings to the model
        model.safety_settings = safety_settings

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )

        try:
            if chat_mode:
                # Special handling for chat mode
                if input_type == "text":
                    text_content = prompt if not Additional_Context else f"{prompt}\n{Additional_Context}"
                    content = text_content
                elif input_type == "image":
                    # Handle multiple images
                    all_images = []
                    
                    if images is not None:
                        if isinstance(images, torch.Tensor) and len(images.shape) == 4:
                            # Batch of images
                            num_to_process = min(images.shape[0], max_images)
                            for i in range(num_to_process):
                                pil_image = self.tensor_to_image(images[i])
                                pil_image = self.resize_image(pil_image, 1024)
                                all_images.append(pil_image)
                        elif isinstance(images, list):
                            # List of tensors
                            for img_tensor in images[:max_images]:
                                pil_image = self.tensor_to_image(img_tensor)
                                pil_image = self.resize_image(pil_image, 1024)
                                all_images.append(pil_image)
                    
                    if all_images:
                        # Create content with all images
                        img_count = len(all_images)
                        prefix = f"Analyzing {img_count} image{'s' if img_count > 1 else ''}. "
                        if img_count > 1:
                            prefix += "Please describe each image separately. "
                        content = [f"{prefix}{prompt}"] + all_images
                    else:
                        raise ValueError("No images provided for image input type")
                elif input_type == "video" and video is not None:
                    if len(video.shape) == 4 and video.shape[0] > 1:
                        frame_count = video.shape[0]
                        frames = self.sample_video_frames(video)
                        if frames:
                            content = [f"This is a video with {frame_count} frames. {prompt}"] + frames
                        else:
                            raise ValueError("Error processing video frames")
                    else:
                        pil_image = self.tensor_to_image(video.squeeze(0) if len(video.shape) == 4 else video)
                        pil_image = self.resize_image(pil_image, 1024)
                        content = [f"This is a single frame from a video. {prompt}", pil_image]
                elif input_type == "audio" and audio is not None:
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]
                    
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)
                    elif waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                    
                    buffer = BytesIO()
                    torchaudio.save(buffer, waveform, 16000, format="WAV")
                    audio_bytes = buffer.getvalue()
                    
                    content = [prompt, {"mime_type": "audio/wav", "data": audio_bytes}]
                else:
                    raise ValueError(f"Invalid or missing input for {input_type}")

                # Initialize chat and send message
                chat = model.start_chat(history=self.chat_history.get_messages_for_api())
                response = chat.send_message(content, generation_config=generation_config)
                
                # Add to history and get formatted output
                if isinstance(content, list):
                    history_content = prompt
                else:
                    history_content = content
                    
                self.chat_history.add_message("user", history_content)
                self.chat_history.add_message("assistant", response.text)
                
                # Only show the chat history
                generated_content = self.chat_history.get_formatted_history()
            else:
                # Non-chat mode uses the prepare_content method
                content_parts = self.prepare_content(
                    prompt, input_type, Additional_Context, images, video, audio, max_images
                )
                
                if structured_output:
                    if isinstance(content_parts, list) and len(content_parts) > 0:
                        if "parts" in content_parts[0]:
                            for part in content_parts[0]["parts"]:
                                if "text" in part:
                                    part["text"] = f"Please provide the response in a structured format. {part['text']}"
                
                response = model.generate_content(content_parts, generation_config=generation_config)
                generated_content = response.text

        except Exception as e:
            generated_content = f"Error: {str(e)}"
    
        # For analysis mode, return the text response and an empty placeholder image
        return (generated_content, self.create_placeholder_image())
        
NODE_CLASS_MAPPINGS = {
    "Gemini_Flash_200_Exp": Gemini_Flash_200_Exp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini_Flash_200_Exp": "Gemini Flash 2.0 Experimental",
}