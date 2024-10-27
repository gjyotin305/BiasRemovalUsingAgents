from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class VisualAgent(object):
    def __init__(self, persona: str, temperature: int = 0) -> None:
        self.persona = persona
        self.temperature = temperature
    
    def run_agent(self, image: Image) -> str:
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        model.to("cuda:0")
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{self.persona}"}
                ]
            },
        ]
        prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=100)
        result = processor.decode(output[0], skip_special_tokens=True)
        return result