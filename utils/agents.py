from PIL import Image
import torch
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer
)


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


class LanguageAgent(object):
    def __init__(self, persona: str, temperature: int = 0) -> None:
        self.persona = persona
        self.temperature = temperature
    

    def run_agent(self, query: str) -> str:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
        messages = [
            {
                "role": "system", 
                "content": f"{self.persona}"
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        outputs = model.generate(inputs, max_new_tokens=32)
        text = tokenizer.batch_decode(outputs)[0]
        return text