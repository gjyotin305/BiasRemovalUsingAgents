from PIL import Image
import torch
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)


class VisualAgent(object):
    def __init__(self, persona: str, temperature: int = 0) -> None:
        self.persona = persona
        self.temperature = temperature
    
    def run_agent(self, images: Image) -> str:
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")

        model.to("cuda:0")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": f"{self.persona}"}
                ]
            },
        ]
        prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        processor.tokenizer.padding_side = "left"
        inputs = processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=300)
        result_ = processor.decode(output[0], skip_special_tokens=True)

        decoded_out = str(result_).split("[/INST]")[-1]
        print("+"*100)
        print(decoded_out)
        print("+"*100)
        return decoded_out



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
        outputs = model.generate(inputs, max_new_tokens=300)
        
        result_ = tokenizer.decode(outputs[0])

        result_dict = str(result_).split("<|assistant|>")[-1]

        result_dict_ = str(result_dict).split("<|end|>")[0]


        return result_dict_